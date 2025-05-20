import os
import random
import pandas as pd
import json
import re
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Literal, Optional, Tuple, Dict, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from fuzzywuzzy import process, fuzz
import time

# Load environment variables
load_dotenv()
print("Environment variables loaded.")

LYZR_API_URL = os.getenv('LYZR_API_URL')
LYZR_API_KEY = os.getenv('LYZR_API_KEY')
LYZR_USER_ID = os.getenv('LYZR_USER_ID')
LYZR_AGENT_ID = os.getenv('LYZR_AGENT_ID')
MAX_ROWS = int(os.getenv('MAX_ROWS', 50000000000))
TOP_K = 10
DATA_SOURCE = os.getenv('DATA_SOURCE', 'file')

print(f"Configuration: LYZR_API_URL={LYZR_API_URL}, LYZR_USER_ID={LYZR_USER_ID}, LYZR_AGENT_ID={LYZR_AGENT_ID}, MAX_ROWS={MAX_ROWS}, TOP_K={TOP_K}, DATA_SOURCE={DATA_SOURCE}")

# Global variables
DB_ENGINE = None
SESSION_STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan handler defined and called.")  # Debug log to confirm handler is called
    # Startup logic
    global DB_ENGINE
    print(f"DATA_SOURCE is set to: {DATA_SOURCE}")
    # Since you previously showed successful database usage, let's initialize the database regardless of DATA_SOURCE
    # If you still need the DATA_SOURCE logic, you can reintroduce the condition
    try:
        db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        print(f"Initializing RDS database connection with URL: {db_url}")
        DB_ENGINE = create_engine(db_url, poolclass=QueuePool, pool_size=5, max_overflow=10)
        print("RDS database connection pool initialized successfully.", DB_ENGINE)
    except Exception as e:
        print(f"Error initializing RDS database connection: {str(e)}")
        raise
    yield  # This yields control back to FastAPI during the app's lifetime
    # Shutdown logic
    print("Lifespan handler shutting down.")
    if DB_ENGINE:
        print("Closing database connection pool.")
        DB_ENGINE.dispose()
        print("Database connection pool closed.")

# Debug log to confirm lifespan function is defined
print("Lifespan function defined.")

app = FastAPI(title="Zwende Search Agent", version="1.0.0", lifespan=lifespan)
print("FastAPI app initialized.")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS middleware configured.")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class Product(BaseModel):
    Handle: str
    Variant_ID: str
    Title: str
    Category_Name: Optional[str] = None
    collection_title: Optional[str] = None
    Variant_Price: float
    URL: str
    total_sold: Optional[float] = None
    total_in_collection: Optional[int] = None

def get_product_metadata(category: Optional[str] = None, limit_categories: Optional[List[str]] = None, product_type: Optional[str] = None, theme: Optional[str] = None, max_products: int = 50) -> str:
    start_time = time.time()
    print(f"[{start_time}] Fetching product metadata: category={category}, limit_categories={limit_categories}, product_type={product_type}, theme={theme}, max_products={max_products}")
    
    if DATA_SOURCE == 'database':
        if DB_ENGINE is None:
            print("Database connection not initialized.")
            end_time = time.time()
            print(f"[{end_time}] get_product_metadata completed in {end_time - start_time:.3f} seconds")
            return "Database connection not initialized."
        try:
            query = """
                SELECT 
                    p."Handle", 
                    p."Variant ID", 
                    p."Title", 
                    p."Product Description", 
                    p."Type", 
                    p."Tags", 
                    p."Category: Name", 
                    p."Variant Price", 
                    p."Image Src", 
                    COALESCE(SUM(o."Price: Total Line Items")::float, 0) as total_sold
                FROM products p
                LEFT JOIN orders o ON p."Handle" = o."Line: Product Handle" AND p."Variant ID" = o."Line: Variant ID"
                WHERE p."Variant ID" IS NOT NULL AND p."Variant Price" IS NOT NULL
            """
            params = {}
            if category:
                query += ' AND LOWER(p."Category: Name") = LOWER(%(category)s)'
                params['category'] = category
            elif limit_categories:
                query += ' AND LOWER(p."Category: Name") = ANY(%(categories)s)'
                params['categories'] = [c.lower() for c in limit_categories]
            else:
                category_query = 'SELECT DISTINCT "Category: Name" FROM products WHERE "Category: Name" IS NOT NULL'
                print(f"Executing category query: {category_query}")
                categories_df = pd.read_sql_query(category_query, DB_ENGINE)
                all_categories = categories_df['Category: Name'].tolist()
                sampled_categories = random.sample(all_categories, min(20, len(all_categories)))
                query += ' AND LOWER(p."Category: Name") = ANY(%(categories)s)'
                params['categories'] = [c.lower() for c in sampled_categories]
                print(f"Sampled categories: {sampled_categories}")

            if product_type:
                query += ' AND (LOWER(p."Title") LIKE %(product_type)s OR LOWER(p."Tags") LIKE %(product_type)s OR LOWER(p."Product Description") LIKE %(product_type)s)'
                params['product_type'] = f'%{product_type.lower()}%'
            if theme:
                query += ' AND (LOWER(p."Tags") LIKE %(theme)s OR LOWER(p."Title") LIKE %(theme)s OR LOWER(p."Product Description") LIKE %(theme)s)'
                params['theme'] = f'%{theme.lower()}%'

            query += f' GROUP BY p."Handle", p."Variant ID", p."Title", p."Product Description", p."Type", p."Tags", p."Category: Name", p."Variant Price", p."Image Src" LIMIT {max_products}'
            print(f"Executing product query: {query} with params: {params}")
            subset = pd.read_sql_query(query, DB_ENGINE, params=params)
            print(f"Product query result: {len(subset)} rows")

            smart_query = """
                SELECT "Product: Handle", "Title" as collection_title, "Rule: Product Column", "Rule: Condition", "Body HTML", "Sort Order", "Published"
                FROM smart_collections
                WHERE "Product: Handle" = ANY(%(handles)s) AND "Product: Handle" IS NOT NULL
            """
            print(f"Executing smart collection query: {smart_query} with handles: {subset['Handle'].tolist()}")
            smart_subset = pd.read_sql_query(smart_query, DB_ENGINE, params={'handles': subset['Handle'].tolist()})
            print(f"Smart collection query result: {len(smart_subset)} rows")
        except Exception as e:
            print(f"Error querying product metadata: {str(e)}")
            end_time = time.time()
            print(f"[{end_time}] get_product_metadata completed in {end_time - start_time:.3f} seconds")
            return f"Error fetching product metadata: {str(e)}"
    if subset.empty:
        print("No products found for the specified criteria.")
        end_time = time.time()
        print(f"[{end_time}] get_product_metadata completed in {end_time - start_time:.3f} seconds")
        return f"No products found for the specified criteria."

    metadata = []
    for _, row in subset.iterrows():
        product_info = {
            "Handle": row.get('Handle', ''),
            "Variant ID": row.get('Variant ID', ''),
            "Title": row.get('Title', ''),
            "Product Description": row.get('Product Description', '')[:500],
            "Type": row.get('Type', ''),
            "Tags": row.get('Tags', ''),
            "Category: Name": row.get('Category: Name', row.get('Product Category', 'Uncategorized')),
            "Variant Price": float(row.get('Variant Price', 0.0)),
            "Image Src": row.get('Image Src', ''),
            "total_sold": float(row.get('total_sold', 0)) if pd.notnull(row.get('total_sold')) else 0
        }
        if category and 'kids' not in category.lower() and ('kids' in str(product_info['Tags']).lower() or 'kids' in str(product_info['Title']).lower()):
            continue
        metadata.append(product_info)
    print(f"Processed {len(metadata)} product metadata entries")

    for _, row in smart_subset.iterrows():
        collection_info = {
            "Collection Title": row.get('collection_title', ''),
            "Rule: Product Column": row.get('Rule: Product Column', ''),
            "Rule: Condition": row.get('Rule: Condition', ''),
            "Body HTML": row.get('Body HTML', ''),
            "Sort Order": row.get('Sort Order', ''),
            "Published": bool(row.get('Published', False)),
            "Product: Handle": row.get('Product: Handle', '')
        }
        for product in metadata:
            if product["Handle"] == collection_info["Product: Handle"]:
                product.update(collection_info)
    print(f"Updated metadata with smart collection info")

    metadata_str = ""
    for i, product in enumerate(metadata[:max_products], 1):
        metadata_str += f"Product {i}:\n"
        for key, value in product.items():
            if value and isinstance(value, (str, float, int, bool)):
                metadata_str += f"  {key}: {value}\n"
        metadata_str += "\n"
    print(f"Generated metadata string: {metadata_str[:500]}... (truncated)")

    end_time = time.time()
    print(f"[{end_time}] get_product_metadata completed in {end_time - start_time:.3f} seconds")
    return metadata_str if metadata_str else "No metadata available."

@app.get("/categories", response_model=List[str])
def get_categories():
    
    print("Fetching categories",DB_ENGINE)
    if DB_ENGINE is None:
        print("Database connection not initialized.")
        raise HTTPException(status_code=500, detail="Database connection not initialized.")
    try:
        query = 'SELECT DISTINCT "Category: Name" FROM products WHERE "Category: Name" IS NOT NULL'
        print(f"Executing categories query: {query}")
        categories_df = pd.read_sql_query(query, DB_ENGINE)
        categories = categories_df['Category: Name'].tolist()
        print(f"Retrieved categories: {categories}")
        return JSONResponse(content=sorted(categories))
    except Exception as e:
        print(f"Error fetching categories from RDS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching categories from RDS: {str(e)}")

@app.get("/search/{source}/{category}", response_model=List[Product])
def search(source: Literal['orders', 'collections'], category: str):
    print(f"Search request: source={source}, category={category}")
    try:
        if DATA_SOURCE == 'database':
            if DB_ENGINE is None:
                print("Database connection not initialized.")
                raise HTTPException(status_code=500, detail="Database connection not initialized.")
            if source == 'orders':
                query = """
                    SELECT 
                        p."Handle", 
                        p."Variant ID" AS "Variant_ID", 
                        p."Title", 
                        p."Category: Name" AS Category_Name, 
                        p."Variant Price" AS Variant_Price, 
                        p."URL", 
                        COALESCE(SUM(o."Price: Total Line Items")::float, 0) AS total_sold
                    FROM products p
                    LEFT JOIN orders o ON p."Handle" = o."Line: Product Handle" AND p."Variant ID" = o."Line: Variant ID"
                    WHERE LOWER(p."Category: Name") = LOWER(%(category)s)
                        AND p."Variant ID" IS NOT NULL 
                        AND p."Variant Price" IS NOT NULL
                    GROUP BY p."Handle", p."Variant ID", p."Title", p."Category: Name", p."Variant Price", p."URL"
                    ORDER BY total_sold DESC
                    LIMIT %(max_rows)s
                """
            else:
                query = """
                    SELECT 
                        p."Handle", 
                        p."Variant ID" AS "Variant_ID", 
                        p."Title", 
                        sc."Title" AS collection_title, 
                        p."Variant Price" AS Variant_Price, 
                        p."URL", 
                        COUNT(*) AS total_in_collection
                    FROM products p
                    INNER JOIN smart_collections sc ON p."Handle" = sc."Product: Handle"
                    WHERE LOWER(sc."Title") = LOWER(%(category)s)
                        AND p."Variant ID" IS NOT NULL 
                        AND p."Variant Price" IS NOT NULL
                        AND sc."Product: Handle" IS NOT NULL
                    GROUP BY p."Handle", p."Variant ID", p."Title", sc."Title", p."Variant Price", p."URL"
                    ORDER BY total_in_collection DESC
                    LIMIT %(top_k)s
                """
            params = {'category': category, 'max_rows': MAX_ROWS, 'top_k': TOP_K}
            print(f"Executing search query: {query} with params: {params}")
            df = pd.read_sql_query(query, DB_ENGINE, params=params)
            print(f"Search query result: {len(df)} rows")
        # else:
        #     data_dir = os.getenv('DATA_DIR', './data')
        #     product_file = os.path.join(data_dir, os.getenv('PRODUCT_FILE', 'ProductExport230425.xlsx'))
        #     orders_file = os.path.join(data_dir, os.getenv('ORDER_FILE', 'OrderExport230425.xlsx'))
        #     smart_file = os.path.join(data_dir, os.getenv('SMART_FILE', 'Smart Collection Export.csv'))

        #     print(f"Loading files: products={product_file}, orders={orders_file}, smart={smart_file}")
        #     products = pd.read_excel(product_file)
        #     orders = pd.read_excel(orders_file)
        #     smart = pd.read_csv(smart_file)
        #     print(f"File data loaded: products={len(products)} rows, orders={len(orders)} rows, smart={len(smart)} rows")

        #     category_cols = [c for c in products.columns if 'category' in c.lower()]
        #     category_col = category_cols[0] if category_cols else 'Product Category'
        #     print(f"Selected category column: {category_col}")

        #     products = products[products['Variant ID'].notnull() & products['Variant Price'].notnull()]
        #     smart = smart[smart['Product: Handle'].notnull()]
        #     print(f"Filtered products (non-null Variant ID and Price): {len(products)} rows, smart: {len(smart)} rows")

        #     if source == 'orders':
        #         df = products.merge(
        #             orders.groupby(['Line: Product Handle', 'Line: Variant ID'])['Price: Total Line Items'].sum().reset_index(name='total_sold'),
        #             left_on=['Handle', 'Variant ID'],
        #             right_on=['Line: Product Handle', 'Line: Variant ID'],
        #             how='left'
        #         ).fillna({'total_sold': 0})
        #         df = df[df[category_col].str.lower() == category.lower()]
        #         df = df.sort_values('total_sold', ascending=False).head(MAX_ROWS)
        #         df = df.rename(columns={
        #             'Variant ID': 'Variant_ID',
        #             'Category: Name': 'Category_Name',
        #             'Variant Price': 'Variant_Price'
        #         })
        #         print(f"Orders search result: {len(df)} rows")
        #     else:
        #         df = products.merge(
        #             smart.rename(columns={'Title': 'collection_title'}),
        #             left_on='Handle',
        #             right_on='Product: Handle',
        #             how='inner'
        #         )
        #         df = df[df['collection_title'].str.lower() == category.lower()]
        #         df = df.groupby(['Handle', 'Variant ID', 'Title', 'collection_title', 'Variant Price', 'URL']).size().reset_index(name='total_in_collection')
        #         df = df.sort_values('total_in_collection', ascending=False).head(TOP_K)
        #         df = df.rename(columns={
        #             'Variant ID': 'Variant_ID',
        #             'Variant Price': 'Variant_Price'
        #         })
        #         print(f"Collections search result: {len(df)} rows")

        if df.empty:
            print(f"No products found for {source} - {category}")
            raise HTTPException(404, detail=f"No products found for {source} – {category}")

        records = []
        for i, record in enumerate(df.to_dict(orient='records')):
            sanitized_record = {
                "Handle": str(record.get('Handle', f"default-product-{i}")),
                "Variant_ID": str(record.get('Variant_ID', f"default-variant-{i}")),
                "Title": str(record.get('Title', f"Default Product {i}")),
                "Category_Name": str(record.get('Category_Name', '')) or None,
                "collection_title": str(record.get('collection_title', '')) or None,
                "Variant_Price": float(record.get('Variant_Price', 0.0)),
                "URL": str(record.get('URL', 'https://example.com')),
                "total_sold": float(record.get('total_sold', 0.0)) if record.get('total_sold') is not None else None,
                "total_in_collection": int(record.get('total_in_collection', 0)) if record.get('total_in_collection') is not None else None
            }
            records.append(sanitized_record)
        print(f"Sanitized {len(records)} search records: {records[:2]}... (truncated)")
        return records

    except Exception as e:
        print(f"Error processing search request: {str(e)}")
        raise HTTPException(500, detail=f"Error processing request: {str(e)}")

def call_lyzr_api(payload: dict, headers: dict) -> Union[dict, str]:
    start_time = time.time()
    print(f"[{start_time}] Calling Lyzr API with payload: {json.dumps(payload, indent=2)}")
    try:
        response = requests.post(LYZR_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        print(f"[{time.time()}] Lyzr API response: {json.dumps(response_json, indent=2)}")
        
        agent_response = response_json.get("response")
        if not agent_response:
            print("No response field in Lyzr API output")
            end_time = time.time()
            print(f"[{end_time}] call_lyzr_api completed in {end_time - start_time:.3f} seconds")
            return None
        
        if isinstance(agent_response, str):
            cleaned_response = re.sub(r'^```json\n|```$', '', agent_response, flags=re.MULTILINE).strip()
            try:
                parsed_response = json.loads(cleaned_response)
                print(f"[{time.time()}] Parsed JSON response: {json.dumps(parsed_response, indent=2)}")
                end_time = time.time()
                print(f"[{end_time}] call_lyzr_api completed in {end_time - start_time:.3f} seconds")
                return parsed_response
            except json.JSONDecodeError:
                print(f"Response is not JSON, treating as plain text: {cleaned_response}")
                end_time = time.time()
                print(f"[{end_time}] call_lyzr_api completed in {end_time - start_time:.3f} seconds")
                return cleaned_response
        
        end_time = time.time()
        print(f"[{end_time}] call_lyzr_api completed in {end_time - start_time:.3f} seconds")
        return agent_response
    except requests.RequestException as e:
        print(f"Error communicating with Lyzr API: {str(e)}")
        end_time = time.time()
        print(f"[{end_time}] call_lyzr_api completed in {end_time - start_time:.3f} seconds")
        raise HTTPException(status_code=500, detail=f"Error communicating with Lyzr agent: {str(e)}")
    except ValueError as e:
        print(f"Failed to parse Lyzr API response: {str(e)}")
        end_time = time.time()
        print(f"[{end_time}] call_lyzr_api completed in {end_time - start_time:.3f} seconds")
        raise HTTPException(status_code=500, detail=f"Failed to parse Lyzr API response: {str(e)}")

def detect_intent(message: str, session: dict, headers: dict) -> str:
    start_time = time.time()
    print(f"[{start_time}] Detecting intent for message: {message}, session: {session}")
    
    message_lower = message.lower().strip()
    product_type = session.get('product_type', '') or ''
    category = session.get('category', '') or ''
    theme = session.get('theme', '') or ''
    conversation_history = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in zip(session.get('questions_asked', []), session.get('user_responses', []))
    ) or "No conversation history available."
    print(f"Conversation history: {conversation_history}")

    # Get available categories
    try:
        categories = json.loads(get_categories().body.decode('utf-8'))
        categories_str = ", ".join(categories) if categories else "No categories available."
        print(f"Available categories: {categories_str}")
    except Exception as e:
        print(f"Error fetching categories: {str(e)}")
        categories_str = "No categories available."

    # Check for category change
    category_change = False
    new_category = None
    for cat in categories:
        if cat.lower() in message_lower and cat != category:
            category_change = True
            new_category = cat
            break

    # Update recipient context
    recipient_context = extract_recipient_context(message)
    session["recipient_context"] = recipient_context

    # Construct prompt for intent detection
    prompt = (
        f"User message: {message}\n\n"
        f"Session context:\n"
        f"- Product type: {product_type or 'None'}\n"
        f"- Category: {category or 'None'}\n"
        f"- Theme: {theme or 'None'}\n"
        f"- Recipient context: {json.dumps(recipient_context)}\n\n"
        f"Conversation history:\n{conversation_history}\n\n"
        f"Available categories: {categories_str}\n\n"
        f"You are a shopping assistant for Zwende. Determine the intent of the user's message based on the message, session context, recipient context, and conversation history. "
        f"Possible intents include:\n"
        f"- 'greeting': User sends greetings like 'hi', 'hey', 'heyy', 'heyyy', 'hello', or 'hola' at the start of the conversation, including variations with extra characters (e.g., 'heyyy').\n"
        f"- 'acknowledgment': User expresses gratitude or confirmation (e.g., 'thanks', 'okay').\n"
        f"- 'price_concern': User asks about price or affordability (e.g., 'expensive', 'budget').\n"
        f"- 'craftsmanship_explanation': User inquires about materials or quality (e.g., 'handcrafted', 'material').\n"
        f"- 'debug_request': User requests debug information (e.g., 'show all items').\n"
        f"- 'category_change': User specifies a new category different from the current session category.\n"
        f"- 'product_request': User requests a product, specifies a product type, theme, or category, or continues a product-related conversation.\n"
        f"If the message is unclear or does not match any specific intent, default to 'product_request'. "
        f"If the user mentions a category (e.g., 'home decor') different from the current session category ({category}), classify as 'category_change'. "
        f"Consider the recipient context (e.g., gender: {recipient_context.get('gender')}, relation: {recipient_context.get('relation')}) to ensure appropriate intent classification. "
        f"Return a JSON object with a single key 'intent' containing the classified intent as a string."
    )
    print(f"Intent detection prompt: {prompt}")

    payload = {
        "user_id": LYZR_USER_ID,
        "agent_id": LYZR_AGENT_ID,
        "session_id": session.get('session_id', 'unknown'),
        "message": prompt
    }

    agent_response = call_lyzr_api(payload, headers)
    
    if isinstance(agent_response, dict) and 'intent' in agent_response:
        intent = agent_response['intent']
        if intent in ['acknowledgment', 'price_concern', 'craftsmanship_explanation', 'debug_request', 'category_change', 'product_request', 'greeting']:
            # Handle category change
            if intent == 'category_change' and new_category:
                session["category"] = new_category
                session["suggested_categories"] = []
                session["stage"] = "category_detection"  # Reset stage to reprocess with new category
                session["product_type"] = None  # Clear product_type to avoid conflicts
                print(f"Detected category change to: {new_category}, resetting stage and clearing product_type")
            print(f"Detected intent: {intent}")
            end_time = time.time()
            print(f"[{end_time}] detect_intent completed in {end_time - start_time:.3f} seconds")
            return intent
    elif isinstance(agent_response, str):
        try:
            parsed_response = json.loads(agent_response)
            intent = parsed_response.get('intent', 'product_request')
            if intent in ['acknowledgment', 'price_concern', 'craftsmanship_explanation', 'debug_request', 'category_change', 'product_request']:
                if intent == 'category_change' and new_category:
                    session["category"] = new_category
                    session["suggested_categories"] = []
                    session["stage"] = "category_detection"
                    session["product_type"] = None
                    print(f"Parsed category change to: {new_category}, resetting stage and clearing product_type")
                print(f"Parsed intent from string response: {intent}")
                end_time = time.time()
                print(f"[{end_time}] detect_intent completed in {end_time - start_time:.3f} seconds")
                return intent
        except json.JSONDecodeError:
            print(f"Failed to parse agent response as JSON: {agent_response}")

    # Fallback to product_request
    print(f"Invalid or unexpected Lyzr API response: {agent_response}. Defaulting to 'product_request'.")
    end_time = time.time()
    print(f"[{end_time}] detect_intent completed in {end_time - start_time:.3f} seconds")
    return "product_request"

def is_query_vague(message: str, categories: List[str], session: dict, headers: dict) -> Tuple[bool, Optional[str], Optional[str]]:
    print(f"Checking if query is vague: message={message}, categories={categories}, session={session}")
    message_lower = message.lower().strip()
    
    # Define specific product types and themes
    product_types = ["nameplate", "mug", "wallet", "decor", "case", "gift", "toy", "rattle", "soft toy", "stacking toy"]
    theme_keywords = ['ganesha', 'religious', 'floral', 'modern', "traditional", "sensory", "musical"]
    
    # Check for greetings
    greetings = ['hi', 'hey', 'hello']
    if message_lower in greetings:
        print("Detected greeting query")
        return True, None, "greeting"
    
    # Check for specific product types
    for product_type in product_types:
        if product_type in message_lower:
            print(f"Detected specific product type: {product_type}")
            return False, product_type, None
    
    # Check for themes
    for theme in theme_keywords:
        if theme in message_lower:
            print(f"Detected theme: {theme}")
            return False, theme, None
    
    # Fuzzy match against categories
    best_match, score = process.extractOne(message_lower, categories, scorer=fuzz.partial_ratio)
    print(f"Fuzzy match against categories: best_match={best_match}, score={score}")
    if score > 80:
        return False, best_match, None
    
    # Check for specific product titles
    metadata = get_product_metadata(
        category=session.get("category"),
        product_type=session.get("product_type"),
        theme=session.get("theme"),
        max_products=100
    )
    product_titles = []
    if metadata and "No metadata available" not in metadata:
        for line in metadata.split("\n"):
            if line.startswith("  Title:"):
                title = line.replace("  Title:", "").strip()
                product_titles.append(title.lower())
    
    if product_titles:
        best_title_match, title_score = process.extractOne(message_lower, product_titles, scorer=fuzz.partial_ratio)
        print(f"Fuzzy match against product titles: best_title_match={best_title_match}, title_score={title_score}")
        if title_score > 85:
            return False, best_title_match, None
    
    # Detect child-related queries and age
    child_keywords = ['boy', 'girl', 'son', 'daughter', 'child', 'kid', 'baby']
    age_pattern = r'\b(\d+)\s*(?:year|yr|years|old)\b'
    is_child_query = any(keyword in message_lower for keyword in child_keywords)
    age_match = re.search(age_pattern, message_lower)
    
    if is_child_query:
        age = age_match.group(1) if age_match else None
        context = f"child_gift_age_{age}" if age else "child_gift"
        print(f"Detected child-related query: context={context}")
        return True, None, context
    
    # Treat any other vague or occasion-based query as vague, passing the message as context
    print(f"Vague query detected, context={message_lower}")
    return True, None, message_lower

def suggest_categories(query: str, context: str, categories: List[str], headers: dict, recipient_context: Dict[str, str] = None) -> List[str]:
    print(f"Suggesting categories: query={query}, context={context}, categories={categories}, recipient_context={recipient_context}")
    
    # Define gender-inappropriate categories
    female_oriented_categories = ['Jewelry Sets', 'Earrings', 'Necklaces']  # Categories to avoid for male recipients
    couple_oriented_categories = ['Home Decor', 'Gift Giving', 'Decor', 'Personalized Gifts']  # Suitable for anniversaries

    # Adjust categories based on recipient context
    filtered_categories = categories
    if recipient_context.get('gender') == 'male':
        filtered_categories = [cat for cat in categories if cat not in female_oriented_categories]
    if recipient_context.get('occasion') == 'anniversary':
        filtered_categories = [cat for cat in filtered_categories if cat in couple_oriented_categories or cat in categories]

    prompt = (
        f"User query: {query}\n\n"
        f"Context: {context}\n\n"
        f"Recipient context: {json.dumps(recipient_context)}\n\n"
        f"Available categories: {', '.join(filtered_categories)}\n\n"
        f"You are a shopping assistant for Zwende. Based on the user's query, context, and recipient context (gender: {recipient_context.get('gender')}, "
        f"relation: {recipient_context.get('relation')}, occasion: {recipient_context.get('occasion')}), suggest 5 to 10 relevant categories from the provided list. "
        f"If the context indicates a gift for a young child (e.g., 'child_gift_age_1'), prioritize age-appropriate categories. "
        f"If the recipient is male, avoid categories like {', '.join(female_oriented_categories)} unless explicitly mentioned. "
        f"For marriage anniversaries, prioritize couple-oriented categories like {', '.join(couple_oriented_categories)}. "
        f"Include kids' products only if the query explicitly mentions a child or baby. "
        f"Return a JSON object with a single key 'suggested_categories' containing a list of category names."
    )
    print(f"Suggest categories prompt: {prompt}")

    payload = {
        "user_id": LYZR_USER_ID,
        "agent_id": LYZR_AGENT_ID,
        "session_id": "category_suggestion",
        "message": prompt
    }
    
    try:
        agent_response = call_lyzr_api(payload, headers)
        print(f"Suggest categories agent response: {agent_response}")
        if isinstance(agent_response, dict) and 'suggested_categories' in agent_response:
            suggested = agent_response['suggested_categories']
            valid_categories = [cat for cat in suggested if cat in filtered_categories][:10]
            print(f"Valid suggested categories: {valid_categories}")
            return valid_categories
        else:
            print(f"Invalid LLM response: {agent_response}. Using fallback categories.")
            fallback_categories = couple_oriented_categories if recipient_context.get('occasion') == 'anniversary' else ['Gift Giving', 'Decor', 'Mugs']
            return [cat for cat in fallback_categories if cat in filtered_categories][:5]
    except Exception as e:
        print(f"Error suggesting categories: {str(e)}. Using fallback categories.")
        fallback_categories = couple_oriented_categories if recipient_context.get('occasion') == 'anniversary' else ['Gift Giving', 'Decor', 'Mugs']
        return [cat for cat in fallback_categories if cat in filtered_categories][:5]

def fetch_specific_product(product_title: str, product_type: Optional[str] = None, theme: Optional[str] = None) -> Optional[dict]:
    print(f"Fetching specific product: product_title={product_title}, product_type={product_type}, theme={theme}")
    metadata = get_product_metadata(product_type=product_type, theme=theme, max_products=100)
    if not metadata or "No metadata available" in metadata:
        print("No metadata available for specific product")
        return None
    
    current_product = None
    product_title_lower = product_title.lower().strip()
    
    for line in metadata.split("\n"):
        if line.startswith("Product "):
            if current_product is not None:
                # Check if the previous product matches
                if fuzz.partial_ratio(product_title_lower, current_product.get("Title", "").lower()) > 85:
                    print(f"Found matching product: {current_product}")
                    return current_product
            current_product = {}
        elif line.startswith("  "):
            key, value = line[2:].split(": ", 1) if ": " in line else (line[2:], "")
            if key and value:
                current_product[key] = value
    
    # Check the last product
    if current_product and fuzz.partial_ratio(product_title_lower, current_product.get("Title", "").lower()) > 85:
        print(f"Found matching product (last): {current_product}")
        return current_product
    
    print("No matching product found")
    return None

def fetch_product_data(category=None, suggested_categories=None, product_type=None, theme=None, price_sensitive=False, recipient_context: Dict[str, str] = None):
    print(f"Fetching product data: category={category}, suggested_categories={suggested_categories}, product_type={product_type}, theme={theme}, price_sensitive={price_sensitive}, recipient_context={recipient_context}")
    
    try:
        if DATA_SOURCE == 'database':
            query = """
                SELECT 
                    p."Handle", 
                    p."Variant ID" AS "Variant_ID", 
                    p."Title", 
                    p."Category: Name" AS Category_Name, 
                    p."Variant Price" AS Variant_Price, 
                    p."URL", 
                    p."Product Description", 
                    p."Vendor", 
                    p."Tags", 
                    p."Image Src", 
                    COALESCE(SUM(o."Price: Total Line Items")::float, 0) AS total_sold
                FROM products p
                LEFT JOIN orders o ON p."Handle" = o."Line: Product Handle" AND p."Variant ID" = o."Line: Variant ID"
                WHERE p."Variant ID" IS NOT NULL AND p."Variant Price" IS NOT NULL
            """
            params = {}
            if category:
                query += ' AND LOWER(p."Category: Name") = LOWER(%(category)s)'
                params['category'] = category
            elif suggested_categories:
                query += ' AND LOWER(p."Category: Name") = ANY(%(categories)s)'
                params['categories'] = [c.lower() for c in suggested_categories]
            if product_type:
                query += ' AND (LOWER(p."Title") LIKE %(product_type)s OR LOWER(p."Tags") LIKE %(product_type)s OR LOWER(p."Product Description") LIKE %(product_type)s)'
                params['product_type'] = f'%{product_type.lower()}%'
            if theme:
                query += ' AND (LOWER(p."Tags") LIKE %(theme)s OR LOWER(p."Title") LIKE %(theme)s OR LOWER(p."Product Description") LIKE %(theme)s)'
                params['theme'] = f'%{theme.lower()}%'
            
            # Gender-based filtering
            if recipient_context and recipient_context.get('gender') == 'male':
                query += ' AND NOT (LOWER(p."Tags") LIKE \'%earrings%\' OR LOWER(p."Tags") LIKE \'%necklace%\' OR LOWER(p."Tags") LIKE \'%jewelry%\' OR LOWER(p."Category: Name") IN (\'jewelry sets\', \'earrings\'))'
            if recipient_context and recipient_context.get('occasion') == 'anniversary':
                query += ' AND (LOWER(p."Tags") LIKE \'%personalized%\' OR LOWER(p."Tags") LIKE \'%wedding%\' OR LOWER(p."Tags") LIKE \'%love%\' OR LOWER(p."Category: Name") IN (\'home decor\', \'gift giving\', \'decor\'))'

            query += ' GROUP BY p."Handle", p."Variant ID", p."Title", p."Category: Name", p."Variant Price", p."URL", p."Product Description", p."Vendor", p."Tags", p."Image Src"'
            query += ' ORDER BY ' + ('p."Variant Price" ASC' if price_sensitive else 'total_sold DESC')
            query += ' LIMIT 50'
            print(f"Executing product data query: {query} with params: {params}")
            subset = pd.read_sql_query(query, DB_ENGINE, params=params)
            print(f"Product data query result: {len(subset)} rows")

            if subset.empty:
                print("No products found for the specified criteria.")
                return f"No products found for the specified criteria.", []

            unique_subset = []
            seen_titles = set()
            for _, row in subset.iterrows():
                title = row['Title'].lower()
                if not any(title.startswith(seen) or seen.startswith(title) for seen in seen_titles):
                    unique_subset.append(row)
                    seen_titles.add(title)
            subset = pd.DataFrame(unique_subset)
            print(f"Filtered unique products: {len(subset)} rows")

            records = subset.to_dict(orient='records')
            product_info = f"Available products ({'sorted by price' if price_sensitive else 'top-selling'}, up to 50):\n"
            for i, record in enumerate(records[:50]):
                title = record.get('Title', 'Unknown Product')
                price = record.get('Variant_Price', 0.0)
                url = record.get('URL', 'https://example.com')
                variant_id = record.get('Variant_ID', 'unknown-variant')
                description = record.get('Product Description', '')[:200]
                image_src = record.get('Image Src', 'https://example.com/image.jpg')
                product_info += f"{i+1}. {title} (Variant ID: {variant_id})\n   Price: ${price:.2f}\n   URL: {url}\n   Description: {description}\n   Image: {image_src}\n"

            recommended_products = [
                {
                    "Title": record.get('Title', 'Unknown Product'),
                    "Variant ID": record.get('Variant_ID', 'unknown-variant'),
                    "Price": float(record.get('Variant_Price', 0.0)),
                    "URL": record.get('URL', 'https://example.com'),
                    "Description": record.get('Product Description', ''),
                    "Vendor": record.get('Vendor', ''),
                    "Tags": record.get('Tags', ''),
                    "Image Src": record.get('Image Src', 'https://example.com/image.jpg')
                } for record in records[:TOP_K]
            ]
            print(f"Generated product info: {product_info[:500]}... (truncated)")
            print(f"Recommended products: {recommended_products}")
            return product_info, recommended_products
    except Exception as e:
        print(f"Error fetching product data: {str(e)}")
        return f"Error fetching products: {str(e)}", []

def extract_recipient_context(message: str) -> Dict[str, str]:
    """Extract recipient context (gender, relation, occasion) from the user's message."""
    message_lower = message.lower().strip()
    context = {"gender": None, "relation": None, "occasion": None}

    # Gender and relation keywords
    male_keywords = ['brother', 'father', 'husband', 'son', 'boy', 'man']
    female_keywords = ['sister', 'mother', 'wife', 'daughter', 'girl', 'woman']
    neutral_keywords = ['friend', 'colleague', 'partner']
    
    for keyword in male_keywords:
        if keyword in message_lower:
            context["gender"] = "male"
            context["relation"] = keyword
            break
    for keyword in female_keywords:
        if keyword in message_lower:
            context["gender"] = "female"
            context["relation"] = keyword
            break
    for keyword in neutral_keywords:
        if keyword in message_lower:
            context["relation"] = keyword
            break

    # Occasion keywords
    occasion_keywords = {
        'marriage anniversary': 'anniversary',
        'wedding': 'wedding',
        'birthday': 'birthday',
        'christmas': 'christmas'
    }
    for phrase, occasion in occasion_keywords.items():
        if phrase in message_lower:
            context["occasion"] = occasion
            break

    print(f"Extracted recipient context: {context}")
    return context

@app.post("/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    print(f"[{start_time}] Chat request received: session_id={request.session_id}, message={request.message}")
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": LYZR_API_KEY
    }

    if request.session_id not in SESSION_STATE:
        SESSION_STATE[request.session_id] = {
            "category": None,
            "suggested_categories": [],
            "questions_asked": [],
            "user_responses": [],
            "stage": "category_detection",
            "product_metadata": [],
            "price_sensitive": False,
            "intent": None,
            "recommended_products": [],
            "last_product_info": None,
            "product_type": None,
            "theme": None,
            "session_id": request.session_id,
            "context": None,
            "recipient_context": {}
        }
        print(f"Initialized new session: {SESSION_STATE[request.session_id]}")

    session = SESSION_STATE[request.session_id]
    session["intent"] = detect_intent(request.message, session, headers)
    print(f"Updated session with intent: {session['intent']}")

    categories = json.loads(get_categories().body.decode('utf-8'))
    print(f"Categories for chat: {categories}")

    session["user_responses"].append(request.message)
    print(f"Appended user response: {session['user_responses']}")

    # Update recipient context
    if not session["recipient_context"]:
        session["recipient_context"] = extract_recipient_context(request.message)
        print(f"Updated recipient context: {session['recipient_context']}")

    # Temporary fallback for theme detection
    if not session.get("theme"):
        theme_keywords = ['ganesha', 'religious', 'floral', 'modern', 'traditional']
        for keyword in theme_keywords:
            if keyword in request.message.lower():
                session["theme"] = keyword
                print(f"Detected theme via fallback: {session['theme']}")
                break

    if session["intent"] == "price_concern":
        session["price_sensitive"] = True
        print(f"Set price_sensitive to True due to price_concern intent")

    welcome_message = (
        "Hey! Welcome to Zwende! ❤️\n\n"
        "We have one-of-a-kind handcrafted products and DIY experiences from independent artists and artisans in India.✨\n\n"
        "Please select an option below to proceed.\n\n"
        "1. Nameplates for home\n"
        "2. Name Hangings for Kids\n"
        "3. Terracotta Mugs & Cups\n"
        "4. Order Tracking\n"
        "5. Others"
    )

    if session["intent"] == "greeting":
        session["category"] = None
        session["context"] = "greeting"
        session["suggested_categories"] = ["Name Plates", "Dolls, Playsets & Toy Figures", "Mugs", "Gift Giving"]
        session["stage"] = "follow_up"
        session["questions_asked"].append(welcome_message)
        print(f"Greeting detected, returning welcome message")

        end_time = time.time()
        print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
        return {"response": welcome_message}
    
    if session["stage"] == "category_detection" or session["intent"] == "category_change":
        print(f"Processing category_detection stage (intent: {session['intent']})")
        try:
            is_vague, matched_item, context = is_query_vague(request.message, categories, session, headers)
            print(f"is_query_vague result: is_vague={is_vague}, matched_item={matched_item}, context={context}")
        except Exception as e:
            print(f"Error in is_query_vague: {str(e)}")
            is_vague, matched_item, context = True, None, "greeting"

        if is_vague:
            session["category"] = None
            session["context"] = context
            print(f"Query is vague, setting category=None, context={context}")
            suggested_categories = suggest_categories(
                request.message, context, categories, headers, session["recipient_context"]
            )
            session["suggested_categories"] = suggested_categories
            session["stage"] = "follow_up"
            print(f"Suggested categories: {suggested_categories}, stage updated to follow_up")

            metadata = get_product_metadata(
                limit_categories=suggested_categories,
                product_type=session.get("product_type"),
                theme=session.get("theme"),
                max_products=50
            )
            print(f"Metadata for vague query: {metadata[:500]}... (truncated)")

            prompt = (
                f"User query: {request.message}\n\n"
                f"Context: {context}\n\n"
                f"Recipient context: {json.dumps(session['recipient_context'])}\n\n"
                f"Available categories: {', '.join(categories)}\n\n"
                f"Suggested categories: {', '.join(suggested_categories)}\n\n"
                f"Conversation history: None\n\n"
                f"Product metadata:\n{metadata}\n\n"
                f"You are a shopping assistant for Zwende. The user's query is vague or indicates a gift without specific details. "
                f"Generate a follow-up question to clarify their preferences, such as product type, theme, material, or occasion, "
                f"considering the context ({context}) and recipient context (gender: {session['recipient_context'].get('gender')}, "
                f"relation: {session['recipient_context'].get('relation')}, occasion: {session['recipient_context'].get('occasion')}). "
                f"Exclude kids' products unless explicitly mentioned. "
                f"For male recipients, avoid suggesting jewelry or female-oriented products. "
                f"For anniversaries, suggest couple-oriented items like home decor. "
                f"Return a JSON object with 'category' (set to 'None'), 'suggested_categories', 'question', 'attributes' (list), and 'theme' (if detected)."
            )
            print(f"Prompt for vague query: {prompt}")

            payload = {
                "user_id": LYZR_USER_ID,
                "agent_id": LYZR_AGENT_ID,
                "session_id": request.session_id,
                "message": prompt
            }
            agent_response = call_lyzr_api(payload, headers) or {
                "category": "None",
                "suggested_categories": suggested_categories,
                "question": f"Can you tell me more about what you're looking for, like the type of product or theme? Suggested categories: {', '.join(suggested_categories)}",
                "attributes": [],
                "theme": session.get("theme")
            }
            print(f"Agent response for vague query: {agent_response}")

            detected_category = agent_response.get("category", "None").strip()
            suggested_categories = agent_response.get("suggested_categories", suggested_categories)
            follow_up_question = agent_response.get("question", "Can you tell me more about what you're looking for?")
            attributes = agent_response.get("attributes", [])
            theme = agent_response.get("theme", session.get("theme"))

            session["category"] = detected_category if detected_category != "None" and detected_category in categories else None
            session["suggested_categories"] = suggested_categories
            session["stage"] = "follow_up"
            session["questions_asked"].append(follow_up_question)
            session["product_metadata"] = attributes if isinstance(attributes, list) else []
            session["theme"] = theme
            print(f"Updated session: {session}")
            
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return {"response": follow_up_question}

        if matched_item:
            if matched_item in categories:
                session["category"] = matched_item
                session["suggested_categories"] = []
                print(f"Matched category: {matched_item}")
            elif matched_item in theme_keywords:
                session["theme"] = matched_item
                print(f"Matched theme: {matched_item}")
            else:
                session["product_type"] = matched_item
                print(f"Matched product_type: {matched_item}")

        metadata = get_product_metadata(
            category=session["category"],
            product_type=session.get("product_type"),
            theme=session.get("theme"),
            max_products=50
        ) if session["category"] else get_product_metadata(
            limit_categories=session["suggested_categories"] or categories[:5],
            product_type=session.get("product_type"),
            theme=session.get("theme"),
            max_products=50
        )
        prompt = (
            f"User query: {request.message}\n\n"
            f"Available categories: {', '.join(categories)}\n\n"
            f"Recipient context: {json.dumps(session['recipient_context'])}\n\n"
            f"You are a shopping assistant for Zwende. Based on the user's query and recipient context (gender: {session['recipient_context'].get('gender')}, "
            f"relation: {session['recipient_context'].get('relation')}, occasion: {session['recipient_context'].get('occasion')}), "
            f"identify the most relevant category or product type. "
            f"If the query specifies a product type (e.g., 'nameplate') or theme (e.g., 'ganesha'), prioritize products matching those in the metadata. "
            f"If the query is vague, suggest 5 to 10 relevant categories based on the product metadata. "
            f"Exclude kids' products unless explicitly mentioned. "
            f"For male recipients, avoid female-oriented products like jewelry unless requested. "
            f"For anniversaries, prioritize couple-oriented items like home decor. "
            f"Return a JSON object with 'category', 'suggested_categories', 'question' (a follow-up question to clarify preferences), 'attributes' (list), and 'theme' (if detected)."
        )
        print(f"Prompt for specific query: {prompt}")

        payload = {
            "user_id": LYZR_USER_ID,
            "agent_id": LYZR_AGENT_ID,
            "session_id": request.session_id,
            "message": prompt
        }
        agent_response = call_lyzr_api(payload, headers) or {
            "category": None,
            "suggested_categories": categories[:8],
            "question": "Can you share more details about the type of product, theme, or occasion you're interested in?",
            "attributes": [],
            "theme": session.get("theme")
        }
        print(f"Agent response for specific query: {agent_response}")

        # Handle agent_response to avoid NoneType errors
        detected_category = agent_response.get("category")  # Get category, allow None
        if detected_category == "None" or detected_category is None:
            detected_category = None  # Normalize to None
        else:
            detected_category = str(detected_category).strip()  # Strip only if not None
        suggested_categories = agent_response.get("suggested_categories", [])
        follow_up_question = agent_response.get("question", "Can you share more details about what you're looking for?")
        attributes = agent_response.get("attributes", [])
        theme = agent_response.get("theme", session.get("theme"))

        session["category"] = detected_category if detected_category and detected_category in categories else None
        session["suggested_categories"] = suggested_categories
        session["stage"] = "follow_up"
        session["questions_asked"].append(follow_up_question)
        session["product_metadata"] = attributes if isinstance(attributes, list) else []
        session["theme"] = theme
        print(f"Updated session: {session}")

        end_time = time.time()
        print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
        return {"response": follow_up_question}

    elif session["stage"] == "follow_up":
        print("Processing follow_up stage")
        if len(session["questions_asked"]) < 2:
            metadata = get_product_metadata(
                category=session["category"],
                product_type=session.get("product_type"),
                theme=session.get("theme"),
                max_products=50
            ) if session["category"] else get_product_metadata(
                limit_categories=session["suggested_categories"] or categories[:5],
                product_type=session.get("product_type"),
                theme=session.get("theme"),
                max_products=50
            )
            print(f"Metadata for follow_up: {metadata[:500]}... (truncated)")

            conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(session["questions_asked"], session["user_responses"][1:]))
            attributes_str = ", ".join(str(attr) for attr in session["product_metadata"]) if session["product_metadata"] and isinstance(session["product_metadata"], (list, tuple)) else ""
            print(f"Conversation history: {conversation_history}, attributes: {attributes_str}")

            prompt = (
                f"User query: {session['user_responses'][0]}\n\n"
                f"Detected category: {session['category'] or 'None'}\n\n"
                f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
                f"Detected product type: {session.get('product_type', 'None')}\n\n"
                f"Detected theme: {session.get('theme', 'None')}\n\n"
                f"Recipient context: {json.dumps(session['recipient_context'])}\n\n"
                f"Product metadata:\n{metadata}\n\n"
                f"Conversation history:\n{conversation_history}\n\n"
                f"Current user response: {request.message}\n\n"
                f"Previous attributes: {attributes_str}\n\n"
                f"You are a shopping assistant for Zwende. Generate a follow-up question to narrow down the user's preferences, "
                f"considering their initial query, current response, detected product type, theme, recipient context, and conversation history. "
                f"Ask about specific attributes like material, size, color, relevant to the product type and theme based on the suggested category and product metadata only. "
                f"If no clear category is detected, suggest relevant categories or product types based on the suggested category and product metadata only. "
                f"Exclude kids' products unless explicitly mentioned. "
                f"For male recipients, avoid female-oriented products like jewelry unless requested. "
                f"For anniversaries, prioritize couple-oriented items like home decor. "
                f"Return a JSON object with 'category' (update if clearer), 'question' (context-aware follow-up), 'attributes' (list of product attributes), and 'theme' (if detected)."
            )
            print(f"Prompt for follow_up: {prompt}")

            payload = {
                "user_id": LYZR_USER_ID,
                "agent_id": LYZR_AGENT_ID,
                "session_id": request.session_id,
                "message": prompt
            }
            agent_response = call_lyzr_api(payload, headers) or {
                "category": session["category"] or "None",
                "suggested_categories": session["suggested_categories"] or categories[:8],
                "question": f"Can you provide more details about the {session.get('product_type', 'product')} you're looking for, such as size, color, or budget?",
                "attributes": [],
                "theme": session.get("theme")
            }
            print(f"Agent response for follow_up: {agent_response}")

            follow_up_question = agent_response.get("question", f"Can you provide more details about the {session.get('product_type', 'product')} you're looking for?")
            attributes = agent_response.get("attributes", session["product_metadata"])
            updated_category = agent_response.get("category", session["category"] or "None")
            theme = agent_response.get("theme", session.get("theme"))

            if updated_category != "None" and updated_category in categories:
                session["category"] = updated_category
                session["suggested_categories"] = []
            session["questions_asked"].append(follow_up_question)
            session["product_metadata"] = attributes if isinstance(attributes, list) else []
            session["theme"] = theme
            print(f"Updated session: {session}")
            
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return {"response": follow_up_question}

        else:
            print("Moving to recommendation stage")
            session["stage"] = "recommendation"
            product_type = session.get("product_type")
            theme = session.get("theme")
            metadata = get_product_metadata(
                category=session["category"],
                product_type=product_type,
                theme=theme,
                max_products=50
            ) if session["category"] else get_product_metadata(
                limit_categories=session["suggested_categories"] or categories[:5],
                product_type=product_type,
                theme=theme,
                max_products=50
            )
            print(f"Metadata for recommendation: {metadata[:500]}... (truncated)")

            conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(session["questions_asked"], session["user_responses"][1:]))
            print(f"Conversation history for recommendation: {conversation_history}")

            product_info, recommended_products = fetch_product_data(
                category=session["category"],
                suggested_categories=session["suggested_categories"],
                product_type=product_type,
                theme=theme,
                price_sensitive=session["price_sensitive"],
                recipient_context=session["recipient_context"]
            )
            print(f"Product info for recommendation: {product_info[:500]}... (truncated)")
            print(f"Recommended products: {recommended_products}")

            session["last_product_info"] = product_info
            session["recommended_products"] = recommended_products

            prompt = (
                f"User query: {session['user_responses'][0]}\n\n"
                f"Detected category: {session['category'] or 'None'}\n\n"
                f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
                f"Detected product type: {session.get('product_type', 'None')}\n\n"
                f"Detected theme: {session.get('theme', 'None')}\n\n"
                f"Recipient context: {json.dumps(session['recipient_context'])}\n\n"
                f"Conversation history:\n{conversation_history}\n\n"
                f"Current user response: {request.message}\n\n"
                f"Product metadata:\n{metadata}\n\n"
                f"Product data:\n{product_info}\n\n"
                f"You are a shopping assistant for Zwende. Recommend up to {TOP_K} unique products matching the user's preferences, and "
                f"including the specified product type ({product_type or 'any'}) and theme ({theme or 'any'}). only suggest from the available products and category"
                f"{'Prioritize affordable options.' if session['price_sensitive'] else 'Select top-selling products.'} "
                f"Consider the recipient context (gender: {session['recipient_context'].get('gender')}, "
                f"relation: {session['recipient_context'].get('relation')}, occasion: {session['recipient_context'].get('occasion')}). "
                f"For male recipients, avoid female-oriented products like jewelry unless requested. "
                f"For anniversaries, prioritize couple-oriented items like home decor or personalized gifts. "
                f"List each product with title, price, URL, description, vendor, tags, and image URL. "
                f"If insufficient details are provided, generate a follow-up question about specific attributes (e.g., size, color) instead of recommendations."
            )
            print(f"Prompt for recommendation: {prompt}")

            payload = {
                "user_id": LYZR_USER_ID,
                "agent_id": LYZR_AGENT_ID,
                "session_id": request.session_id,
                "message": prompt
            }
            agent_response = call_lyzr_api(payload, headers) or f"Can you provide more details about the {product_type or 'product'} you're looking for, such as size, color?"
            print(f"Agent response for recommendation: {agent_response}")

            session["stage"] = "post_recommendation"
            session["questions_asked"].append("Recommendation provided")
            print(f"Updated session: {session}")
            
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return {"response": agent_response}

    elif session["stage"] == "post_recommendation":
        print("Processing post_recommendation stage")
        product_type = session.get("product_type")
        theme = session.get("theme")
        metadata = get_product_metadata(
            category=session["category"],
            product_type=product_type,
            theme=theme,
            max_products=50
        ) if session["category"] else get_product_metadata(
            limit_categories=session["suggested_categories"] or categories[:5],
            product_type=product_type,
            theme=theme,
            max_products=50
        )
        print(f"Metadata for post_recommendation: {metadata[:500]}... (truncated)")

        conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(session["questions_asked"], session["user_responses"][1:]))
        product_info = session.get("last_product_info", "No product information available.")
        print(f"Conversation history: {conversation_history}, product_info: {product_info[:500]}... (truncated)")

        if product_type and theme and len(session["user_responses"]) >= 2:
            product_info, recommended_products = fetch_product_data(
                category=session["category"],
                suggested_categories=session["suggested_categories"],
                product_type=product_type,
                theme=theme,
                price_sensitive=session["price_sensitive"],
                recipient_context=session["recipient_context"]
            )
            print(f"Product info for post_recommendation: {product_info[:500]}... (truncated)")
            print(f"Recommended products: {recommended_products}")

            session["last_product_info"] = product_info
            session["recommended_products"] = recommended_products

            prompt = (
                f"User query: {session['user_responses'][0]}\n\n"
                f"Detected category: {session['category'] or 'None'}\n\n"
                f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
                f"Detected product type: {product_type or 'None'}\n\n"
                f"Detected theme: {theme or 'None'}\n\n"
                f"Recipient context: {json.dumps(session['recipient_context'])}\n\n"
                f"Conversation history:\n{conversation_history}\n\n"
                f"Current user response: {request.message}\n\n"
                f"Product metadata:\n{metadata}\n\n"
                f"Product data:\n{product_info}\n\n"
                f"You are a shopping assistant for Zwende. The user has specified a {product_type} with a {theme} theme. "
                f"Recommend up to {TOP_K} unique products matching these preferences. "
                f"{'Prioritize affordable options.' if session['price_sensitive'] else 'Select top-selling products.'} "
                f"Consider the recipient context (gender: {session['recipient_context'].get('gender')}, "
                f"relation: {session['recipient_context'].get('relation')}, occasion: {session['recipient_context'].get('occasion')}). "
                f"For male recipients, avoid female-oriented products like jewelry unless requested. "
                f"For anniversaries, prioritize couple-oriented items like home decor or personalized gifts. "
                f"List each product with title, price, URL, description, vendor, tags, and image URL."
                f"Ensure both the product URL and image URL are included for each product. "
            )
            print(f"Prompt for post_recommendation (with product_type and theme): {prompt}")
        else:
            prompt = (
                f"User query: {session['user_responses'][0]}\n\n"
                f"Detected category: {session['category'] or 'None'}\n\n"
                f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
                f"Detected product type: {product_type or 'None'}\n\n"
                f"Detected theme: {theme or 'None'}\n\n"
                f"Recipient context: {json.dumps(session['recipient_context'])}\n\n"
                f"Conversation history:\n{conversation_history}\n\n"
                f"Current user response: {request.message}\n\n"
                f"Product metadata:\n{metadata}\n\n"
                f"Product data:\n{product_info}\n\n"
                f"You are a shopping assistant for Zwende. Generate a follow-up question to clarify the user's preferences for a {product_type or 'product'}, "
                f"such as size, color, material based on the conversation history, current response, and recipient context. "
                f"For male recipients, avoid female-oriented products like jewelry unless requested. "
                f"For anniversaries, prioritize couple-oriented items like home decor. "
                f"Return a JSON object with 'category', 'question', 'attributes' (list of product attributes), and 'theme' (if detected)."
            )
            print(f"Prompt for post_recommendation (follow-up question): {prompt}")

        payload = {
            "user_id": LYZR_USER_ID,
            "agent_id": LYZR_AGENT_ID,
            "session_id": request.session_id,
            "message": prompt
        }
        agent_response = call_lyzr_api(payload, headers) or f"Can you provide more details about the {product_type or 'product'} you're looking for, such as size, color?"
        print(f"Agent response for post_recommendation: {agent_response}")
        if isinstance(agent_response, dict):
            follow_up_question = agent_response.get("question", f"Can you provide more details about the {product_type or 'product'} you're looking for?")
            attributes = agent_response.get("attributes", session["product_metadata"])
            updated_category = agent_response.get("category", session["category"] or "None")
            theme = agent_response.get("theme", session.get("theme"))

            if updated_category != "None" and updated_category in categories:
                session["category"] = updated_category
                session["suggested_categories"] = []
            session["questions_asked"].append(follow_up_question)
            session["product_metadata"] = attributes if isinstance(attributes, list) else []
            session["theme"] = theme
            print(f"Updated session: {session}")
            
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return {"response": follow_up_question}

        end_time = time.time()
        print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
        return {"response": agent_response}
    
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}