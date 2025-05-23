import os
import random
import pandas as pd
import numpy as np
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
from bs4 import BeautifulSoup
from sqlalchemy.exc import DatabaseError

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
    print("Lifespan handler defined and called.")
    global DB_ENGINE
    print(f"DATA_SOURCE is set to: {DATA_SOURCE}")
    try:
        db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        print(f"Initializing RDS database connection with URL: {db_url}")
        DB_ENGINE = create_engine(db_url, poolclass=QueuePool, pool_size=5, max_overflow=10)
        print("RDS database connection pool initialized successfully.", DB_ENGINE)
    except Exception as e:
        print(f"Error initializing RDS database connection: {str(e)}")
        raise
    yield
    print("Lifespan handler shutting down.")
    if DB_ENGINE:
        print("Closing database connection pool.")
        DB_ENGINE.dispose()
        print("Database connection pool closed.")

print("Lifespan function defined.")

app = FastAPI(title="Zwende Search Agent", version="1.0.0", lifespan=lifespan)
print("FastAPI app initialized.")

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

# Hardcoded responses
WELCOME_MESSAGE = {
    "response": (
        "Hey! Welcome to Zwende! ❤️\n\n"
        "We have one-of-a-kind handcrafted products and DIY experiences from independent artists and artisans in India.✨\n\n"
        "Please select an option below to proceed:"
    ),
    "type": "interactive",
    "buttons": [
        "Nameplates for Home",
        "Name Hangings for Kids",
        "Terracotta Mugs & Cups",
        "Order Tracking",
        "Others"
    ],
}

NAMEPLATES_MESSAGE = {
    "response": (
        "🏡 Personalized Nameplates for Home\n"
        "1600+ Unique Designs!\n\n"
        "✨ Customize with names, fonts & colors\n"
        "🎨 Handcrafted by skilled artists\n"
        "🌿 Minimal, ethnic and photo personalized options\n"
        "🚚 Free Shipping Above ₹999\n\n"
        "Please select the type of nameplate:"
    ),
    "type": "interactive",
    "buttons": [
        "Outdoor Name Plates",
        "Minimal Name Plates",
        "Photo Personalized Nameplate",
        "Ethnic Nameplates",
        "Vernacular Nameplates"
    ]
}

NAMEPLATE_FILTERS = {
    "outdoor name plates": {
        "type": "Nameboards",
        "tags": ["utility:outdoors"],
        "response": "Here are the top 10 best-selling Outdoor Name Plates:"
    },
    "minimal name plates": {
        "type": "Nameboards",
        "tags": ["style:minimalist", "demographic:adults"],
        "response": "Here are the top 10 best-selling Minimal Name Plates:"
    },
    "photo personalized nameplate": {
        "type": "Nameboards",
        "tags": ["personalisation:photo based caricature"],
        "response": "Here are the top 10 best-selling Photo Personalized Nameplates:"
    },
    "ethnic nameplates": {
        "type": "Nameboards",
        "tags": ["style:ethnic"],
        "response": "Here are the top 10 best-selling Ethnic Nameplates:"
    },
    "vernacular nameplates": {
        "type": "Nameboards",
        "tags": ["type of nameboard:vernacular"],
        "response": "Here are the top 10 best-selling Vernacular Nameplates:"
    }
}

NAME_HANGINGS_MESSAGE = {
    "response": (
        "🎁 Name Hangings for Kids – 1000+ Cute Designs!\n\n"
        "✨ Customize with names, themes & colors\n"
        "🎨 Handcrafted in non-toxic, child-safe materials\n"
        "🦄 Cute designs for nurseries, rooms & doors\n"
        "🚚 Free Shipping Above ₹999 | Bulk Orders Accepted\n\n"
        "Please select the type of name hanging:"
    ),
    "type": "interactive",
    "buttons": [
        "Unicorn Theme 🦄",
        "Superhero Theme 🦹‍♂️",
        "Swing Designs 🪆",
        "Moon Designs 🌝",
        "Rainbow Designs 🌈",
        "Jungle Theme 🐒",
        "Themes for Boys ⚽",
        "Themes for Girls 👸",
        "Space Theme 🚀"
    ]
}

MUGS_MESSAGE = {
    "response": (
        "🌿 Terracotta Mugs & Cups - 200+ Designs\n\n"
        "✨ Personalize with photos & names\n"
        "🎨 Hand-painted terracotta designs\n"
        "🌿 Perfect for gifting & daily use\n"
        "💸 Discount in Bulk Orders\n"
        "🚚 Free Shipping Above ₹999 | Bulk Orders Accepted\n\n"
        "Please select the type of mug:"
    ),
    "type": "interactive",
    "buttons": [
        "Mugs for Father's Day",
        "Mugs for Kids",
        "Mugs for Wedding",
        "Mugs for Couple",
        "Mugs Showing Hobbies"
    ]
}

NAME_HANGINGS_FILTERS = {
    "unicorn theme": {
        "type": ["Nameboards", "Bunting"],
        "tags": ["themes:unicorn", "demographic:kids"],
        "response": "Here are the top 10 best-selling Unicorn Theme Name Hangings:"
    },
    "superhero theme": {
        "type": ["Nameboards", "Bunting"],
        "tags": ["themes:superhero", "demographic:kids"],
        "response": "Here are the top 10 best-selling Superhero Theme Name Hangings:"
    },
    "swing designs": {
        "type": ["Nameboards", "Bunting"],
        "tags": ["themes:with swing", "demographic:kids"],
        "response": "Here are the top 10 best-selling Swing Designs Name Hangings:"
    },
    "moon designs": {
        "type": ["Nameboards", "Bunting"],
        "tags": ["shape:moon", "demographic:kids"],
        "response": "Here are the top 10 best-selling Moon Designs Name Hangings:"
    },
    "rainbow designs": {
        "type": ["Nameboards", "Bunting"],
        "tags": ["artform:macrame", "demographic:kids"],
        "response": "Here are the top 10 best-selling Rainbow Designs Name Hangings:"
    },
    "jungle theme": {
        "type": ["Nameboards", "Bunting"],
        "tags": ["themes:jungle", "demographic:kids"],
        "response": "Here are the top 10 best-selling Jungle Theme Name Hangings:"
    },
    "themes for boys": {
        "type": ["Nameboards"],
        "tags": ["demographic:kids", "category:name boards", "demographic:boy"],
        "response": "Here are the top 10 best-selling Themes for Boys Name Hangings:"
    },
    "themes for girls": {
        "type": ["Nameboards"],
        "tags": ["demographic:kids", "category:name boards", "demographic:girl"],
        "response": "Here are the top 10 best-selling Themes for Girls Name Hangings:"
    },
    "space theme": {
        "type": ["Nameboards", "Bunting"],
        "tags": ["themes:space", "demographic:kids"],
        "response": "Here are the top 10 best-selling Space Theme Name Hangings:"
    }
}

MUGS_FILTERS = {
    "mugs for father's day": {
        "type": ["Mugs"],
        "vendor": ["Happy Earth Studio"],
        "tags": ["festival:father's day"],
        "response": "Here are the top 10 best-selling Mugs for Father's Day:"
    },
    "mugs for kids": {
        "type": ["Mugs"],
        "vendor": ["Happy Earth Studio"],
        "tags": ["demographic:kids"],
        "title_not_contains": ["rakhi", "christmas", "planter"],
        "response": "Here are the top 10 best-selling Mugs for Kids:"
    },
    "mugs for wedding": {
        "type": ["Mugs"],
        "vendor": ["Happy Earth Studio"],
        "tags": ["occasion:wedding"],
        "response": "Here are the top 10 best-selling Mugs for Wedding:"
    },
    "mugs for couple": {
        "type": ["Mugs"],
        "vendor": ["Happy Earth Studio"],
        "tags": ["demographic:parents"],
        "response": "Here are the top 10 best-selling Mugs for Couples:"
    },
    "mugs showing hobbies": {
        "type": ["Mugs"],
        "vendor": ["Happy Earth Studio"],
        "title_contains": ["Hobby"],
        "response": "Here are the top 10 best-selling Mugs Showing Hobbies:"
    }
}

HARDCODED_CATEGORIES = ["Name Plates", "Dolls, Playsets & Toy Figures", "Mugs"]

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

def get_distinct_tags(category: str) -> List[str]:
    """Fetch distinct tags for a given category from the products table."""
    start_time = time.time()
    print(f"[{start_time}] Fetching distinct tags for category: {category}")
    
    if DB_ENGINE is None:
        print("Database connection not initialized.")
        return []
    
    try:
        query = """
            SELECT DISTINCT TRIM(LOWER(tag)) AS tag
            FROM products p, UNNEST(STRING_TO_ARRAY(p."Tags", ',')) AS tag
            WHERE LOWER(p."Category: Name") = LOWER(%(category)s)
            AND p."Tags" IS NOT NULL
            AND p."Tags" != ''
            AND TRIM(LOWER(tag)) != ''
            ORDER BY tag
        """
        params = {'category': category}
        print(f"Executing tags query: {query} with params: {params}")
        tags_df = pd.read_sql_query(query, DB_ENGINE, params=params)
        tags = tags_df['tag'].str.strip().tolist()
        # Filter out empty or irrelevant tags (e.g., too short or duplicates)
        tags = list(dict.fromkeys([tag for tag in tags if tag and len(tag) > 2]))
        print(f"Retrieved {len(tags)} distinct tags: {tags}")
        end_time = time.time()
        print(f"[{end_time}] get_distinct_tags completed in {end_time - start_time:.3f} seconds")
        return tags
    except Exception as e:
        print(f"Error fetching distinct tags: {str(e)}")
        end_time = time.time()
        print(f"[{end_time}] get_distinct_tags completed in {end_time - start_time:.3f} seconds")
        return []

@app.get("/categories", response_model=List[str])
def get_categories():
    print("Fetching categories")
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
                "URL": str(record.get('URL')),
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

    try:
        categories = json.loads(get_categories().body.decode('utf-8'))
        categories_str = ", ".join(categories) if categories else "No categories available."
        print(f"Available categories: {categories_str}")
    except Exception as e:
        print(f"Error fetching categories: {str(e)}")
        categories_str = "No categories available."

    # Check for category match first, before greeting detection
    category_change = False
    new_category = None
    for cat in categories:
        if cat.lower() in message_lower and cat != category:
            category_change = True
            new_category = cat
            break
    if not new_category:  # Fuzzy match for close category names
        best_match, score = process.extractOne(message_lower, categories, scorer=fuzz.partial_ratio)
        if score > 80:
            category_change = True
            new_category = best_match

    recipient_context = extract_recipient_context(message)
    session["recipient_context"] = recipient_context

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
        f"- 'greeting': User sends greetings like 'hi', 'hey', 'hello' with no specific category or product request.\n"
        f"- 'acknowledgment': User expresses gratitude or confirmation (e.g., 'thanks', 'okay').\n"
        f"- 'price_concern': User asks about price or affordability (e.g., 'expensive', 'budget').\n"
        f"- 'craftsmanship_explanation': User inquires about materials or quality (e.g., 'handcrafted', 'material').\n"
        f"- 'debug_request': User requests debug information (e.g., 'show all items').\n"
        f"- 'category_change': User specifies a new category different from the current session category.\n"
        f"- 'product_request': User requests a product, specifies a product type, theme, category, or subcategory, or continues a product-related conversation.\n"
        f"If the message contains a clear category reference (e.g., 'art and crafts', 'mugs'), classify as 'category_change' even if it starts with a greeting (e.g., 'heyy'). "
        f"If the message is unclear or does not match any specific intent, default to 'product_request'. "
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
            if intent == 'category_change' and new_category:
                session["category"] = new_category
                session["suggested_categories"] = []
                session["stage"] = "category_detection"
                session["product_type"] = None
                session["subcategory_tags"] = []
                print(f"Detected category change to: {new_category}, resetting stage and clearing product_type and subcategory_tags")
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
                    session["subcategory_tags"] = []
                    print(f"Parsed category change to: {new_category}, resetting stage and clearing product_type and subcategory_tags")
                print(f"Parsed intent from string response: {intent}")
                end_time = time.time()
                print(f"[{end_time}] detect_intent completed in {end_time - start_time:.3f} seconds")
                return intent
        except json.JSONDecodeError:
            print(f"Failed to parse agent response as JSON: {agent_response}")

    print(f"Invalid or unexpected Lyzr API response: {agent_response}. Defaulting to 'product_request'.")
    end_time = time.time()
    print(f"[{end_time}] detect_intent completed in {end_time - start_time:.3f} seconds")
    return "product_request"

def detect_category(message: str, headers: dict) -> Optional[str]:
    start_time = time.time()
    message_lower = message.lower().strip()
    print(f"[{start_time}] Detecting category for message: {message_lower}")

    # Define hardcoded categories
    hardcoded_categories = ["Name Plates", "Dolls, Playsets & Toy Figures", "Mugs"]

    # Step 1: Try quick keyword matching as a first pass (for efficiency)
    category_keywords = {
        "Name Plates": ["name plate", "nameplates", "name-plate", "nameplates for home"],
        "Name Hanging": ["name hanging", "name hangings", "namehanging", "name hangings for kids", "kids namehanging"],
        "Mugs": ["mug", "mugs", "cup", "cups", "terracotta mugs & cups"]
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            print(f"[{time.time()}] Keyword match found for category: {category}")
            end_time = time.time()
            print(f"[{end_time}] detect_category completed in {end_time - start_time:.3f} seconds")
            return category

    # Step 2: If no keyword match, use LLM for fuzzy matching
    prompt = (
        f"User message: {message}\n\n"
        f"Available categories: {', '.join(hardcoded_categories)}\n\n"
        f"You are a shopping assistant for Zwende. Based on the user's message, determine if it refers to one of the following categories: {', '.join(hardcoded_categories)}. "
        f"Consider possible typos, synonyms, or related terms. For example:\n"
        f"- 'nameplte' or 'home sign' should map to 'Name Plates'.\n"
        f"- ''namehanging' should map to 'Dolls, Playsets & Toy Figures'.\n"
        f"- 'coffee cup' or 'muggs' should map to 'Mugs'.\n"
        f"If the message clearly refers to one of these categories, return the category name. "
        f"If the message is vague or refers to something else (e.g., 'wallet', 'greeting'), return null. "
        f"Return a JSON object with a single key 'category' containing the category name or null."
    )
    print(f"[{time.time()}] LLM prompt for category detection: {prompt}")

    payload = {
        "user_id": LYZR_USER_ID,
        "agent_id": LYZR_AGENT_ID,
        "session_id": "category_detection",
        "message": prompt
    }

    try:
        agent_response = call_lyzr_api(payload, headers)
        print(f"[{time.time()}] LLM response: {agent_response}")

        if isinstance(agent_response, dict) and 'category' in agent_response:
            category = agent_response['category']
            if category in hardcoded_categories:
                print(f"[{time.time()}] LLM detected category: {category}")
                end_time = time.time()
                print(f"[{end_time}] detect_category completed in {end_time - start_time:.3f} seconds")
                return category
            elif category is None:
                print(f"[{time.time()}] LLM returned null, no category matched")
                end_time = time.time()
                print(f"[{end_time}] detect_category completed in {end_time - start_time:.3f} seconds")
                return None
            else:
                print(f"[{time.time()}] LLM returned invalid category: {category}, falling back to None")
                end_time = time.time()
                print(f"[{end_time}] detect_category completed in {end_time - start_time:.3f} seconds")
                return None
        else:
            print(f"[{time.time()}] Invalid LLM response format: {agent_response}, falling back to None")
            end_time = time.time()
            print(f"[{end_time}] detect_category completed in {end_time - start_time:.3f} seconds")
            return None
    except Exception as e:
        print(f"[{time.time()}] Error in LLM call: {str(e)}, falling back to None")
        end_time = time.time()
        print(f"[{end_time}] detect_category completed in {end_time - start_time:.3f} seconds")
        return None # No hardcoded category matched

def is_query_vague(message: str, categories: List[str], session: dict, headers: dict) -> Tuple[bool, Optional[str], Optional[str]]:
    print(f"Checking if query is vague: message={message}, categories={categories}, session={session}")
    message_lower = message.lower().strip()
    
    product_types = []
    theme_keywords = []
    
    greetings = ['hi', 'hey', 'hello']
    if message_lower in greetings:
        print("Detected greeting query")
        return True, None, "greeting"
    
    for product_type in product_types:
        if product_type in message_lower:
            print(f"Detected specific product type: {product_type}")
            return False, product_type, None
    
    for theme in theme_keywords:
        if theme in message_lower:
            print(f"Detected theme: {theme}")
            return False, theme, None
    
    best_match, score = process.extractOne(message_lower, categories, scorer=fuzz.partial_ratio)
    print(f"Fuzzy match against categories: best_match={best_match}, score={score}")
    if score > 80:
        return False, best_match, None
    
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
    
    child_keywords = ['boy', 'girl', 'son', 'daughter', 'child', 'kid', 'baby']
    age_pattern = r'\b(\d+)\s*(?:year|yr|years|old)\b'
    is_child_query = any(keyword in message_lower for keyword in child_keywords)
    age_match = re.search(age_pattern, message_lower)
    
    if is_child_query:
        age = age_match.group(1) if age_match else None
        context = f"child_gift_age_{age}" if age else "child_gift"
        print(f"Detected child-related query: context={context}")
        return True, None, context
    
    print(f"Vague query detected, context={message_lower}")
    return True, None, message_lower

def suggest_categories(query: str, context: str, categories: List[str], headers: dict, recipient_context: Dict[str, str] = None) -> List[str]:
    print(f"Suggesting categories: query={query}, context={context}, categories={categories}, recipient_context={recipient_context}")
    
    filtered_categories = categories
    
    prompt = (
        f"User query: {query}\n\n"
        f"Context: {context}\n\n"
        f"Recipient context: {json.dumps(recipient_context)}\n\n"
        f"Available categories: {', '.join(filtered_categories)}\n\n"
        f"You are a shopping assistant for Zwende. Based on the user's query, context, and recipient context (gender: {recipient_context.get('gender')}, "
        f"relation: {recipient_context.get('relation')}, occasion: {recipient_context.get('occasion')}), suggest 5 to 10 relevant categories from the provided list. "
        f"If the context indicates a gift for a young child (e.g., 'child_gift_age_1'), prioritize age-appropriate categories. "
        f"Recommend based on gender as well (no female category to male or no male category to female) "
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
        
    except Exception as e:
        return filtered_categories[:5]

def identify_subcategory_tags(message: str, category: str, available_tags: List[str], headers: dict) -> List[str]:
    """Identify relevant subcategory tags from the user's message based on available tags."""
    start_time = time.time()
    print(f"[{start_time}] Identifying subcategory tags: message={message}, category={category}, available_tags={available_tags}")
    
    prompt = (
        f"User message: {message}\n\n"
        f"Category: {category}\n\n"
        f"Available tags: {', '.join(available_tags)}\n\n"
        f"You are a shopping assistant for Zwende. Based on the user's message and the specified category, identify the most relevant tags from the provided list that match the user's preferences. "
        f"If the message is vague or does not clearly match any tags, select the most general or popular tags (up to 3). "
        f"Return a JSON object with a single key 'selected_tags' containing a list of selected tag names."
    )
    print(f"Identify subcategory tags prompt: {prompt}")

    payload = {
        "user_id": LYZR_USER_ID,
        "agent_id": LYZR_AGENT_ID,
        "session_id": "subcategory_tag_identification",
        "message": prompt
    }
    
    try:
        agent_response = call_lyzr_api(payload, headers)
        print(f"Agent response for subcategory tags: {agent_response}")
        if isinstance(agent_response, dict) and 'selected_tags' in agent_response:
            selected_tags = [tag for tag in agent_response['selected_tags'] if tag in available_tags][:3]
            print(f"Selected tags: {selected_tags}")
            end_time = time.time()
            print(f"[{end_time}] identify_subcategory_tags completed in {end_time - start_time:.3f} seconds")
            return selected_tags
        else:
            print(f"Invalid LLM response: {agent_response}. Using fallback tags.")
            # Fallback to top 3 available tags
            selected_tags = available_tags[:3]
            print(f"Fallback selected tags: {selected_tags}")
            end_time = time.time()
            print(f"[{end_time}] identify_subcategory_tags completed in {end_time - start_time:.3f} seconds")
            return selected_tags
    except Exception as e:
        print(f"Error identifying subcategory tags: {str(e)}. Using fallback tags.")
        selected_tags = available_tags[:3]
        print(f"Fallback selected tags: {selected_tags}")
        end_time = time.time()
        print(f"[{end_time}] identify_subcategory_tags completed in {end_time - start_time:.3f} seconds")
        return selected_tags

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
                if fuzz.partial_ratio(product_title_lower, current_product.get("Title", "").lower()) > 85:
                    print(f"Found matching product: {current_product}")
                    return current_product
            current_product = {}
        elif line.startswith("  "):
            key, value = line[2:].split(": ", 1) if ": " in line else (line[2:], "")
            if key and value:
                current_product[key] = value
    
    if current_product and fuzz.partial_ratio(product_title_lower, current_product.get("Title", "").lower()) > 85:
        print(f"Found matching product (last): {current_product}")
        return current_product
    
    print("No matching product found")
    return None

def strip_html(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ").strip()

def is_llm_flow(category: str) -> bool:
    """Determine if the category uses the LLM-driven flow."""
    hardcoded_categories = ["Name Plates", "Dolls, Playsets & Toy Figures", "Mugs"]
    return category and category not in hardcoded_categories

def fetch_product_data(
    category: Optional[str] = None,
    suggested_categories: Optional[List[str]] = None,
    product_type: Optional[str] = None,
    theme: Optional[str] = None,
    price_sensitive: bool = False,
    recipient_context: Optional[Dict[str, str]] = None,
    nameplate_type: Optional[str] = None,
    name_hanging_type: Optional[str] = None,
    mug_type: Optional[str] = None,
    subcategory_tags: Optional[List[str]] = None
):
    print(
        f"Fetching product data: category={category}, suggested_categories={suggested_categories}, "
        f"product_type={product_type}, theme={theme}, price_sensitive={price_sensitive}, "
        f"recipient_context={recipient_context}, nameplate_type={nameplate_type}, "
        f"name_hanging_type={name_hanging_type}, mug_type={mug_type}, subcategory_tags={subcategory_tags}"
    )

    # Input validation
    if not any([category, suggested_categories, nameplate_type, name_hanging_type, mug_type]):
        print("At least one of category, suggested_categories, nameplate_type, name_hanging_type, or mug_type must be provided")
        return "Error: Invalid input. Please specify a category or product type.", [], 0

    if subcategory_tags and not category:
        print("subcategory_tags provided without a category")
        return "Error: Subcategory tags require a valid category.", [], 0

    try:
        if DATA_SOURCE == 'database':
            # Base query
            query = """
                SELECT 
                    p."Handle", 
                    p."Variant ID" AS "Variant_ID", 
                    p."Title", 
                    p."Category: Name" AS category_name, 
                    COALESCE(p."Variant Price", 0.0) AS variant_price, 
                    p."URL", 
                    p."Product Description", 
                    p."Vendor", 
                    p."Tags", 
                    p."Image Src", 
                    COALESCE(SUM(o."Price: Total Line Items")::float, 0) AS total_sold
                FROM products p
                LEFT JOIN orders o ON p."Handle" = o."Line: Product Handle" AND p."Variant ID" = o."Line: Variant ID"
                WHERE p."Variant ID" IS NOT NULL
            """
            params = {}

            # Handle LLM flow with subcategory tags
            if subcategory_tags and category and is_llm_flow(category):
                query += ' AND LOWER(p."Category: Name") = LOWER(%(category)s)'
                params['category'] = category
                # Handle up to three tags as per the provided query
                tags_to_use = subcategory_tags[:3]  # Limit to 3 tags
                if not tags_to_use:
                    print("No valid subcategory tags provided for LLM flow")
                    return "Error: No valid subcategory tags provided.", [], 0
                query += ' AND ('
                for i, tag in enumerate(tags_to_use):
                    if i > 0:
                        query += ' OR '
                    query += f'LOWER(p."Tags") LIKE %(tag_{i})s'
                    params[f'tag_{i}'] = f'%{tag.lower()}%'
                    print(f"LLM flow tag parameter: tag_{i} = {params[f'tag_{i}']}")
                query += ')'
            # Handle hardcoded flows
            elif nameplate_type:
                filters = NAMEPLATE_FILTERS.get(nameplate_type.lower())
                if filters:
                    query += ' AND p."Type" = %(type)s'
                    params['type'] = filters["type"]
                    for i, tag in enumerate(filters["tags"]):
                        query += f' AND LOWER(p."Tags") LIKE %(tag_{i})s'
                        params[f'tag_{i}'] = f'%{tag.lower()}%'
                        print(f"Nameplate tag parameter: tag_{i} = {params[f'tag_{i}']}")
            elif name_hanging_type:
                filters = NAME_HANGINGS_FILTERS.get(name_hanging_type.lower())
                if filters:
                    query += ' AND p."Type" = ANY(%(types)s)'
                    params['types'] = filters["type"]
                    for i, tag in enumerate(filters["tags"]):
                        query += f' AND LOWER(p."Tags") LIKE %(tag_{i})s'
                        params[f'tag_{i}'] = f'%{tag.lower()}%'
                        print(f"Name hanging tag parameter: tag_{i} = {params[f'tag_{i}']}")
            elif mug_type:
                filters = MUGS_FILTERS.get(mug_type.lower())
                if filters:
                    query += ' AND p."Type" = ANY(%(types)s)'
                    params['types'] = filters["type"]
                    if "vendor" in filters:
                        query += ' AND p."Vendor" = ANY(%(vendors)s)'
                        params['vendors'] = filters["vendor"]
                    for i, tag in enumerate(filters.get("tags", [])):
                        query += f' AND LOWER(p."Tags") LIKE %(tag_{i})s'
                        params[f'tag_{i}'] = f'%{tag.lower()}%'
                        print(f"Mug tag parameter: tag_{i} = {params[f'tag_{i}']}")
                    if "title_contains" in filters:
                        for i, term in enumerate(filters["title_contains"]):
                            query += f' AND LOWER(p."Title") LIKE %(title_contains_{i})s'
                            params[f'title_contains_{i}'] = f'%{term.lower()}%'
                            print(f"Mug title_contains parameter: title_contains_{i} = {params[f'title_contains_{i}']}")
                    if "title_not_contains" in filters:
                        for i, term in enumerate(filters["title_not_contains"]):
                            query += f' AND LOWER(p."Title") NOT LIKE %(title_not_contains_{i})s'
                            params[f'title_not_contains_{i}'] = f'%{term.lower()}%'
                            print(f"Mug title_not_contains parameter: title_not_contains_{i} = {params[f'title_not_contains_{i}']}")
            # Handle generic category or suggested categories
            elif category:
                query += ' AND LOWER(p."Category: Name") = LOWER(%(category)s)'
                params['category'] = category
            elif suggested_categories:
                query += ' AND LOWER(p."Category: Name") = LOWER(%(category)s)'
                params['categories'] = [c.lower() for c in suggested_categories]

            # Additional filters
            if product_type and not (nameplate_type or name_hanging_type or mug_type):
                query += ' AND (LOWER(p."Title") LIKE %(product_type)s OR LOWER(p."Tags") LIKE %(product_type)s OR LOWER(p."Product Description") LIKE %(product_type)s)'
                params['product_type'] = f'%{product_type.lower()}%'
                print(f"Product type parameter: product_type = {params['product_type']}")
            if theme:
                query += ' AND (LOWER(p."Tags") LIKE %(theme)s OR LOWER(p."Title") LIKE %(theme)s OR LOWER(p."Product Description") LIKE %(theme)s)'
                params['theme'] = f'%{theme.lower()}%'
                print(f"Theme parameter: theme = {params['theme']}")

            # Recipient context filters
            if recipient_context and recipient_context.get('gender') == 'male':
                query += ' AND NOT (LOWER(p."Tags") LIKE %(tag_earrings)s OR LOWER(p."Tags") LIKE %(tag_necklace)s OR LOWER(p."Tags") LIKE %(tag_jewelry)s OR LOWER(p."Category: Name") = ANY(%(excluded_categories)s))'
                params['tag_earrings'] = '%earrings%'
                params['tag_necklace'] = '%necklace%'
                params['tag_jewelry'] = '%jewelry%'
                params['excluded_categories'] = ['jewelry sets', 'earrings']
                print(f"Gender-based parameters: tag_earrings={params['tag_earrings']}, tag_necklace={params['tag_necklace']}, tag_jewelry={params['tag_jewelry']}, excluded_categories={params['excluded_categories']}")
            
            if recipient_context and recipient_context.get('occasion') == 'anniversary':
                query += ' AND (LOWER(p."Tags") LIKE %(tag_personalized)s OR LOWER(p."Tags") LIKE %(tag_wedding)s OR LOWER(p."Tags") LIKE %(tag_love)s OR LOWER(p."Category: Name") = ANY(%(anniversary_categories)s))'
                params['tag_personalized'] = '%personalized%'
                params['tag_wedding'] = '%wedding%'
                params['tag_love'] = '%love%'
                params['anniversary_categories'] = ['home decor',  'decor']
                print(f"Anniversary-based parameters: tag_personalized={params['tag_personalized']}, tag_wedding={params['tag_wedding']}, tag_love={params['tag_love']}, anniversary_categories={params['anniversary_categories']}")

            # Finalize query
            query += ' GROUP BY p."Handle", p."Variant ID", p."Title", p."Category: Name", p."Variant Price", p."URL", p."Product Description", p."Vendor", p."Tags", p."Image Src"'
            query += ' ORDER BY ' + ('p."Variant Price" ASC' if price_sensitive else 'total_sold DESC')
            query += ' LIMIT 50'

            print(f"Executing product data query: {query}")
            print(f"Query parameters: {params}")
            try:
                subset = pd.read_sql_query(query, DB_ENGINE, params=params)
            except DatabaseError as db_err:
                print(f"Database query error: {str(db_err)}")
                return f"Error querying database: {str(db_err)}", [], 0

            print(f"Product data query result: {len(subset)} rows")
            print(f"DataFrame columns: {list(subset.columns)}")
            print(f"Sample data (first 2 rows): {subset.head(2).to_dict(orient='records')}")

            if subset.empty:
                print("No products found for the specified criteria.")
                return f"No products found for category '{category}' with specified filters.", [], 0

            if 'variant_price' not in subset.columns:
                print("Error: 'variant_price' column missing in DataFrame")
                return "Error: 'variant_price' column not found in database query results.", [], 0

            # Clean NaN and infinite values
            nan_count = subset['variant_price'].isna().sum()
            inf_count = subset['variant_price'].isin([float('inf'), float('-inf')]).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"Found {nan_count} NaN and {inf_count} infinite values in variant_price")
            subset['variant_price'] = subset['variant_price'].replace([np.nan, float('inf'), float('-inf')], 0.0)

            # Filter unique products with valid images
            unique_subset = []
            seen_handles = set()
            for _, row in subset.iterrows():
                handle = row['Handle']
                if handle not in seen_handles and row['Image Src'] and row['Image Src'] != 'None':
                    unique_subset.append(row)
                    seen_handles.add(handle)
            subset = pd.DataFrame(unique_subset)
            print(f"Filtered unique products: {len(subset)} rows")

            if subset.empty:
                print("No valid products found after filtering.")
                return "No valid products found (all products have missing images or invalid data).", [], 0

            records = subset.to_dict(orient='records')
            products = [
                {
                    "product name": record.get('Title', 'Unknown Product'),
                    "description": strip_html(record.get('Product Description', '')),
                    "price": float(record.get('variant_price', 0.0)),  # Ensure valid float
                    "Link to product": record.get('URL'),
                    "Image URL": record.get('Image Src')
                } for record in records[:10]
            ]
            total_products = len(records)
            print(f"Generated {len(products)} products, total available: {total_products}")
            return None, products, total_products

    except DatabaseError as db_err:
        print(f"Database error: {str(db_err)}")
        return f"Error accessing database: {str(db_err)}", [], 0
    except pd.io.sql.DatabaseError as pd_err:
        print(f"Pandas database error: {str(pd_err)}")
        return f"Error processing database query: {str(pd_err)}", [], 0
    except Exception as e:
        print(f"Unexpected error fetching product data: {str(e)}")
        return f"Unexpected error fetching products: {str(e)}", [], 0

def extract_recipient_context(message: str) -> Dict[str, str]:
    message_lower = message.lower().strip()
    context = {"gender": None, "relation": None, "occasion": None}

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

    # Initialize session if not exists
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
            "recipient_context": {},
            "flow_state": "initial",
            "subcategory_tags": []
        }
        print(f"Initialized new session: {SESSION_STATE[request.session_id]}")

    session = SESSION_STATE[request.session_id]
    message_lower = request.message.lower().strip()
    session["user_responses"].append(request.message)
    print(f"Appended user response: {session['user_responses']}")

    # Extract recipient context
    session["recipient_context"] = extract_recipient_context(request.message)
    print(f"Updated recipient context: {session['recipient_context']}")

    # Fetch categories
    categories = json.loads(get_categories().body.decode('utf-8'))
    print(f"Categories for chat: {categories}")

    # Enhanced intent detection
    session["intent"] = detect_intent(request.message, session, headers)
    specific_product = None

    # Helper function to determine if flow is LLM-driven
    def is_llm_flow(category):
        hardcoded_categories = ["Name Plates", "Dolls, Playsets & Toy Figures", "Mugs"]
        return category not in hardcoded_categories

    # Helper function for LLM-based category suggestion
    def suggest_categories_llm(
        query: str,
        context: str,
        categories: List[str],
        headers: dict,
        recipient_context: Dict[str, str],
        session_id: str
    ) -> Tuple[Optional[str], List[str], str]:
        """Suggest categories and generate a natural follow-up question using LLM."""
        print(f"Suggesting categories via LLM: query={query}, context={context}, categories={categories}, recipient_context={recipient_context}")
        
        filtered_categories = categories
        
        conversation_history = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(SESSION_STATE.get(session_id, {}).get('questions_asked', []), SESSION_STATE.get(session_id, {}).get('user_responses', []))
        ) or "No conversation history available."

        prompt = (
            f"User query: {query}\n\n"
            f"Context: {context}\n\n"
            f"Recipient context: {json.dumps(recipient_context)}\n\n"
            f"Conversation history: {conversation_history}\n\n"
            f"Available categories: {', '.join(filtered_categories)}\n\n"
            f"You are a shopping assistant for Zwende. Based on the user's query, context, recipient context, and conversation history, "
            f"identify the most relevant category (if clearly specified) and suggest 5–10 relevant categories from the provided list. "
            f"Additionally, generate a natural, engaging, and context-aware follow-up question to prompt the user to select a category or clarify their intent. "
            f"Consider the recipient context (gender: {recipient_context.get('gender')}, relation: {recipient_context.get('relation')}, "
            f"occasion: {recipient_context.get('occasion')}) to prioritize appropriate categories and tailor the response. "
            f"For example:\n"
            f"- If the query mentions 'gift for brother', prioritize categories like 'Wallets & Money Clips', 'Personalized Gifts', or 'Drinkware' and ask, "
            f"'Are you looking for a stylish wallet, a personalized gift, or perhaps some unique drinkware for your brother?'.\n"
            f"- If the query mentions 'birthday gift for child', prioritize 'Dolls, Playsets & Toy Figures' or 'Personalized Gifts' and ask, "
            f"'Would you like to explore fun toys or personalized gifts for the child's birthday?'.\n"
            f"- If the query mentions 'marriage anniversary', prioritize couple-oriented categories and ask, "
            f"'Are you looking for a romantic gift for your anniversary, like personalized decor or couple mugs?'.\n"
            f"- If the query mentions 'art and craft', prioritize 'Arts & Crafts' or 'Art & Craft Kits' and ask, "
            f"'Do you want to dive into creative art and craft kits or explore handcrafted art pieces?'.\n"
            f"'I’d love to help you find the perfect gift! Are you thinking of something like home decor, personalized items, or maybe unique drinkware?'.\n"
            f"Avoid generic phrases like 'It looks like you're looking for something special' or 'I'm not sure exactly what you're looking for'. "
            f"Instead, craft a response that feels personalized and directly relates to the user's query or context. "
            f"Return a JSON object with three keys:\n"
            f"- 'selected_category': The most relevant category if clearly identified, otherwise null.\n"
            f"- 'suggested_categories': A list of 5–10 suggested categories from the available categories.\n"
            f"- 'response': A natural, engaging follow-up question listing the suggested categories as options or prompting for clarification."
        )
        print(f"LLM category suggestion prompt: {prompt}")

        payload = {
            "user_id": LYZR_USER_ID,
            "agent_id": LYZR_AGENT_ID,
            "session_id": session_id,
            "message": prompt
        }
        
        try:
            agent_response = call_lyzr_api(payload, headers)
            print(f"LLM category suggestion response: {agent_response}")
            if isinstance(agent_response, dict):
                selected_category = agent_response.get('selected_category')
                suggested_categories = agent_response.get('suggested_categories', [])
                response = agent_response.get('response', '')
                valid_categories = [cat for cat in suggested_categories if cat in filtered_categories][:10]
                if selected_category and selected_category in filtered_categories:
                    print(f"LLM identified category: {selected_category}, suggested: {valid_categories}, response: {response}")
                    return selected_category, valid_categories, response
                print(f"No specific category identified, suggested: {valid_categories}, response: {response}")
                return None, valid_categories, response
            
            # Fallback if response is not a dict
            print(f"Invalid LLM response: {agent_response}. Using fallback.")
            fallback_response = (
                f"I’d love to help you find the perfect gift! Are you thinking of something like "
                f"{', '.join(filtered_categories[:5])}? Please choose one or tell me more!"
            )
            return None, filtered_categories[:5], fallback_response
            
        except Exception as e:
            print(f"Error in LLM category suggestion: {str(e)}. Using fallback categories.")
            fallback_response = (
                f"I’d love to help you find the perfect gift! Are you thinking of something like "
                f"{', '.join(filtered_categories[:5])}? Please choose one or tell me more!"
            )
            return None, filtered_categories[:5], fallback_response

    # Handle flow states
    if session["flow_state"] == "initial":
        # Check for hardcoded category match using keywords
        identified_category = detect_category(request.message, headers)
        
        if identified_category in ["Name Plates", "Name Hanging", "Mugs"]:
            session["category"] = identified_category
            if identified_category == "Name Plates":
                session["flow_state"] = "nameplate_selection"
                session["questions_asked"].append(NAMEPLATES_MESSAGE["response"])
                print(f"Category 'Name Plates' detected, moving to nameplate_selection")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return NAMEPLATES_MESSAGE
            elif identified_category == "Name Hanging":
                session["flow_state"] = "name_hanging_selection"
                session["questions_asked"].append(NAME_HANGINGS_MESSAGE["response"])
                print(f"Category 'Dolls, Playsets & Toy Figures' detected, moving to name_hanging_selection")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return NAME_HANGINGS_MESSAGE
            elif identified_category == "Mugs":
                session["flow_state"] = "mug_selection"
                session["questions_asked"].append(MUGS_MESSAGE["response"])
                print(f"Category 'Mugs' detected, moving to mug_selection")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return MUGS_MESSAGE

        # If no hardcoded category is detected, proceed with existing logic
        # Check for category match and intent
        message_lower = message_lower.strip()
        matched_category = None
        for cat in categories:
            if cat.lower() in message_lower:
                matched_category = cat
                break
        if not matched_category:  # Fuzzy match for close category names
            best_match, score = process.extractOne(message_lower, categories, scorer=fuzz.partial_ratio)
            if score > 80:
                matched_category = best_match

        # Always check intent first, prioritizing greeting detection
        if session["intent"] == "greeting":
            session["category"] = None
            session["context"] = "greeting"
            session["suggested_categories"] = ["Name Plates", "Dolls, Playsets & Toy Figures", "Mugs"]
            session["stage"] = "follow_up"
            session["flow_state"] = "awaiting_category"
            session["questions_asked"].append(WELCOME_MESSAGE["response"])
            print(f"Greeting detected, returning welcome message with buttons")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return WELCOME_MESSAGE

        # If intent is product_request or category_change, prioritize category detection
        # If intent is product_request or category_change, prioritize category detection
        if session["intent"] in ["product_request", "category_change"]:
            is_vague, specific_item, context = is_query_vague(request.message, categories, session, headers)
            if not is_vague and specific_item:
                print(f"Specific item detected: {specific_item}")
                product = fetch_specific_product(specific_item, session["product_type"], session["theme"])
                if product:
                    response = {
                        "response": f"Found product: {product['Title']}",
                        "type": "interactive_prod",
                        "products": [{
                            "product name": product["Title"],
                            "description": strip_html(product.get("Product Description", "")),
                            "price": float(product.get("Variant Price", 0)),
                            "Link to product": product.get("URL"),
                            "Image URL": product.get("Image Src")
                        }],
                        "metadata": {
                            "total_products": 1
                        }
                    }
                    session["recommended_products"] = response["products"]
                    session["questions_asked"].append(response["response"])
                    print(f"Generated response for specific product: {response['response']}")
                    end_time = time.time()
                    print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                    return response

            # Try to detect category using LLM
            matched_category = detect_category(request.message, headers)
            if matched_category:
                session["category"] = matched_category
                session["flow_state"] = "subcategory_selection"
                session["stage"] = "subcategory_selection"
                print(f"Category {matched_category} detected via LLM, moving to subcategory_selection")

                tags = get_distinct_tags(matched_category)
                session["subcategory_tags"] = tags
                if not tags:
                    print(f"No tags found for category {matched_category}, proceeding to recommendation")
                    session["flow_state"] = "recommendation"
                    session["stage"] = "recommendation"
                    error_message, recommended_products, total_products = fetch_product_data(
                        category=session["category"],
                        price_sensitive=session["price_sensitive"],
                        recipient_context=session["recipient_context"],
                        subcategory_tags=[f"category:{matched_category.lower()}"]
                    )
                    session["last_product_info"] = error_message
                    session["recommended_products"] = recommended_products
                    session["questions_asked"].append("Recommendation provided")

                    response = {
                        "response": f"Here are the top products in {matched_category}:\n{error_message or 'Products retrieved successfully.'}",
                        "type": "interactive_prod",
                        "products": recommended_products,
                        "metadata": {
                            "total_products": total_products
                        }
                    }
                    print(f"Generated response: {response['response'][:500]}... (truncated)")
                    end_time = time.time()
                    print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                    return response

                prompt = (
                    f"Category: {session['category']}\n\n"
                    f"Available tags (subcategories): {', '.join(tags)}\n\n"
                    f"You are a shopping assistant for Zwende. Generate a user-friendly follow-up question to help the user narrow down their preferences within the {session['category']} category. "
                    f"Focus on the theme (e.g., contemporary, traditional, minimalist) and type (e.g., specific product types relevant to the category) based on the provided tags. "
                    f"Create a clear, engaging, and concise question suitable for a UI, presenting the available tags as polished options without prefixes like 'type of:', 'style:', or 'category:'. "
                    f"For each option, include a brief, appealing description in parentheses to enhance user understanding (e.g., for Drinkware: 'Personalized (custom names or photos)', for Earrings: 'Danglers (vibrant, hanging designs)'). "
                    f"Ensure the question encourages the user to select a specific style or type, avoiding technical tag formats in the displayed text. "
                    f"Return a JSON object with two keys:\n"
                    f"- 'response': The follow-up question with formatted options as shown above.\n"
                    f"- 'options': The list of original tag names (e.g., 'type of earring:danglers', 'style:contemporary') for backend use."
                )
                payload = {
                    "user_id": LYZR_USER_ID,
                    "agent_id": LYZR_AGENT_ID,
                    "session_id": request.session_id,
                    "message": prompt
                }
                agent_response = call_lyzr_api(payload, headers) or {
                    "response": f"Which type of {session['category']} would you like to explore? 🎨✨\n" + "\n".join([f"- {tag.split(':')[-1].title()} (unique, handcrafted design)" for tag in tags[:4]]),
                }
                print(f"Agent response for subcategory question: {agent_response}")

                session["questions_asked"].append(agent_response["response"])
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return {
                    "response": agent_response["response"],
                    "type": "text",
                }

            # If no category is matched, use LLM to suggest categories
            selected_category, suggested_categories, llm_response = suggest_categories_llm(
                query=request.message,
                context=session.get("context", "unknown"),
                categories=categories,
                headers=headers,
                recipient_context=session["recipient_context"],
                session_id=request.session_id
            )
            session["suggested_categories"] = suggested_categories
            session["stage"] = "follow_up"
            session["flow_state"] = "awaiting_category"

            if selected_category:
                session["category"] = selected_category
                session["flow_state"] = "subcategory_selection"
                session["stage"] = "subcategory_selection"
                print(f"LLM identified category: {selected_category}, moving to subcategory_selection")

                tags = get_distinct_tags(selected_category)
                session["subcategory_tags"] = tags
                if not tags:
                    print(f"No tags found for category {selected_category}, proceeding to recommendation")
                    session["flow_state"] = "recommendation"
                    session["stage"] = "recommendation"
                    error_message, recommended_products, total_products = fetch_product_data(
                        category=session["category"],
                        price_sensitive=session["price_sensitive"],
                        recipient_context=session["recipient_context"],
                        subcategory_tags=[f"category:{selected_category.lower()}"]
                    )
                    session["last_product_info"] = error_message
                    session["recommended_products"] = recommended_products
                    session["questions_asked"].append("Recommendation provided")

                    response = {
                        "response": f"Here are the top products in {selected_category}:\n{error_message or 'Products retrieved successfully.'}",
                        "type": "interactive_prod",
                        "products": recommended_products,
                        "metadata": {
                            "total_products": total_products
                        }
                    }
                    print(f"Generated response: {response['response'][:500]}... (truncated)")
                    end_time = time.time()
                    print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                    return response

                prompt = (
                    f"Category: {session['category']}\n\n"
                    f"Available tags (subcategories): {', '.join(tags)}\n\n"
                    f"You are a shopping assistant for Zwende. Generate a user-friendly follow-up question to help the user narrow down their preferences within the {session['category']} category. "
                    f"Focus on the theme (e.g., contemporary, traditional, minimalist) and type (e.g., specific product types relevant to the category) based on the provided tags. "
                    f"Create a clear, engaging, and concise question suitable for a UI, presenting the available tags as polished options without prefixes like 'type of:', 'style:', or 'category:'. "
                    f"For each option, include a brief, appealing description in parentheses to enhance user understanding (e.g., for Drinkware: 'Personalized (custom names or photos)', for Earrings: 'Danglers (vibrant, hanging designs)'). "
                    f"Ensure the question encourages the user to select a specific style or type, avoiding technical tag formats in the displayed text. "
                    f"Return a JSON object with two keys:\n"
                    f"- 'response': The follow-up question with formatted options as shown above.\n"
                    f"- 'options': The list of original tag names (e.g., 'type of earring:danglers', 'style:contemporary') for backend use."
                )
                payload = {
                    "user_id": LYZR_USER_ID,
                    "agent_id": LYZR_AGENT_ID,
                    "session_id": request.session_id,
                    "message": prompt
                }
                agent_response = call_lyzr_api(payload, headers) or {
                    "response": f"Which type of {session['category']} would you like to explore? 🎨✨\n" + "\n".join([f"- {tag.split(':')[-1].title()} (unique, handcrafted design)" for tag in tags[:4]]),
                }
                print(f"Agent response for subcategory question: {agent_response}")

                session["questions_asked"].append(agent_response["response"])
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return {
                    "response": agent_response["response"],
                    "type": "text",
                }

            # If no specific category, return LLM-generated response
            response = {
                "response": llm_response or (
                    f"I’d love to help you find the perfect gift! Are you thinking of something like "
                    f"{', '.join(suggested_categories)}? Please choose one or tell me more!"
                ),
                "type": "text",
            }
            session["questions_asked"].append(response["response"])
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response

    elif session["flow_state"] == "awaiting_category":
        # Map user input to categories, including suggested categories from previous response
        category_map = {
            "1": "Name Plates",
            "nameplates for home": "Name Plates",
            "name plates": "Name Plates",
            "2": "Dolls, Playsets & Toy Figures",
            "name hangings for kids": "Dolls, Playsets & Toy Figures",
            "3": "Mugs",
            "terracotta mugs & cups": "Mugs",
            "4": "Order Tracking",
            "order tracking": "Order Tracking",
        }
        # Add suggested categories from session to category_map
        for idx, cat in enumerate(session.get("suggested_categories", []), 1):
            category_map[str(idx)] = cat
            category_map[cat.lower()] = cat

        selected_category = None
        for key, value in category_map.items():
            if message_lower == key or key in message_lower:
                selected_category = value
                break

        if specific_product:
            selected_category = session["category"]

        if selected_category:
            if selected_category == "Name Plates":
                session["category"] = selected_category
                session["flow_state"] = "nameplate_selection"
                session["questions_asked"].append(NAMEPLATES_MESSAGE["response"])
                print(f"Nameplates for Home selected, moving to nameplate_selection")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return NAMEPLATES_MESSAGE
            elif selected_category == "Dolls, Playsets & Toy Figures":
                session["category"] = selected_category
                session["flow_state"] = "name_hanging_selection"
                session["questions_asked"].append(NAME_HANGINGS_MESSAGE["response"])
                print(f"Name Hangings for Kids selected, moving to name_hanging_selection")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return NAME_HANGINGS_MESSAGE
            elif selected_category == "Mugs":
                session["category"] = selected_category
                session["flow_state"] = "mug_selection"
                session["questions_asked"].append(MUGS_MESSAGE["response"])
                print(f"Terracotta Mugs & Cups selected, moving to mug_selection")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return MUGS_MESSAGE
            elif selected_category == "Order Tracking":
                session["flow_state"] = "order_tracking"
                session["questions_asked"].append("Please use https://www.zwende.com/pages/track-your-order to track your order")
                print(f"Order Tracking selected")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return {
                    "response": "Please use https://www.zwende.com/pages/track-your-order to track your order",
                    "type": "text"
                }
            else:
                # Handle other categories (e.g., Drinkware, Wallets & Money Clips)
                session["category"] = selected_category
                session["flow_state"] = "subcategory_selection"
                session["stage"] = "subcategory_selection"
                print(f"Category {selected_category} selected, moving to subcategory_selection")

                tags = get_distinct_tags(selected_category)
                session["subcategory_tags"] = tags
                if not tags:
                    print(f"No tags found for category {selected_category}, proceeding to recommendation")
                    session["flow_state"] = "recommendation"
                    session["stage"] = "recommendation"
                    error_message, recommended_products, total_products = fetch_product_data(
                        category=session["category"],
                        price_sensitive=session["price_sensitive"],
                        recipient_context=session["recipient_context"],
                        subcategory_tags=[f"category:{selected_category.lower()}"]
                    )
                    session["last_product_info"] = error_message
                    session["recommended_products"] = recommended_products
                    session["questions_asked"].append("Recommendation provided")

                    response = {
                        "response": f"Here are the top products in {selected_category}:\n{error_message or 'Products retrieved successfully.'}",
                        "type": "interactive_prod",
                        "products": recommended_products,
                        "metadata": {
                            "total_products": total_products
                        }
                    }
                    print(f"Generated response: {response['response'][:500]}... (truncated)")
                    end_time = time.time()
                    print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                    return response

                prompt = (
                    f"Category: {session['category']}\n\n"
                    f"Available tags (subcategories): {', '.join(tags)}\n\n"
                    f"You are a shopping assistant for Zwende. Generate a user-friendly follow-up question to help the user narrow down their preferences within the {session['category']} category. "
                    f"Focus on the theme (e.g., contemporary, traditional, minimalist) and type (e.g., specific product types relevant to the category) based on the provided tags. "
                    f"Create a clear, engaging, and concise question suitable for a UI, presenting the available tags as polished options without prefixes like 'type of:', 'style:', or 'category:'. "
                    f"For each option, include a brief, appealing description in parentheses to enhance user understanding (e.g., for Drinkware: 'Personalized (custom names or photos)', for Earrings: 'Danglers (vibrant, hanging designs)'). "
                    f"Ensure the question encourages the user to select a specific style or type, avoiding technical tag formats in the displayed text. "
                    f"Example format for a generic category:\n"
                    f"Which type of {session['category']} would you like to explore? 🎨✨\n"
                    f"- Option 1 (description of style or type)\n"
                    f"- Option 2 (description of style or type)\n"
                    f"- Option 3 (description of style or type)\n"
                    f"Return a JSON object with two keys:\n"
                    f"- 'response': The follow-up question with formatted options as shown above.\n"
                    f"- 'options': The list of original tag names (e.g., 'type of earring:danglers', 'style:contemporary') for backend use."
                )
                payload = {
                    "user_id": LYZR_USER_ID,
                    "agent_id": LYZR_AGENT_ID,
                    "session_id": request.session_id,
                    "message": prompt
                }
                agent_response = call_lyzr_api(payload, headers) or {
                    "response": f"Which type of {session['category']} would you like to explore? 🎨✨\n" + "\n".join([f"- {tag.split(':')[-1].title()} (unique, handcrafted design)" for tag in tags[:4]]),
                    # "options": tags[:4]
                }
                print(f"Agent response for subcategory question: {agent_response}")

                session["questions_asked"].append(agent_response["response"])
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return {
                    "response": agent_response["response"],
                    "type": "text",
                    # "options": agent_response["options"]
                }

        # Handle "Others" or vague input with LLM-based category suggestion
        session["category"] = None
        session["flow_state"] = "clarify_others"
        session["stage"] = "clarify_others"
        print(f"Category 'Others' or vague input selected, moving to clarify_others")

        selected_category, suggested_categories, llm_response = suggest_categories_llm(
            query=request.message,
            context=session.get("context", "unknown"),
            categories=categories,
            headers=headers,
            recipient_context=session["recipient_context"],
            session_id=request.session_id
        )
        session["suggested_categories"] = suggested_categories

        if selected_category:
            session["category"] = selected_category
            session["flow_state"] = "subcategory_selection"
            session["stage"] = "subcategory_selection"
            print(f"LLM identified category: {selected_category}, moving to subcategory_selection")

            tags = get_distinct_tags(selected_category)
            session["subcategory_tags"] = tags
            if not tags:
                print(f"No tags found for category {selected_category}, proceeding to recommendation")
                session["flow_state"] = "recommendation"
                session["stage"] = "recommendation"
                error_message, recommended_products, total_products = fetch_product_data(
                    category=session["category"],
                    price_sensitive=session["price_sensitive"],
                    recipient_context=session["recipient_context"],
                    subcategory_tags=[f"category:{selected_category.lower()}"]
                )
                session["last_product_info"] = error_message
                session["recommended_products"] = recommended_products
                session["questions_asked"].append("Recommendation provided")

                response = {
                    "response": f"Here are the top products in {selected_category}:\n{error_message or 'Products retrieved successfully.'}",
                    "type": "interactive_prod",
                    "products": recommended_products,
                    "metadata": {
                        "total_products": total_products
                    }
                }
                print(f"Generated response: {response['response'][:500]}... (truncated)")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return response

            prompt = (
                f"Category: {session['category']}\n\n"
                f"Available tags (subcategories): {', '.join(tags)}\n\n"
                f"You are a shopping assistant for Zwende. Generate a user-friendly follow-up question to help the user narrow down their preferences within the {session['category']} category. "
                f"Focus on the theme (e.g., contemporary, traditional, minimalist) and type (e.g., specific product types relevant to the category) based on the provided tags. "
                f"Create a clear, engaging, and concise question suitable for a UI, presenting the available tags as polished options without prefixes like 'type of:', 'style:', or 'category:'. "
                f"For each option, include a brief, appealing description in parentheses to enhance user understanding (e.g., for Drinkware: 'Personalized (custom names or photos)', for Earrings: 'Danglers (vibrant, hanging designs)'). "
                f"Ensure the question encourages the user to select a specific style or type, avoiding technical tag formats in the displayed text. "
                f"Example format for a generic category:\n"
                f"Which type of {session['category']} would you like to explore? 🎨✨\n"
                f"- Option 1 (description of style or type)\n"
                f"- Option 2 (description of style or type)\n"
                f"- Option 3 (description of style or type)\n"
                f"Return a JSON object with two keys:\n"
                f"- 'response': The follow-up question with formatted options as shown above.\n"
                f"- 'options': The list of original tag names (e.g., 'type of earring:danglers', 'style:contemporary') for backend use."
            )
            payload = {
                "user_id": LYZR_USER_ID,
                "agent_id": LYZR_AGENT_ID,
                "session_id": request.session_id,
                "message": prompt
            }
            agent_response = call_lyzr_api(payload, headers) or {
                "response": f"Which type of {session['category']} would you like to explore? 🎨✨\n" + "\n".join([f"- {tag.split(':')[-1].title()} (unique, handcrafted design)" for tag in tags[:4]]),
                # "options": tags[:4]
            }
            print(f"Agent response for subcategory question: {agent_response}")

            session["questions_asked"].append(agent_response["response"])
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return {
                "response": agent_response["response"],
                "type": "text",
                # "options": agent_response["options"]
            }

        # If no specific category is identified, ask for clarification with suggested categories
        response = {
            "response": (
                f"{llm_response}" +
                "\n".join([f"- {cat}" for cat in suggested_categories]) +
                "\nPlease choose one or tell me more about what you're looking for!"
            ),
            "type": "text",
            # "options": suggested_categories
        }
        session["questions_asked"].append(response["response"])
        end_time = time.time()
        print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
        return response

    elif session["flow_state"] == "clarify_others":
        # Use LLM to suggest categories and generate a follow-up question
        selected_category, suggested_categories, llm_response = suggest_categories_llm(
            query=request.message,
            context=session.get("context", "unknown"),
            categories=categories,
            headers=headers,
            recipient_context=session["recipient_context"],
            session_id=request.session_id
        )
        session["suggested_categories"] = suggested_categories

        if selected_category:
            session["category"] = selected_category
            session["flow_state"] = "subcategory_selection"
            session["stage"] = "subcategory_selection"
            print(f"LLM identified category: {selected_category}, moving to subcategory_selection")

            tags = get_distinct_tags(selected_category)
            session["subcategory_tags"] = tags
            if not tags:
                print(f"No tags found for category {selected_category}, proceeding to recommendation")
                session["flow_state"] = "recommendation"
                session["stage"] = "recommendation"
                error_message, recommended_products, total_products = fetch_product_data(
                    category=session["category"],
                    price_sensitive=session["price_sensitive"],
                    recipient_context=session["recipient_context"],
                    subcategory_tags=[f"category:{selected_category.lower()}"]
                )
                session["last_product_info"] = error_message
                session["recommended_products"] = recommended_products
                session["questions_asked"].append("Recommendation provided")

                response = {
                    "response": f"Here are the top products in {selected_category}:\n{error_message or 'Products retrieved successfully.'}",
                    "type": "interactive_prod",
                    "products": recommended_products,
                    "metadata": {
                        "total_products": total_products
                    }
                }
                print(f"Generated response: {response['response'][:500]}... (truncated)")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return response

            prompt = (
                f"Category: {session['category']}\n\n"
                f"Available tags (subcategories): {', '.join(tags)}\n\n"
                f"You are a shopping assistant for Zwende. Generate a user-friendly follow-up question to help the user narrow down their preferences within the {session['category']} category. "
                f"Focus on the theme (e.g., contemporary, traditional, minimalist) and type (e.g., specific product types relevant to the category) based on the provided tags. "
                f"Create a clear, engaging, and concise question suitable for a UI, presenting the available tags as polished options without prefixes like 'type of:', 'style:', or 'category:'. "
                f"For each option, include a brief, appealing description in parentheses to enhance user understanding (e.g., for Mugs: 'Personalized (custom names or photos)', for Earrings: 'Danglers (vibrant, hanging designs)'). "
                f"Ensure the question encourages the user to select a specific style or type, and avoid including raw tag names (e.g., 'style:minimalist') in the response text. "
                f"Example format for a generic category:\n"
                f"Which type of {session['category']} would you like to explore? 🎨✨\n"
                f"- Option 1 (description of style or type)\n"
                f"- Option 2 (description of style or type)\n"
                f"- Option 3 (description of style or type)\n"
                f"Return a JSON object with two keys:\n"
                f"- 'response': The follow-up question with formatted options as shown above.\n"
                f"- 'options': The list of original tag names (e.g., 'type of earring:danglers', 'style:contemporary') for backend use."
            )
            payload = {
                "user_id": LYZR_USER_ID,
                "agent_id": LYZR_AGENT_ID,
                "session_id": request.session_id,
                "message": prompt
            }
            agent_response = call_lyzr_api(payload, headers) or {
                "response": f"Which type of {session['category']} would you like to explore? 🎨✨\n" + "\n".join([f"- {tag.split(':')[-1].title()} (unique, handcrafted design)" for tag in tags[:4]]),
                "options": tags[:4]
            }
            print(f"Agent response for subcategory question: {agent_response}")

            # Sanitize response to ensure no raw tags in response text
            response_text = agent_response["response"]
            for tag in tags:
                if tag in response_text:
                    response_text = response_text.replace(tag, tag.split(':')[-1].title())
            agent_response["response"] = response_text

            session["questions_asked"].append(agent_response["response"])
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return {
                "response": agent_response["response"],
                "type": "text",
                # "options": agent_response["options"]
            }

        # Use LLM-generated response instead of hardcoded message
        response = {
            "response": llm_response or (
                f"I’d love to help you find the perfect item! Are you thinking of something like "
                f"{', '.join(suggested_categories)}? Please choose one or tell me more!"
            ),
            "type": "text",
            "options": suggested_categories
        }
        session["questions_asked"].append(response["response"])
        end_time = time.time()
        print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
        return response

    elif session["flow_state"] == "nameplate_selection":
        nameplate_type_map = {
            "1": "outdoor name plates",
            "outdoor name plates": "outdoor name plates",
            "2": "minimal name plates",
            "minimal name plates": "minimal name plates",
            "3": "photo personalized nameplate",
            "photo personalized nameplate": "photo personalized nameplate",
            "4": "ethnic nameplates",
            "ethnic nameplates": "ethnic nameplates",
            "5": "vernacular nameplates",
            "vernacular nameplates": "vernacular nameplates"
        }
        selected_nameplate_type = None
        for key, value in nameplate_type_map.items():
            if message_lower == key or key in message_lower:
                selected_nameplate_type = value
                break

        if selected_nameplate_type and selected_nameplate_type in NAMEPLATE_FILTERS:
            session["flow_state"] = "recommendation"
            session["stage"] = "recommendation"
            session["product_type"] = "Nameboards"
            filters = NAMEPLATE_FILTERS[selected_nameplate_type]
            print(f"Nameplate type {selected_nameplate_type} selected, fetching products")

            error_message, recommended_products, total_products = fetch_product_data(
                category=session["category"],
                product_type=filters["type"],
                nameplate_type=selected_nameplate_type,
                price_sensitive=session["price_sensitive"],
                recipient_context=session["recipient_context"]
            )
            session["last_product_info"] = error_message
            session["recommended_products"] = recommended_products
            session["questions_asked"].append("Recommendation provided")

            if error_message:
                print(f"Error in fetch_product_data: {error_message}")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return {
                    "response": error_message,
                    "type": "text"
                }

            representative_image = recommended_products[0]["Image URL"] if recommended_products else None
            response = {
                "response": filters["response"],
                "type": "interactive_prod",
                "image_url": representative_image,
                "products": recommended_products,
                "metadata": {
                    "total_products": total_products
                }
            }
            print(f"Generated response: {response['response']} with {len(recommended_products)} products")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response
        else:
            session["questions_asked"].append(NAMEPLATES_MESSAGE["response"])
            print(f"Invalid nameplate selection, repeating options")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return NAMEPLATES_MESSAGE

    elif session["flow_state"] == "name_hanging_selection":
        name_hanging_type_map = {
            "1": "unicorn theme",
            "unicorn theme": "unicorn theme",
            "2": "superhero theme",
            "superhero theme": "superhero theme",
            "3": "swing designs",
            "swing designs": "swing designs",
            "4": "moon designs",
            "moon designs": "moon designs",
            "5": "rainbow designs",
            "rainbow designs": "rainbow designs",
            "6": "jungle theme",
            "jungle theme": "jungle theme",
            "7": "themes for boys",
            "themes for boys": "themes for boys",
            "8": "themes for girls",
            "themes for girls": "themes for girls",
            "9": "space theme",
            "space theme": "space theme"
        }
        selected_name_hanging_type = None
        for key, value in name_hanging_type_map.items():
            if message_lower == key or key in message_lower:
                selected_name_hanging_type = value
                break

        if selected_name_hanging_type and selected_name_hanging_type in NAME_HANGINGS_FILTERS:
            session["flow_state"] = "recommendation"
            session["stage"] = "recommendation"
            session["product_type"] = "Nameboards"
            filters = NAME_HANGINGS_FILTERS[selected_name_hanging_type]
            print(f"Name hanging type {selected_name_hanging_type} selected, fetching products")

            error_message, recommended_products, total_products = fetch_product_data(
                category=session["category"],
                product_type=filters["type"][0],
                name_hanging_type=selected_name_hanging_type,
                price_sensitive=session["price_sensitive"],
                recipient_context=session["recipient_context"]
            )
            session["last_product_info"] = error_message
            session["recommended_products"] = recommended_products
            session["questions_asked"].append("Recommendation provided")

            if error_message:
                print(f"Error in fetch_product_data: {error_message}")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return {
                    "response": error_message,
                    "type": "text"
                }

            representative_image = recommended_products[0]["Image URL"] if recommended_products else None
            response = {
                "response": filters["response"],
                "type": "interactive_prod",
                "image_url": representative_image,
                "products": recommended_products,
                "metadata": {
                    "total_products": total_products
                }
            }
            print(f"Generated response: {response['response']} with {len(recommended_products)} products")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response
        else:
            session["questions_asked"].append(NAME_HANGINGS_MESSAGE["response"])
            print(f"Invalid name hanging selection, repeating options")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return NAME_HANGINGS_MESSAGE

    elif session["flow_state"] == "mug_selection":
        mug_type_map = {
            "1": "mugs for father's day",
            "mugs for father's day": "mugs for father's day",
            "2": "mugs for kids",
            "mugs for kids": "mugs for kids",
            "3": "mugs for wedding",
            "mugs for wedding": "mugs for wedding",
            "4": "mugs for couple",
            "mugs for couple": "mugs for couple",
            "5": "mugs showing hobbies",
            "mugs showing hobbies": "mugs showing hobbies"
        }
        selected_mug_type = None
        for key, value in mug_type_map.items():
            if message_lower == key or key in message_lower:
                selected_mug_type = value
                break

        if selected_mug_type and selected_mug_type in MUGS_FILTERS:
            session["flow_state"] = "recommendation"
            session["stage"] = "recommendation"
            session["product_type"] = "Mugs"
            filters = MUGS_FILTERS[selected_mug_type]
            print(f"Mug type {selected_mug_type} selected, fetching products")

            error_message, recommended_products, total_products = fetch_product_data(
                category=session["category"],
                product_type=filters["type"][0],
                mug_type=selected_mug_type,
                price_sensitive=session["price_sensitive"],
                recipient_context=session["recipient_context"]
            )
            session["last_product_info"] = error_message
            session["recommended_products"] = recommended_products
            session["questions_asked"].append("Recommendation provided")

            if error_message:
                print(f"Error in fetch_product_data: {error_message}")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return {
                    "response": error_message,
                    "type": "text"
                }

            representative_image = recommended_products[0]["Image URL"] if recommended_products else None
            response = {
                "response": filters["response"],
                "type": "interactive_prod",
                "image_url": representative_image,
                "products": recommended_products,
                "metadata": {
                    "total_products": total_products
                }
            }
            print(f"Generated response: {response['response']} with {len(recommended_products)} products")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response
        else:
            session["questions_asked"].append(MUGS_MESSAGE["response"])
            print(f"Invalid mug selection, repeating options")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return MUGS_MESSAGE

    elif session["flow_state"] == "subcategory_selection":
        available_tags = session.get("subcategory_tags", [])
        if not available_tags:
            print(f"No tags available for category {session['category']}, proceeding to recommendation")
            session["flow_state"] = "recommendation"
            session["stage"] = "recommendation"
            error_message, recommended_products, total_products = fetch_product_data(
                category=session["category"],
                price_sensitive=session["price_sensitive"],
                recipient_context=session["recipient_context"],
                subcategory_tags=[f"category:{session['category'].lower()}"]
            )
            session["last_product_info"] = error_message
            session["recommended_products"] = recommended_products
            session["questions_asked"].append("Recommendation provided")

            response = {
                "response": f"Here are the top products in {session['category']}:\n{error_message or 'Products retrieved successfully.'}",
                "type": "interactive_prod",
                "products": recommended_products,
                "metadata": {
                    "total_products": total_products
                }
            }
            print(f"Generated response: {response['response'][:500]}... (truncated)")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response

        selected_tags = identify_subcategory_tags(message_lower, session["category"], available_tags, headers)
        if not selected_tags:
            selected_tags = available_tags[:3]  # Fallback to top 3 tags
        session["subcategory_tags"] = selected_tags
        session["flow_state"] = "recommendation"
        session["stage"] = "recommendation"
        print(f"Selected subcategory tags: {selected_tags}, moving to recommendation")

        error_message, recommended_products, total_products = fetch_product_data(
            category=session["category"],
            subcategory_tags=selected_tags,
            price_sensitive=session["price_sensitive"],
            recipient_context=session["recipient_context"]
        )
        session["last_product_info"] = error_message
        session["recommended_products"] = recommended_products
        session["questions_asked"].append("Recommendation provided")

        if error_message:
            print(f"Error in fetch_product_data: {error_message}")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return {
                "response": error_message,
                "type": "text"
            }

        response = {
            "response": f"Here are the top products in {session['category']} matching your preferences:\n{', '.join([tag.split(':')[-1].title() for tag in selected_tags])}",
            "type": "interactive_prod",
            "products": recommended_products,
            "metadata": {
                "total_products": total_products
            }
        }
        print(f"Generated response: {response['response']} with {len(recommended_products)} products")
        end_time = time.time()
        print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
        return response

    elif session["flow_state"] == "recommendation":
        if session["intent"] == "category_change":
            session["flow_state"] = "subcategory_selection"
            session["stage"] = "subcategory_selection"
            session["subcategory_tags"] = []
            print(f"Category changed to {session['category']}, moving to subcategory_selection")

            tags = get_distinct_tags(session["category"])
            session["subcategory_tags"] = tags
            if not tags:
                print(f"No tags found for category {session['category']}, proceeding to recommendation")
                error_message, recommended_products, total_products = fetch_product_data(
                    category=session["category"],
                    price_sensitive=session["price_sensitive"],
                    recipient_context=session["recipient_context"],
                    subcategory_tags=[f"category:{session['category'].lower()}"]
                )
                session["last_product_info"] = error_message
                session["recommended_products"] = recommended_products
                session["questions_asked"].append("Recommendation provided")

                response = {
                    "response": f"Here are the top products in {session['category']}:\n{error_message or 'Products retrieved successfully.'}",
                    "type": "interactive_prod",
                    "products": recommended_products,
                    "metadata": {
                        "total_products": total_products
                    }
                }
                print(f"Generated response: {response['response'][:500]}... (truncated)")
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return response

            prompt = (
                f"Category: {session['category']}\n\n"
                f"Available tags (subcategories): {', '.join(tags)}\n\n"
                f"You are a shopping assistant for Zwende. Generate a user-friendly follow-up question to help the user narrow down their preferences within the {session['category']} category. "
                f"Focus on the theme (e.g., contemporary, traditional, minimalist) and type (e.g., specific product types relevant to the category) based on the provided tags. "
                f"Create a clear, engaging, and concise question suitable for a UI, presenting the available tags as polished options without prefixes like 'type of:', 'style:', or 'category:'. "
                f"For each option, include a brief, appealing description in parentheses to enhance user understanding (e.g., for Mugs: 'Personalized (custom names or photos)', for Earrings: 'Danglers (vibrant, hanging designs)'). "
                f"Ensure the question encourages the user to select a specific style or type, and avoid including raw tag names (e.g., 'style:minimalist') in the response text. "
                f"Example format for a generic category:\n"
                f"Which type of {session['category']} would you like to explore? 🎨✨\n"
                f"- Option 1 (description of style or type)\n"
                f"- Option 2 (description of style or type)\n"
                f"- Option 3 (description of style or type)\n"
                f"Return a JSON object with two keys:\n"
                f"- 'response': The follow-up question with formatted options as shown above.\n"
                f"- 'options': The list of original tag names (e.g., 'type of earring:danglers', 'style:contemporary') for backend use."
            )
            payload = {
                "user_id": LYZR_USER_ID,
                "agent_id": LYZR_AGENT_ID,
                "session_id": request.session_id,
                "message": prompt
            }
            agent_response = call_lyzr_api(payload, headers) or {
                "response": f"Which type of {session['category']} would you like to explore? 🎨✨\n" + "\n".join([f"- {tag.split(':')[-1].title()} (unique, handcrafted design)" for tag in tags[:4]]),
                "options": tags[:4]
            }
            print(f"Agent response for subcategory question: {agent_response}")

            # Sanitize response to ensure no raw tags in response text
            response_text = agent_response["response"]
            for tag in tags:
                if tag in response_text:
                    response_text = response_text.replace(tag, tag.split(':')[-1].title())
            agent_response["response"] = response_text

            session["questions_asked"].append(agent_response["response"])
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return {
                "response": agent_response["response"],
                "type": "text",
                # "options": agent_response["options"]
            }

        # Handle other intents during recommendation stage
        if session["intent"] == "price_concern":
            session["price_sensitive"] = True
            print("Price concern detected, setting price_sensitive to True")
            error_message, recommended_products, total_products = fetch_product_data(
                category=session["category"],
                product_type=session["product_type"],
                theme=session["theme"],
                price_sensitive=True,
                recipient_context=session["recipient_context"],
                subcategory_tags=session["subcategory_tags"]
            )
            session["last_product_info"] = error_message
            session["recommended_products"] = recommended_products
            session["questions_asked"].append("Recommendation provided with price sensitivity")

            response = {
                "response": f"Here are the top budget-friendly products in {session['category']}:\n{error_message or 'Products retrieved successfully.'}",
                "type": "interactive_prod",
                "products": recommended_products,
                "metadata": {
                    "total_products": total_products
                }
            }
            print(f"Generated response: {response['response']} with {len(recommended_products)} products")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response

        elif session["intent"] == "craftsmanship_explanation":
            print("Craftsmanship explanation requested")
            response = {
                "response": (
                    "Our products at Zwende are handcrafted by skilled artisans across India. "
                    "Each item is made with high-quality, sustainable materials, ensuring uniqueness and durability. "
                    "Would you like to know more about a specific product's materials or continue browsing?"
                ),
                "type": "text"
            }
            session["questions_asked"].append(response["response"])
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response

        elif session["intent"] == "debug_request":
            print("Debug request detected")
            metadata = get_product_metadata(
                category=session["category"],
                product_type=session["product_type"],
                theme=session["theme"],
                max_products=100
            )
            response = {
                "response": f"Debug Info:\nCategory: {session['category']}\nProduct Type: {session['product_type']}\nTheme: {session['theme']}\nMetadata Sample:\n{metadata[:500]}...",
                "type": "text"
            }
            session["questions_asked"].append(response["response"])
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response

        else:
            # Default to product_request or continuation
            is_vague, specific_item, context = is_query_vague(request.message, categories, session, headers)
            if not is_vague and specific_item:
                print(f"Specific item detected: {specific_item}")
                product = fetch_specific_product(specific_item, session["product_type"], session["theme"])
                if product:
                    response = {
                        "response": f"Found product: {product['Title']}",
                        "type": "interactive_prod",
                        "products": [{
                            "product name": product["Title"],
                            "description": strip_html(product.get("Product Description", "")),
                            "price": float(product.get("Variant Price", 0)),
                            "Link to product": product.get("URL"),
                            "Image URL": product.get("Image Src")
                        }],
                        "metadata": {
                            "total_products": 1
                        }
                    }
                    session["recommended_products"] = response["products"]
                    session["questions_asked"].append(response["response"])
                    print(f"Generated response for specific product: {response['response']}")
                    end_time = time.time()
                    print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                    return response

            # If vague or no specific item, refine or continue
            if is_vague:
                print(f"Vague query detected, context: {context}")
                selected_category, suggested_categories, llm_response = suggest_categories_llm(
                    query=request.message,
                    context=context,
                    categories=categories,
                    headers=headers,
                    recipient_context=session["recipient_context"],
                    session_id=request.session_id
                )
                session["suggested_categories"] = suggested_categories
                session["stage"] = "follow_up"
                session["context"] = context

                response = {
                    "response": llm_response or (
                        f"I’d love to help you find the perfect item! Are you thinking of something like "
                        f"{', '.join(suggested_categories)}? Please choose one or tell me more!"
                    ),
                    "type": "text",
                    "options": suggested_categories
                }
                session["questions_asked"].append(response["response"])
                end_time = time.time()
                print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
                return response

            # Continue with current category and tags
            error_message, recommended_products, total_products = fetch_product_data(
                category=session["category"],
                product_type=session["product_type"],
                theme=session["theme"],
                price_sensitive=session["price_sensitive"],
                recipient_context=session["recipient_context"],
                subcategory_tags=session["subcategory_tags"]
            )
            session["last_product_info"] = error_message
            session["recommended_products"] = recommended_products
            session["questions_asked"].append("Recommendation provided")

            response = {
                "response": f"Here are the top products in {session['category']}:\n{error_message or 'Products retrieved successfully.'}",
                "type": "interactive_prod",
                "products": recommended_products,
                "metadata": {
                    "total_products": total_products
                }
            }
            print(f"Generated response: {response['response']} with {len(recommended_products)} products")
            end_time = time.time()
            print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
            return response

    else:
        print(f"Unknown flow_state: {session['flow_state']}, resetting to initial")
        session["flow_state"] = "initial"
        session["category"] = None
        session["suggested_categories"] = []
        session["stage"] = "category_detection"
        session["product_type"] = None
        session["subcategory_tags"] = []
        session["questions_asked"].append(WELCOME_MESSAGE["response"])
        end_time = time.time()
        print(f"[{end_time}] Chat endpoint completed in {end_time - start_time:.3f} seconds")
        return WELCOME_MESSAGE


# Health Check
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}  