import os
import requests
import json
import re
from typing import List, Literal
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Tuple, Dict
import random
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from fastapi.responses import JSONResponse
from collections import defaultdict
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process  # For fuzzy matching
import gdown

load_dotenv()

# --- CONFIGURATION ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
ORDER_FILE = os.path.join(DATA_DIR, "OrderExport230425.xlsx")
PRODUCT_FILE = os.path.join(DATA_DIR, "ProductExport230425.xlsx")
# SMART_COLLECTION_FILE_ID = os.getenv("SMART_COLLECTION_FILE_ID")  # from .env
SMART_FILE = os.path.join(DATA_DIR, "Smart Collection Export.csv")
# SMART_FILE = os.path.join(DATA_DIR, "Smart Collections.csv")

# gdown.download(id="1_dHsJNwBvsqKza2jWuy4Ml1nBTx54eIy", output=SMART_FILE, quiet=False)
TOP_K = 5
MAX_ROWS = int(os.getenv("MAX_ROWS", "50000"))

# Lyzr API configuration
LYZR_API_URL = os.getenv("LYZR_AGENT_URL")
LYZR_API_KEY = os.getenv("LYZR_API_KEY")
LYZR_USER_ID = os.getenv("LYZR_USER_ID")
LYZR_AGENT_ID = os.getenv("LYZR_AGENT_ID")

# Global variables to hold dataframes
SEARCH1_DF = None
SEARCH2_DF = None
CATEGORY_COL = None
COLLECTION_COL = None
SMART_DF = None

# Session state storage
SESSION_STATE = defaultdict(lambda: {
    "category": None,
    "suggested_categories": [],
    "questions_asked": [],
    "user_responses": [],
    "stage": "category_detection",
    "product_metadata": None,
    "price_sensitive": False,
    "intent": None,
    "recommended_products": []  # Store recommended products
})

# Setup lifespan context manager
@asynccontextmanager
async def lifespan(app):
    global SEARCH1_DF, SEARCH2_DF, CATEGORY_COL, COLLECTION_COL, SMART_DF
    try:
        print("Loading data...")
        SEARCH1_DF, SEARCH2_DF, CATEGORY_COL, COLLECTION_COL, SMART_DF = load_and_prepare()
        print(f"Data loaded successfully. Found {len(SEARCH1_DF)} orders, {len(SEARCH2_DF)} collections, {len(SMART_DF)} smart collections.")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    yield

app = FastAPI(title="Zwende Search Agent", version="1.0.0", lifespan=lifespan)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA LOADING & ETL ---
def load_and_prepare():
    print(f"Loading orders from {ORDER_FILE} (max {MAX_ROWS} rows)")
    print(f"Loading products from {PRODUCT_FILE} (max {MAX_ROWS} rows)")
    print(f"Loading smart collections from {SMART_FILE} (max {MAX_ROWS} rows)")
    
    for file_path in [ORDER_FILE, PRODUCT_FILE, SMART_FILE]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        orders = pd.read_excel(ORDER_FILE, nrows=MAX_ROWS)
        products = pd.read_excel(PRODUCT_FILE, nrows=MAX_ROWS)
        smart = pd.read_csv(SMART_FILE, nrows=MAX_ROWS, low_memory=False)
        print(f"Smart Collections columns: {list(smart.columns)}")
    except Exception as e:
        print(f"Error loading data files: {str(e)}")
        print("Creating minimal fallback datasets")
        if 'orders' not in locals() or len(orders) == 0:
            orders = pd.DataFrame({'Line: Product Handle': ['default-product'], 'Line: Variant ID': ['default-variant'], 'quantity': [1]})
        if 'products' not in locals() or len(products) == 0:
            products = pd.DataFrame({
                'Handle': ['default-product'], 'Variant ID': ['default-variant'], 'Title': ['Default Product'],
                'Product Category': ['Default'], 'Variant Price': [0.0], 'URL': ['https://example.com']
            })
        if 'smart' not in locals() or len(smart) == 0:
            smart = pd.DataFrame({'Product: Handle': ['default-product'], 'Title': ['Default Collection']})
    
    print(f"Loaded {len(orders)} orders, {len(products)} products, and {len(smart)} smart collections")

    category_cols = [c for c in products.columns if 'category' in c.lower()]
    if not category_cols:
        print("No category column found in products. Using default 'Product Category'")
        products['Product Category'] = 'Default Category'
        category_col = 'Product Category'
    else:
        category_col = category_cols[0]
        print(f"Using category column: {category_col}")

    if 'Variant ID' not in products.columns:
        print("No Variant ID column found in products. Creating default.")
        products['Variant ID'] = products['Handle'] + '-variant'
    
    orders['quantity'] = 1
    try:
        df_orders = orders.merge(products, left_on=['Line: Product Handle', 'Line: Variant ID'], 
                               right_on=['Handle', 'Variant ID'], how='inner')
        if len(df_orders) == 0:
            print("Warning: No matching products in orders. Creating sample data.")
            handle = products['Handle'].iloc[0] if len(products) > 0 else 'default-product'
            variant_id = products['Variant ID'].iloc[0] if len(products) > 0 else 'default-variant'
            df_orders = pd.DataFrame({
                'Line: Product Handle': [handle], 'Line: Variant ID': [variant_id], 'quantity': [1], 
                'Handle': [handle], 'Variant ID': [variant_id],
                'Title': ['Sample Product'], category_col: ['Sample Category'],
                'Variant Price': [9.99], 'URL': ['https://example.com']
            })
    except Exception as e:
        print(f"Error merging orders with products: {str(e)}")
        df_orders = pd.DataFrame({
            'Line: Product Handle': ['default-product'], 'Line: Variant ID': ['default-variant'], 
            'quantity': [1], 'Handle': ['default-product'], 'Variant ID': ['default-variant'],
            'Title': ['Default Product'], category_col: ['Default Category'],
            'Variant Price': [0.0], 'URL': ['https://example.com']
        })

    if category_col not in df_orders.columns:
        print(f"WARNING: {category_col} not found in merged orders, creating default")
        df_orders[category_col] = 'Default Category'
    
    if 'Handle' not in df_orders.columns or 'Variant ID' not in df_orders.columns:
        print("WARNING: Handle or Variant ID not found in merged orders")
        df_orders['Handle'] = df_orders.get('Handle', 'default-product')
        df_orders['Variant ID'] = df_orders.get('Variant ID', 'default-variant')
    
    sold = df_orders.groupby([category_col, 'Handle', 'Variant ID'], dropna=False)['quantity'].sum().reset_index(name='total_sold')
    try:
        search1 = sold.sort_values([category_col, 'total_sold'], ascending=[True, False]).reset_index(drop=True)
        search1 = search1.merge(products, on=['Handle', 'Variant ID'], how='left', suffixes=('', '_y'))
        if len(search1) == 0:
            print("Warning: Empty search1 after merge. Creating sample data.")
            search1 = pd.DataFrame({
                'Handle': ['default-product'], 'Variant ID': ['default-variant'], 
                'Title': ['Default Product'], category_col: ['Default Category'], 
                'total_sold': [1], 'Variant Price': [0.0], 'URL': ['https://example.com']
            })
    except Exception as e:
        print(f"Error creating search1: {e}")
        search1 = pd.DataFrame({
            'Handle': ['default-product'], 'Variant ID': ['default-variant'], 
            'Title': ['Default Product'], category_col: ['Default Category'], 
            'total_sold': [1], 'Variant Price': [0.0], 'URL': ['https://example.com']
        })

    collection_cols = [c for c in smart.columns if 'title' in c.lower()]
    if not collection_cols:
        print("No collection title column found in smart collections. Using default 'Title'")
        smart['Title'] = 'Default Collection'
        coll_col = 'Title'
    else:
        coll_col = collection_cols[0]
        print(f"Using collection column: {coll_col}")
    
    smart = smart.rename(columns={coll_col: 'collection_title'})
    try:
        if 'Product: Handle' not in smart.columns:
            print("Warning: 'Product: Handle' not found in smart collections")
            product_handle_cols = [c for c in smart.columns if 'handle' in c.lower() and 'product' in c.lower()]
            if product_handle_cols:
                handle_col = product_handle_cols[0]
                print(f"Using alternative column: {handle_col}")
                smart = smart.rename(columns={handle_col: 'Product: Handle'})
            else:
                print("Creating 'Product: Handle' column with default values")
                handle = products['Handle'].iloc[0] if len(products) > 0 else 'default-product'
                smart['Product: Handle'] = handle
        
        df_coll = smart.merge(products, left_on='Product: Handle', right_on='Handle', how='inner')
        if len(df_coll) == 0:
            print("Warning: No matching products in collections. Creating sample data.")
            handle = products['Handle'].iloc[0] if len(products) > 0 else 'default-product'
            df_coll = pd.DataFrame({
                'Product: Handle': [handle], 'Handle': [handle], 'collection_title': ['Sample Collection'],
                'Title': ['Sample Product'], 'Variant Price': [9.99], 'URL': ['https://example.com']
            })
    except Exception as e:
        print(f"Error merging smart collections with products: {str(e)}")
        df_coll = pd.DataFrame({
            'Product: Handle': ['default-product'], 'Handle': ['default-product'],
            'collection_title': ['Default Collection'], 'Title': ['Default Product'],
            'Variant Price': [0.0], 'URL': ['https://example.com']
        })
    
    if 'collection_title' not in df_coll.columns:
        print("WARNING: collection_title not found after merge, creating default")
        df_coll['collection_title'] = 'Default Collection'
    
    handle_col = 'Handle'
    if handle_col not in df_coll.columns:
        print(f"WARNING: {handle_col} not found after merge")
        if 'Handle_y' in df_coll.columns:
            print(f"Using Handle_y instead")
            df_coll[handle_col] = df_coll['Handle_y']
        elif 'Handle_x' in df_coll.columns:
            print(f"Using Handle_x instead")
            df_coll[handle_col] = df_coll['Handle_x']
        elif 'Product: Handle' in df_coll.columns:
            print(f"Using Product: Handle instead")
            df_coll[handle_col] = df_coll['Product: Handle']
        else:
            print(f"Creating default {handle_col}")
            df_coll[handle_col] = 'default-product'
    
    coll = df_coll.groupby(['collection_title', handle_col], dropna=False).size().reset_index(name='total_in_collection')
    try:
        search2 = coll.sort_values(['collection_title', 'total_in_collection'], ascending=[True, False]).groupby('collection_title', dropna=False).head(TOP_K).reset_index(drop=True)
        handle_col = 'Handle'
        if handle_col not in search2.columns:
            print(f"WARNING: {handle_col} not found in coll dataframe")
            if 'Handle_y' in coll.columns:
                search2[handle_col] = coll['Handle_y']
            elif 'Handle_x' in coll.columns:
                search2[handle_col] = coll['Handle_x']
            else:
                search2[handle_col] = 'default-product'
        
        if handle_col not in products.columns and len(products.columns) > 0:
            handle_cols = [c for c in products.columns if 'handle' in c.lower()]
            if handle_cols:
                products[handle_col] = products[handle_cols[0]]
        
        search2 = search2.merge(products, on=handle_col, how='left', suffixes=('', '_y'))
        if len(search2) == 0:
            print("Warning: Empty search2 after merge. Creating sample data.")
            search2 = pd.DataFrame({
                'Handle': ['default-product'], 'Title': ['Default Product'],
                'collection_title': ['Default Collection'], 'total_in_collection': [1],
                'Variant Price': [0.0], 'URL': ['https://example.com']
            })
    except Exception as e:
        print(f"Error creating search2: {e}")
        search2 = pd.DataFrame({
            'Handle': ['default-product'], 'Title': ['Default Product'],
            'collection_title': ['Default Collection'], 'total_in_collection': [1],
            'Variant Price': [0.0], 'URL': ['https://example.com']
        })

    required_cols = ['Handle', 'Variant ID', 'Title', 'Variant Price', 'URL']
    for col in required_cols:
        if col not in search1.columns:
            print(f"Adding missing column {col} to search1")
            if col == 'Handle':
                search1[col] = 'default-product'
            elif col == 'Variant ID':
                search1[col] = 'default-variant'
            elif col == 'Title':
                search1[col] = 'Default Product'
            elif col == 'Variant Price':
                search1[col] = 0.0
            elif col == 'URL':
                search1[col] = 'https://example.com'
    
    for col in required_cols:
        if col not in search2.columns:
            print(f"Adding missing column {col} to search2")
            if col == 'Handle':
                search2[col] = 'default-product'
            elif col == 'Variant ID':
                search2[col] = 'default-variant'
            elif col == 'Title':
                search2[col] = 'Default Product'
            elif col == 'Variant Price':
                search2[col] = 0.0
            elif col == 'URL':
                search2[col] = 'https://example.com'
    
    return search1, search2, category_col, coll_col, smart

# --- Pydantic Models ---
class Product(BaseModel):
    Handle: str
    Variant_ID: str = Field(..., alias='Variant ID')
    Title: str
    total_sold: int = Field(None)
    total_in_collection: int = Field(None)
    Variant_Price: float = Field(..., alias='Variant Price')
    URL: str
    
    class Config:
        populate_by_name = True

class ChatRequest(BaseModel):
    session_id: str
    message: str

# --- Helper Function to Call Lyzr API ---
def call_lyzr_api(payload, headers):
    try:
        response = requests.post(LYZR_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        print(f"Lyzr API response: {response_json}")
        
        agent_response = response_json.get("response")
        if not agent_response:
            print("No response field in Lyzr API output")
            return None
        
        if isinstance(agent_response, str):
            cleaned_response = re.sub(r'^```json\n|```$', '', agent_response, flags=re.MULTILINE).strip()
            try:
                parsed_response = json.loads(cleaned_response)
                return parsed_response
            except json.JSONDecodeError:
                print(f"Response is not JSON, treating as plain text: {cleaned_response}")
                return cleaned_response
        
        return agent_response
    except requests.RequestException as e:
        print(f"Error communicating with Lyzr API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Lyzr agent: {str(e)}")
    except ValueError as e:
        print(f"Failed to parse Lyzr API response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse Lyzr API response: {str(e)}")

# --- Helper Function to Collect Product Metadata ---
def get_product_metadata(category=None, limit_categories=None, product_type=None, max_products=50):
    if SEARCH1_DF is None or CATEGORY_COL is None or SMART_DF is None:
        return "No product data available."
    
    if category:
        subset = SEARCH1_DF[SEARCH1_DF[CATEGORY_COL].astype(str).str.lower() == category.lower()]
    elif limit_categories:
        subset = SEARCH1_DF[SEARCH1_DF[CATEGORY_COL].astype(str).str.lower().isin([c.lower() for c in limit_categories])]
    else:
        subset = SEARCH1_DF
    
    if product_type:
        subset = subset[
            subset['Title'].str.lower().str.contains(product_type.lower()) |
            subset['Tags'].str.lower().str.contains(product_type.lower(), na=False) |
            subset['Product Description'].str.lower().str.contains(product_type.lower(), na=False)
        ]
    
    if subset.empty:
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
            "Category: Name": row.get('Category: Name', ''),
            "Variant Price": float(row.get('Variant Price', 0.0))
        }
        if category and 'kids' not in category.lower() and ('kids' in str(product_info['Tags']).lower() or 'kids' in str(product_info['Title']).lower()):
            continue
        metadata.append(product_info)
    
    smart_subset = SMART_DF[SMART_DF['Product: Handle'].isin(subset['Handle'])]
    for _, row in smart_subset.iterrows():
        collection_info = {
            "Collection Title": row.get('Title', ''),
            "Rule: Product Column": row.get('Rule: Product Column', ''),
            "Rule: Condition": row.get('Rule: Condition', ''),
            "Product: Handle": row.get('Product: Handle', '')
        }
        for product in metadata:
            if product["Handle"] == collection_info["Product: Handle"]:
                product.update(collection_info)
    
    metadata_str = ""
    for i, product in enumerate(metadata[:max_products], 1):
        metadata_str += f"Product {i}:\n"
        for key, value in product.items():
            if value and isinstance(value, (str, float)):
                metadata_str += f"  {key}: {value}\n"
        metadata_str += "\n"
    
    return metadata_str if metadata_str else "No metadata available."

# --- Helper Function to Detect Intent ---
def detect_intent(message):
    message_lower = message.lower().strip()
    acknowledgment_keywords = []
    price_keywords = ['expensive', 'cost', 'price', 'cheaper', 'affordable', 'budget']
    craftsmanship_keywords = ['craftsmanship', 'material', 'handcrafted', 'quality']
    debug_keywords = ['print all items', 'show all items', 'debug']
    
    if any(keyword in message_lower for keyword in acknowledgment_keywords):
        return "acknowledgment"
    elif any(keyword in message_lower for keyword in price_keywords):
        return "price_concern"
    elif any(keyword in message_lower for keyword in craftsmanship_keywords):
        return "craftsmanship_explanation"
    elif any(keyword in message_lower for keyword in debug_keywords):
        return "debug_request"
    else:
        # Check for specific product details request
        if SESSION_STATE.get(message_lower, {}).get("recommended_products"):
            for product in SESSION_STATE[message_lower]["recommended_products"]:
                if product["Title"].lower() in message_lower:
                    return "specific_item_details"
        return "product_request"

def is_query_vague(message: str, categories: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if the query-Is it vague and doesn't contain specific product types or categories.
    Returns (is_vague, matched_product_type).
    """
    message_lower = message.lower().strip()
    
    # Define specific product types to check for
    product_types = ["nameplate", "mug", "wallet", "decor", "case", "gift"]
    
    # Check for product types
    for product_type in product_types:
        if product_type in message_lower:
            return False, product_type
    
    # Fuzzy match against categories
    best_match, score = process.extractOne(message_lower, categories, scorer=fuzz.partial_ratio)
    if score > 80:  # Threshold for considering a category match
        return False, best_match
    
    # Check for generic gift-related keywords
    gift_keywords = ["gift", "present", "birthday", "anniversary"]
    if any(keyword in message_lower for keyword in gift_keywords) and len(message_lower.split()) < 10:
        return True, None
    
    return False, None


def get_gift_appropriate_categories(categories: List[str], metadata: List[Dict]) -> List[str]:
    """
    Prioritize categories suitable for gifts, especially for a father's birthday.
    Returns a list of up to 10 gift-appropriate categories.
    """
    # Categories explicitly related to gifts or likely suitable for a father's birthday
    gift_priority = [
  "Earrings",
"Art & Craft Kits",
"Decor",
"Jewelry Holders",
"Candles",
"Games & Card Games",
"Handbags",
"Serving Trays",
"Necklaces",
"Educational Toys"]
    
    # Filter categories that exist in the provided list
    valid_gift_categories = [cat for cat in gift_priority if cat in categories]
    
    # Supplement with popular categories based on metadata (e.g., high total_sold)
    category_sales = {}
    for product in metadata:
        category = product.get("Category: Name", "Uncategorized")
        if category in categories and category not in valid_gift_categories:
            total_sold = product.get("total_sold", 0)
            category_sales[category] = category_sales.get(category, 0) + total_sold
    
    # Sort categories by total sales and pick top ones
    sorted_categories = sorted(category_sales, key=category_sales.get, reverse=True)
    additional_categories = sorted_categories[:10 - len(valid_gift_categories)]
    
    # Combine and ensure diversity
    result = valid_gift_categories + additional_categories
    if len(result) < 5:
        # If too few, randomly sample from remaining categories, excluding niche ones
        niche_categories = ["Pacifiers & Teethers", "Baby Toys & Activity Equipment", "Dolls"]
        available = [cat for cat in categories if cat not in result and cat not in niche_categories]
        result.extend(random.sample(available, min(10 - len(result), len(available))))
    
    return result  # Cap at 10 categories

# Update get_product_metadata to support broader category sampling
def get_product_metadata(category: Optional[str] = None, limit_categories: Optional[List[str]] = None, product_type: Optional[str] = None, max_products: int = 50) -> str:
    if SEARCH1_DF is None or CATEGORY_COL is None or SMART_DF is None:
        return "No product data available."
    
    if category:
        subset = SEARCH1_DF[SEARCH1_DF[CATEGORY_COL].astype(str).str.lower() == category.lower()]
    elif limit_categories:
        subset = SEARCH1_DF[SEARCH1_DF[CATEGORY_COL].astype(str).str.lower().isin([c.lower() for c in limit_categories])]
    else:
        # Sample from a broader set of categories for vague queries
        all_categories = SEARCH1_DF[CATEGORY_COL].dropna().unique().tolist()
        sampled_categories = random.sample(all_categories, min(20, len(all_categories)))  # Sample up to 20 categories
        subset = SEARCH1_DF[SEARCH1_DF[CATEGORY_COL].astype(str).str.lower().isin([c.lower() for c in sampled_categories])]
    
    if product_type:
        subset = subset[
            subset['Title'].str.lower().str.contains(product_type.lower()) |
            subset['Tags'].str.lower().str.contains(product_type.lower(), na=False) |
            subset['Product Description'].str.lower().str.contains(product_type.lower(), na=False)
        ]
    
    if subset.empty:
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
            "Category: Name": row.get(CATEGORY_COL, 'Uncategorized'),
            "Variant Price": float(row.get('Variant Price', 0.0)),
            "total_sold": int(row.get('total_sold', 0))
        }
        if category and 'kids' not in category.lower() and ('kids' in str(product_info['Tags']).lower() or 'kids' in str(product_info['Title']).lower()):
            continue
        for _, row in subset.iterrows():
            metadata.append(product_info)
    
    smart_subset = SMART_DF[SMART_DF['Product: Handle'].isin(subset['Handle'])]
    for _, row in smart_subset.iterrows():
        collection_info = {
            "Collection Title": row.get('Title', ''),
            "Rule: Product Column": row.get('Rule: Product Column', ''),
            "Rule: Condition": row.get('Rule: Condition', ''),
            "Product: Handle": row.get('Product: Handle', '')
        }
        for product in metadata:
            if product["Handle"] == collection_info["Product: Handle"]:
                product.update(collection_info)
    
    metadata_str = ""
    for i, product in enumerate(metadata[:max_products], 1):
        metadata_str += f"Product {i}:\n"
        for key, value in product.items():
            if value and isinstance(value, (str, float, int)):
                metadata_str += f"  {key}: {value}\n"
        metadata_str += "\n"
    
    return metadata_str if metadata_str else "No metadata available."
# --- API Endpoints ---
@app.get("/categories", response_model=List[str])
def get_categories():
    if SEARCH1_DF is None or CATEGORY_COL is None:
        raise HTTPException(status_code=500, detail="Category data not available.")
    categories = SEARCH1_DF[CATEGORY_COL].dropna().unique().tolist()
    return JSONResponse(content=sorted(categories))

@app.get("/search/{source}/{category}", response_model=List[Product])
def search(source: Literal['orders', 'collections'], category: str):
    df = SEARCH1_DF if source == 'orders' else SEARCH2_DF
    key = CATEGORY_COL if source == 'orders' else COLLECTION_COL
    
    if key not in df.columns:
        print(f"WARNING: Key column {key} not found in dataframe")
        if len(df.columns) > 0:
            key = df.columns[0]
            print(f"Using {key} as substitute")
        else:
            raise HTTPException(500, detail="Invalid dataframe structure")
    
    try:
        subset = df[df[key].astype(str).str.lower() == category.lower()]
        if subset.empty:
            print(f"No products found for {source} - {category}, returning default")
            raise HTTPException(404, detail=f"No products found for {source} – {category}")
    except Exception as e:
        print(f"Error filtering products: {str(e)}")
        raise HTTPException(500, detail=f"Error processing request: {str(e)}")
        
    try:
        records = subset.to_dict(orient='records')
        required_fields = ['Handle', 'Variant ID', 'Title', 'Variant Price', 'URL']
        for i, record in enumerate(records):
            for field in required_fields:
                if field not in record or record[field] is None:
                    print(f"Missing or null {field} in record {i}, setting default")
                    if field == 'Handle':
                        record[field] = f"default-product-{i}"
                    elif field == 'Variant ID':
                        record[field] = f"default-variant-{i}"
                    elif field == 'Title':
                        record[field] = f"Default Product {i}"
                    elif field == 'Variant Price':
                        record[field] = 0.0
                    elif field == 'URL':
                        record[field] = 'https://example.com'
        return records
    except Exception as e:
        print(f"Error converting to records: {str(e)}")
        raise HTTPException(500, detail=f"Error formatting response: {str(e)}")

# Replace the /chat endpoint with this updated version
@app.post("/chat")
async def chat(request: ChatRequest):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": LYZR_API_KEY
    }
    
    session = SESSION_STATE[request.session_id]
    intent = detect_intent(request.message)
    session["intent"] = intent
    if intent == "price_concern":
        session["price_sensitive"] = True
    
    if SEARCH1_DF is None or CATEGORY_COL is None:
        raise HTTPException(status_code=500, detail="Category data not available.")
    categories = SEARCH1_DF[CATEGORY_COL].dropna().unique().tolist()
    categories_str = ", ".join(categories)

    # Handle acknowledgment
    if intent == "acknowledgment":
        session["user_responses"].append(request.message)
        session["questions_asked"].append("Acknowledgment received")
        return {"response": "You're welcome! Let me know if you need more help or want to explore other options."}

    # Handle specific item details
    if intent == "specific_item_details":
        product_title = None
        for product in session.get("recommended_products", []):
            if product["Title"].lower() in request.message.lower():
                product_title = product["Title"]
                break
        
        if product_title:
            subset = SEARCH1_DF[SEARCH1_DF['Title'].str.lower() == product_title.lower()]
            if not subset.empty:
                record = subset.iloc[0]
                details = (
                    f"Here are the full details for {record['Title']}:\n\n"
                    f"Variant ID: {record.get('Variant ID', 'unknown-variant')}\n"
                    f"Price: ${record.get('Variant Price', 0.0):.2f}\n"
                    f"URL: {record.get('URL', 'https://example.com')}\n"
                    f"Description: {record.get('Product Description', 'No description available.')}\n"
                    f"Category: {record.get(CATEGORY_COL, 'Unknown')}\n"
                    f"Tags: {record.get('Tags', 'None')}\n"
                    f"Type: {record.get('Type', 'None')}\n"
                )
                session["user_responses"].append(request.message)
                session["questions_asked"].append("Specific item details provided")
                return {"response": details}
        
        session["user_responses"].append(request.message)
        session["questions_asked excise_common_categories"].append("Specific item details not found")
        return {"response": "Sorry, I couldn't find details for that item. Could you clarify the product name or describe it further?"}

    # Handle debug request
    if intent == "debug_request":
        product_info = session.get("last_product_info", "No product data available from the last recommendation.")
        session["user_responses"].append(request.message)
        session["questions_asked"].append("Debug request")
        return {"response": f"Here are the products sent to the LLM for the last recommendation:\n\n{product_info}"}

    # Handle price concern
    if intent == "price_concern" and session["stage"] != "category_detection":
        metadata = get_product_metadata(category=session["category"], product_type=session.get("product_type"), max_products=50) if session["category"] else get_product_metadata(limit_categories=session["suggested_categories"] or categories[:5], max_products=50)
        conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(session["questions_asked"], session["user_responses"][1:]))
        
        product_info = ""
        try:
            df = SEARCH1_DF
            key = CATEGORY_COL
            if session["category"]:
                subset = df[df[key].astype(str).str.lower() == session["category"].lower()]
            else:
                subset = df[df[key].astype(str).str.lower().isin([c.lower() for c in session["suggested_categories"]])]
            
            if session.get("product_type"):
                subset = subset[
                    subset['Title'].str.lower().str.contains(session["product_type"].lower()) |
                    subset['Tags'].str.lower().str.contains(session["product_type"].lower(), na=False) |
                    subset['Product Description'].str.lower().str.contains(session["product_type"].lower(), na=False)
                ]
            
            if not subset.empty:
                subset = subset.sort_values('Variant Price', ascending=True).drop_duplicates('Variant ID')
                unique_subset = []
                seen_titles = set()
                for _, row in subset.iterrows():
                    title = row['Title'].lower()
                    if not any(title.startswith(seen) or seen.startswith(title) for seen in seen_titles):
                        unique_subset.append(row)
                        seen_titles.add(title)
                subset = pd.DataFrame(unique_subset)
                records = subset.to_dict(orient='records')
                product_info = f"Available products (sorted by price, up to 50):\n"
                for i, record in enumerate(records[:50]):
                    title = record.get('Title', 'Unknown Product')
                    price = record.get('Variant Price', 0.0)
                    url = record.get('URL', 'https://example.com')
                    variant_id = record.get('Variant ID', 'unknown-variant')
                    description = record.get('Product Description', '')[:200]
                    product_info += f"{i+1}. {title} (Variant ID: {variant_id})\n   Price: ${price:.2f}\n   URL: {url}\n   Description: {description}\n"
            else:
                product_info = f"No products found for the specified criteria."
        except Exception as e:
            product_info = f"Error fetching products: {str(e)}"
        
        session["last_product_info"] = product_info
        prompt = (
            f"User query: {session['user_responses'][0]}\n\n"
            f"Detected category: {session['category'] or 'None'}\n\n"
            f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
            f"Conversation history:\n{conversation_history}\n\n"
            f"Current user response: {request.message}\n\n"
            f"Product metadata:\n{metadata}\n\n"
            f"Product data:\n{product_info}\n\n"
            f"You are a shopping assistant for Zwende. The user has expressed concern about high prices. Respond empathetically and provide a human-like response. "
            f"Briefly explain the value of handcrafted quality and materials, then recommend up to {TOP_K} affordable products "
            f"(lowest prices within the category or suggested categories) that align with their preferences (e.g., {session.get('product_type', 'personalized items')}). "
            f"Ensure recommendations are unique (by Variant ID), diverse (different titles or types), and exclude kids' products unless requested. "
            f"List each product with its title, price, URL, and a brief description. "
            f"If no affordable products are available, suggest alternative categories or ask for clarification. "
            f"Be engaging, polite, and conversational, using phrases like 'I hear you' or 'Let's find something perfect for your budget.'"
        )
        payload = {
            "user_id": "pranav@lyzr.ai",
            "agent_id": "681b2915bb74a5da4a2eda8e",
            "session_id": request.session_id,
            "message": prompt
        }
        
        agent_response = call_lyzr_api(payload, headers)
        if not agent_response or isinstance(agent_response, dict):
            print(f"Invalid or missing Lyzr response: {agent_response}")
            agent_response = "I hear you about the price concerns! Unfortunately, I couldn't find suitable products right now. Could you share more details about your budget or preferences?"
        
        session["user_responses"].append(request.message)
        session["questions_asked"].append("Price concern addressed")
        return {"response": agent_response}

    # Handle craftsmanship explanation
    if intent == "craftsmanship_explanation":
        metadata = get_product_metadata(category=session["category"], product_type=session.get("product_type"), max_products=50) if session["category"] else get_product_metadata(limit_categories=session["suggested_categories"] or categories[:5], max_products=50)
        conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(session["questions_asked"], session["user_responses"][1:]))
        
        prompt = (
            f"User query: {session['user_responses'][0]}\n\n"
            f"Detected category: {session['category'] or 'None'}\n\n"
            f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
            f"Conversation history:\n{conversation_history}\n\n"
            f"Current user response: {request.message}\n\n"
            f"Product metadata:\n{metadata}\n\n"
            f"You are a shopping assistant for Zwende. The user is interested in the craftsmanship or materials of the products. "
            f"Provide a human-like, conversational response explaining the handcrafted nature, quality materials (e.g., terracotta, mango wood), and artisanal techniques used in Zwende products. "
            f"Do not recommend products unless explicitly requested. Be concise, engaging, and empathetic, using phrases like 'Our artisans pour their heart into every piece.' "
            f"If appropriate, ask a follow-up question to clarify their preferences (e.g., 'Are you looking for a specific material like wood or ceramic?')."
        )
        payload = {
            "user_id": "pranav@lyzr.ai",
            "agent_id": "681b2915bb74a5da4a2eda8e",
            "session_id": request.session_id,
            "message": prompt
        }
        
        agent_response = call_lyzr_api(payload, headers)
        if not agent_response or isinstance(agent_response, dict):
            print(f"Invalid or missing Lyzr response: {agent_response}")
            agent_response = "Our products are handcrafted with love by skilled artisans using high-quality materials like terracotta and wood. Could you share more about what you're looking for?"
        
        session["user_responses"].append(request.message)
        session["questions_asked"].append("Craftsmanship explanation provided")
        return {"response": agent_response}

    # Standard conversation flow
    if session["stage"] == "category_detection":
        # Check if the query is vague
        is_vague, matched_item = is_query_vague(request.message, categories)
        
        if is_vague:
            # Return a neutral clarification question without suggesting categories
            session["category"] = None
            session["suggested_categories"] = []
            session["stage"] = "follow_up"
            session["questions_asked"].append("To help me find the perfect gift for your father, could you share some of his.Concurrent interest or hobbies?")
            session["user_responses"].append(request.message)
            session["product_metadata"] = []
            return {"response": "To help me find the perfect gift for your father, could you share some of his interests or hobbies?"}
        
        # If a product type or category is detected, proceed with Lyzr API
        if matched_item:
            if matched_item in categories:
                session["category"] = matched_item
                session["suggested_categories"] = []
            else:
                session["product_type"] = matched_item
        
        # Get metadata for broader category sampling
        if session.get("category"):
            metadata = get_product_metadata(category=session["category"], product_type=session.get("product_type"), max_products=50)
        else:
            # Use gift-appropriate categories for vague queries
            metadata_list = []
            for _, row in SEARCH1_DF.iterrows():
                metadata_list.append({
                    "Category: Name": row.get(CATEGORY_COL, 'Uncategorized'),
                    "total_sold": int(row.get('total_sold', 0))
                })
            gift_categories = get_gift_appropriate_categories(categories, metadata_list)
            metadata = get_product_metadata(limit_categories=gift_categories, product_type=session.get("product_type"), max_products=50)
        
        prompt = (
            f"User query: {request.message}\n\n"
            f"Available categories: {categories_str}\n\n"
            f"You are a shopping assistant for Zwende, an online store specializing in handcrafted products. "
            f"Based on the user's query, identify the most relevant category from the list provided. "
            f"If the query specifies a product type (e.g., 'nameplate' or '{session.get('product_type', '')}'), prioritize products matching that type in the metadata. "
            f"If the query is vague but contains some context (e.g., 'gift for father'), suggest 5 to 10 relevant categories based on the product metadata, depending on for whom user wants to buy"
            f"Ensure suggested categories are diverse and exclude categories like 'Pacifiers & Teethers' or 'Baby Toys & Activity Equipment' unless explicitly relevant. "
            f"Generate one follow-up question to narrow down the user's preferences, using the metadata to inform the question. "
            f"Exclude kids' products unless explicitly mentioned in the query. "
            f"Return a JSON object (not a string or code block) with 'category' (the category name or 'None' if no clear category), "
            f"'suggested_categories' (list of 5 to 10 categories if no clear category), 'question' (the questions should contain all those category for user to pick), "
            f"and 'attributes' (list of key attributes identified from metadata). "
            f"Example: "
            f"{{\n"
            f"  \"category\": \"Decor\",\n"
            f"  \"suggested_categories\": [],\n"
            f"  \"question\": \"What style of nameplate are you looking for?\",\n"
            f"  \"attributes\": [\"type: nameplate\", \"feature: personalized\"]\n"
            f"}}\n\n"
            f"Product metadata:\n"
            f"{metadata}\n\n"
            f"If no relevant categories are found, return: "
            f"{{\n"
            f"  \"category\": \"None\",\n"
            f"  \"suggested_categories\": [],\n"
            f"  \"question\": \"Could you provide more details about what you're looking for?\",\n"
            f"  \"attributes\": []\n"
            f"}}"
        )
        
        payload = {
            "user_id": "pranav@lyzr.ai",
            "agent_id": "681b2915bb74a5da4a2eda8e",
            "session_id": request.session_id,
            "message": prompt
        }
        
        agent_response = call_lyzr_api(payload, headers)
        if not agent_response or not isinstance(agent_response, dict):
            print(f"Invalid or missing Lyzr response: {agent_response}")
            session["category"] = None
            session["suggested_categories"] = []
            session["stage"] = "follow_up"
            session["questions_asked"].append("Could you provide more details about what you're looking for?")
            session["user_responses"].append(request.message)
            session["product_metadata"] = []
            return {"response": "Could you provide more details about what you're looking for?"}
        
        detected_category = agent_response.get("category", "None").strip()
        suggested_categories = agent_response.get("suggested_categories", [])
        follow_up_question = agent_response.get("question", "Could you provide more details about what you're looking for?")
        attributes = agent_response.get("attributes", [])
        
        session["category"] = detected_category if detected_category != "None" else None
        session["suggested_categories"] = suggested_categories
        session["stage"] = "follow_up"
        session["questions_asked"].append(follow_up_question)
        session["user_responses"].append(request.message)
        session["product_metadata"] = attributes
        return {"response": follow_up_question}

    elif session["stage"] == "follow_up":
        session["user_responses"].append(request.message)
        
        # Update product type if mentioned
        if "nameplate" in request.message.lower() and not session.get("product_type"):
            session["product_type"] = "nameplate"
        
        if len(session["questions_asked"]) < 2:
            metadata = get_product_metadata(category=session["category"], product_type=session.get("product_type"), max_products=50) if session["category"] else get_product_metadata(limit_categories=session["suggested_categories"] or categories[:5], product_type=session.get("product_type"), max_products=50)
            conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(session["questions_asked"], session["user_responses"][1:]))
            prompt = (
                f"User query: {session['user_responses'][0]}\n\n"
                f"Detected category: {session['category'] or 'None'}\n\n"
                f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
                f"Product metadata:\n{metadata}\n\n"
                f"Conversation history:\n{conversation_history}\n\n"
                f"Current user response: {request.message}\n\n"
                f"Previous attributes identified: {', '.join(session['product_metadata'])}\n\n"
                f"You are a shopping assistant for Zwende. Based on the conversation history, product metadata, and the user's latest response, "
                f"generate one more follow-up question to further narrow down their preferences. based on available product metadata"
                f"If the user specifies a product type (e.g., 'nameplate'), prioritize that type in the question and metadata. "
                f"If no category was detected, try to identify a category based on the user's response and metadata. "
                f"Exclude kids' products unless explicitly mentioned. "
                f"Return a JSON object (not a string or code block) with 'category' (updated category or 'None'), "
                f"'question' (the follow-up question), and 'attributes' (updated list of attributes). "
                f"Example: "
                f"{{\n"
                f"  \"category\": \"Decor\",\n"
                f"  \"question\": \"Would you prefer a wooden or metal nameplate?\",\n"
                f"  \"attributes\": [\"type: nameplate\", \"feature: personalized\"]\n"
                f"}}"
            )
            payload = {
                "user_id": "pranav@lyzr.ai",
                "agent_id": "681b2915bb74a5da4a2eda8e",
                "session_id": request.session_id,
                "message": prompt
            }
            
            agent_response = call_lyzr_api(payload, headers)
            if not agent_response or not isinstance(agent_response, dict):
                print(f"Invalid or missing Lyzr response for follow-up: {agent_response}")
                session["questions_asked"].append("Could you provide more details about what you're looking for?")
                session["product_metadata"] = session["product_metadata"] or []
                return {"response": "Could you provide more details about what you're looking for?"}
            
            follow_up_question = agent_response.get("question", "Could you provide more details about what you're looking for?")
            attributes = agent_response.get("attributes", session["product_metadata"])
            updated_category = agent_response.get("category", session["category"] or "None")
            
            if updated_category != "None" and updated_category in categories:
                session["category"] = updated_category
                session["suggested_categories"] = []
            session["questions_asked"].append(follow_up_question)
            session["product_metadata"] = attributes
            return {"response": follow_up_question}
        
        else:
            session["stage"] = "recommendation"
            product_type = session.get("product_type")
            metadata = get_product_metadata(category=session["category"], product_type=product_type, max_products=50) if session["category"] else get_product_metadata(limit_categories=session["suggested_categories"] or categories[:5], product_type=product_type, max_products=50)
            conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(session["questions_asked"], session["user_responses"][1:]))
            
            product_info = ""
            try:
                df = SEARCH1_DF
                key = CATEGORY_COL
                if session["category"]:
                    subset = df[df[key].astype(str).str.lower() == session["category"].lower()]
                else:
                    subset = df[df[key].astype(str).str.lower().isin([c.lower() for c in session["suggested_categories"]])]
                
                if product_type:
                    subset = subset[
                        subset['Title'].str.lower().str.contains(product_type.lower()) |
                        subset['Tags'].str.lower().str.contains(product_type.lower(), na=False) |
                        subset['Product Description'].str.lower().str.contains(product_type.lower(), na=False)
                    ]
                
                if not subset.empty:
                    sort_key = 'Variant Price' if session["price_sensitive"] else 'total_sold'
                    sort_ascending = True if session["price_sensitive"] else False
                    subset = subset.sort_values(sort_key, ascending=sort_ascending).drop_duplicates('Variant ID')
                    unique_subset = []
                    seen_titles = set()
                    for _, row in subset.iterrows():
                        title = row['Title'].lower()
                        if not any(title.startswith(seen) or seen.startswith(title) for seen in seen_titles):
                            unique_subset.append(row)
                            seen_titles.add(title)
                    subset = pd.DataFrame(unique_subset)
                    records = subset.to_dict(orient='records')
                    product_info = f"Available products ({'sorted by price' if session['price_sensitive'] else 'top-selling'}, up to 50):\n"
                    for i, record in enumerate(records[:50]):
                        title = record.get('Title', 'Unknown Product')
                        price = record.get('Variant Price', 0.0)
                        url = record.get('URL', 'https://example.com')
                        variant_id = record.get('Variant ID', 'unknown-variant')
                        description = record.get('Product Description', '')[:200]
                        product_info += f"{i+1}. {title} (Variant ID: {variant_id})\n   Price: ${price:.2f}\n   URL: {url}\n   Description: {description}\n"
                    session["recommended_products"] = [
                        {
                            "Title": record.get('Title', 'Unknown Product'),
                            "Variant ID": record.get('Variant ID', 'unknown-variant'),
                            "Price": float(record.get('Variant Price', 0.0)),
                            "URL": record.get('URL', 'https://example.com'),
                            "Description": record.get('Product Description', '')
                        } for record in records[:TOP_K]
                    ]
                else:
                    product_info = f"No products found for the specified criteria."
            except Exception as e:
                product_info = f"Error fetching products: {str(e)}"
            
            session["last_product_info"] = product_info
            prompt = (
                f"User query: {session['user_responses'][0]}\n\n"
                f"Detected category: {session['category'] or 'None'}\n\n"
                f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
                f"Conversation history:\n{conversation_history}\n\n"
                f"Current user response: {request.message}\n\n"
                f"Product metadata:\n{metadata}\n\n"
                f"Product data:\n{product_info}\n\n"
                f"You are a shopping assistant for Zwende. Based on the conversation history, product metadata, and product data, "
                f"provide a friendly, human-like response recommending up to {TOP_K} unique products that match the user's preferences "
                f"(e.g., {product_type or 'personalized items'} for a specific purpose). "
                f"{'The user is concerned about high prices, so prioritize affordable options (lowest prices within the category) and explain their value.' if session['price_sensitive'] else 'Select the best-selling products unless otherwise specified.'} "
                f"Ensure recommendations are unique (by Variant ID), diverse (different titles or types), and exclude kids' products unless requested. "
                f"If the user specifies a product type (e.g., 'nameplate'), only recommend products matching that type (e.g., containing 'nameplate' in Title, Tags, or Description). "
                f"List each product with its title, price, URL, and a brief description, ensuring alignment with the user's preferences. "
                f"If no products match exactly, select the closest matches and explain why. If no products are available, suggest alternative categories or ask for clarification. "
                f"Be engaging, polite, and conversational, using empathetic phrases like 'I hear you' or 'Let's find something perfect.'"
            )
            payload = {
                "user_id": "pranav@lyzr.ai",
                "agent_id": "681b2915bb74a5da4a2eda8e",
                "session_id": request.session_id,
                "message": prompt
            }
            
            agent_response = call_lyzr_api(payload, headers)
            if not agent_response or isinstance(agent_response, dict):
                print(f"Invalid or missing Lyzr response for recommendation: {agent_response}")
                agent_response = "I'm sorry, I couldn't find suitable products. Could you provide more details about what you're looking for?"
            
            session["stage"] = "post_recommendation"
            session["user_responses"].append(request.message)
            session["questions_asked"].append("Recommendation provided")
            return {"response": agent_response}

    elif session["stage"] == "post_recommendation":
        product_type = session.get("product_type")
        metadata = get_product_metadata(category=session["category"], product_type=product_type, max_products=50) if session["category"] else get_product_metadata(limit_categories=session["suggested_categories"] or categories[:5], product_type=product_type, max_products=50)
        conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(session["questions_asked"], session["user_responses"][1:]))
        
        product_info = ""
        try:
            df = SEARCH1_DF
            key = CATEGORY_COL
            if session["category"]:
                subset = df[df[key].astype(str).str.lower() == session["category"].lower()]
            else:
                subset = df[df[key].astype(str).str.lower().isin([c.lower() for c in session["suggested_categories"]])]
            
            if product_type:
                subset = subset[
                    subset['Title'].str.lower().str.contains(product_type.lower()) |
                    subset['Tags'].str.lower().str.contains(session["product_type"].lower(), na=False) |
                    subset['Product Description'].str.lower().str.contains(product_type.lower(), na=False)
                ]
            
            if not subset.empty:
                sort_key = 'Variant Price' if session["price_sensitive"] else 'total_sold'
                sort_ascending = True if session["price_sensitive"] else False
                subset = subset.sort_values(sort_key, ascending=sort_ascending).drop_duplicates('Variant ID')
                unique_subset = []
                seen_titles = set()
                for _, row in subset.iterrows():
                    title = row['Title'].lower()
                    if not any(title.startswith(seen) or seen.startswith(title) for seen in seen_titles):
                        unique_subset.append(row)
                        seen_titles.add(title)
                subset = pd.DataFrame(unique_subset)
                records = subset.to_dict(orient='records')
                product_info = f"Available products ({'sorted by price' if session['price_sensitive'] else 'top-selling'}, up to 50):\n"
                for i, record in enumerate(records[:50]):
                    title = record.get('Title', 'Unknown Product')
                    price = record.get('Variant Price', 0.0)
                    url = record.get('URL', 'https://example.com')
                    variant_id = record.get('Variant ID', 'unknown-variant')
                    description = record.get('Product Description', '')[:200]
                    product_info += f"{i+1}. {title} (Variant ID: {variant_id})\n   Price: ${price:.2f}\n   URL: {url}\n   Description: {description}\n"
                session["recommended_products"] = [
                    {
                        "Title": record.get('Title', 'Unknown Product'),
                        "Variant ID": record.get('Variant ID', 'unknown-variant'),
                        "Price": float(record.get('Variant ---\nPrice', 0.0)),
                        "URL": record.get('URL', 'https://example.com'),
                        "Description": record.get('Product Description', '')
                    } for record in records[:TOP_K]
                ]
            else:
                product_info = f"No products found for the specified criteria."
        except Exception as e:
            product_info = f"Error fetching products: {str(e)}"
        
        session["last_product_info"] = product_info
        prompt = (
            f"User query: {session['user_responses'][0]}\n\n"
            f"Detected category: {session['category'] or 'None'}\n\n"
            f"Suggested categories: {', '.join(session['suggested_categories']) or 'None'}\n\n"
            f"Conversation history:\n{conversation_history}\n\n"
            f"Current user response: {request.message}\n\n"
            f"Product metadata:\n{metadata}\n\n"
            f"Product data:\n{product_info}\n\n"
            f"You are a shopping assistant for Zwende. The user has received recommendations and provided a new response. "
            f"Based on the conversation history, product metadata, and product data, provide a friendly, human-like response. "
            f"If the user specifies a product type (e.g., 'nameplate'), only recommend products matching that type (e.g., containing 'nameplate' in Title, Tags, or Description). "
            f"{'The user is concerned about high prices, so prioritize affordable options (lowest prices within the category) and explain their value.' if session['price_sensitive'] else 'Select the best-selling products unless otherwise specified.'} "
            f"Ensure recommendations are unique (by Variant ID), diverse (different titles or types), and exclude kids' products unless requested. "
            f"List up to {TOP_K} products with their title, price, URL, and a brief description, ensuring alignment with the user's preferences. "
            f"If no products match exactly, select the closest matches and explain why. If no products are available, suggest alternative categories or ask for clarification. "
            f"Be engaging, polite, and conversational, using empathetic phrases like 'I hear you' or 'Let's find something perfect.'"
        )
        payload = {
            "user_id": "pranav@lyzr.ai",
            "agent_id": "681b2915bb74a5da4a2eda8e",
            "session_id": request.session_id,
            "message": prompt
        }
        
        agent_response = call_lyzr_api(payload, headers)
        if not agent_response or isinstance(agent_response, dict):
            print(f"Invalid or missing Lyzr response for recommendation: {agent_response}")
            agent_response = "I'm sorry, I couldn't find suitable products. Could you provide more details about what you're looking for?"
        
        session["user_responses"].append(request.message)
        session["questions_asked"].append("Post-recommendation response")
        return {"response": agent_response}

    else:
        # Reset session state
        session["category"] = None
        session["suggested_categories"] = []
        session["questions_asked"] = []
        session["user_responses"] = []
        session["stage"] = "category_detection"
        session["product_metadata"] = None
        session["price_sensitive"] = False
        session["intent"] = None
        session["recommended_products"] = []

        # Check if the query is vague
        is_vague, matched_item = is_query_vague(request.message, categories)
        
        if is_vague:
            session["stage"] = "follow_up"
            session["questions_asked"].append("To help me find the perfect gift for your father, could you share some of his interests or hobbies?")
            session["user_responses"].append(request.message)
            session["product_metadata"] = []
            return {"response": "To help me find the perfect gift for your father, could you share some of his interests or hobbies?"}
        
        # If a product type or category is detected
        if matched_item:
            if matched_item in categories:
                session["category"] = matched_item
                session["suggested_categories"] = []
            else:
                session["product_type"] = matched_item
        
        prompt = (
            f"User query: {request.message}\n\n"
            f"Available categories: {categories_str}\n\n"
            f"You are a shopping assistant for Zwende. Based on the user's query, identify the most relevant category. "
            f"If the query specifies a product type (e.g., 'nameplate' or '{session.get('product_type', '')}'), prioritize products matching that type in the metadata. "
            f"If the query contains some context, suggest up to 3 relevant categories based on the product metadata and the query. "
            f"Generate one follow-up question to narrow down the user's preferences. "
            f"Exclude kids' products unless explicitly mentioned in the query. "
            f"Return a JSON object (not a string or code block) with 'category' (category name or 'None'), "
            f"'suggested_categories' (list of up to 3 categories), 'question' (the follow-up question), "
            f"and 'attributes' (list of key attributes). "
            f"Example: "
            f"{{\n"
            f"  \"category\": \"Decor\",\n"
            f"  \"suggested_categories\": [],\n"
            f"  \"question\": \"What style of nameplate are you looking for?\",\n"
            f"  \"attributes\": [\"type: nameplate\", \"feature: personalized\"]\n"
            f"}}\n\n"
            f"Product metadata:\n"
            f"{{metadata}}\n\n"
            f"If no relevant categories are found, return: "
            f"{{\n"
            f"  \"category\": \"None\",\n"
            f"  \"suggested_categories\": [],\n"
            f"  \"question\": \"Could you provide more details about what you're looking for?\",\n"
            f"  \"attributes\": []\n"
            f"}}"
        )
        
        # Prepare metadata
        if session.get("category"):
            metadata = get_product_metadata(category=session["category"], product_type=session.get("product_type"), max_products=50)
        else:
            metadata = get_product_metadata(limit_categories=categories[:5], product_type=session.get("product_type"), max_products=50)
        
        payload = {
            "user_id": "pranav@lyzr.ai",
            "agent_id": "681b2915bb74a5da4a2eda8e",
            "session_id": request.session_id,
            "message": prompt.replace("{metadata}", metadata)
        }
        
        agent_response = call_lyzr_api(payload, headers)
        if not agent_response or not isinstance(agent_response, dict):
            print(f"Invalid or missing Lyzr response: {agent_response}")
            session["category"] = None
            session["suggested_categories"] = []
            session["stage"] = "follow_up"
            session["questions_asked"].append("Could you provide more details about what you're looking for?")
            session["user_responses"].append(request.message)
            session["product_metadata"] = []
            return {"response": "Could you provide more details about what you're looking for?"}
        
        detected_category = agent_response.get("category", "None").strip()
        suggested_categories = agent_response.get("suggested_categories", [])
        follow_up_question = agent_response.get("question", "Could you provide more details about what you're looking for?")
        attributes = agent_response.get("attributes", [])
        
        session["category"] = detected_category if detected_category != "None" else None
        session["suggested_categories"] = suggested_categories
        session["stage"] = "follow_up"
        session["questions_asked"].append(follow_up_question)
        session["user_responses"].append(request.message)
        session["product_metadata"] = attributes
        return {"response": follow_up_question}

@app.get("/health")
def health():
    return {'status': 'ok', 'orders': len(SEARCH1_DF), 'collections': len(SEARCH2_DF)}

@app.get("/")
def root():
    return {"message": "Zwende Search Agent API is running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8003"))
    print(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")