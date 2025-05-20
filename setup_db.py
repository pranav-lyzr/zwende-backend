import os
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import gdown
import tempfile

# Load environment variables
load_dotenv()

# Database connection
db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(db_url)

# Google Docs/Drive URLs
ORDER_URL = "https://docs.google.com/spreadsheets/d/1HlCUenUpTW3PeZ3DJKFUd536-S0Swn3t/export?format=xlsx"
PRODUCT_URL = "https://docs.google.com/spreadsheets/d/18JCz7sg1tLB7ZcI44EZ4sjdq5e4Qdh6D/export?format=xlsx"
SMART_URL = "https://drive.google.com/uc?id=1_dHsJNwBvsqKza2jWuy4Ml1nBTx54eIy"

def create_tables():
    create_tables_sql = """
    -- Enable extensions
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    -- Drop tables if they exist
    DROP TABLE IF EXISTS orders CASCADE;
    DROP TABLE IF EXISTS products CASCADE;
    DROP TABLE IF EXISTS smart_collections CASCADE;

    -- Orders table
    CREATE TABLE orders (
        "ID" VARCHAR(255),
        "Name" VARCHAR(255),
        "Created At" TIMESTAMP,
        "Price: Total Line Items" FLOAT,
        "Price: Subtotal" FLOAT,
        "Row #" INTEGER,
        "Top Row" VARCHAR(255),
        "Line: Type" VARCHAR(255),
        "Line: Product ID" VARCHAR(255),
        "Line: Product Handle" VARCHAR(255),
        "Line: Title" VARCHAR(255),
        "Line: Name" VARCHAR(255),
        "Line: Variant ID" VARCHAR(255),
        "Line: Variant Title" VARCHAR(255),
        "Line: Vendor" VARCHAR(255),
        "Line: Properties" TEXT
    );

    -- Products table
    CREATE TABLE products (
        "ID" VARCHAR(255),
        "Handle" VARCHAR(255),
        "Title" VARCHAR(255),
        "Product Description" TEXT,
        "Vendor" VARCHAR(255),
        "Type" VARCHAR(255),
        "Tags" TEXT,
        "Status" VARCHAR(255),
        "Published" BOOLEAN,
        "URL" TEXT,
        "Category: Name" VARCHAR(255),
        "Category" VARCHAR(255),
        "Image Src" TEXT,
        "Variant ID" VARCHAR(255),
        "Option1 Name" VARCHAR(255),
        "Option1 Value" VARCHAR(255),
        "Option2 Name" VARCHAR(255),
        "Option2 Value" VARCHAR(255),
        "Option3 Name" VARCHAR(255),
        "Option3 Value" VARCHAR(255),
        "Variant Image" TEXT,
        "Variant Weight" FLOAT,
        "Variant Weight Unit" VARCHAR(255),
        "Variant Price" FLOAT
    );

    -- Smart Collections table
    CREATE TABLE smart_collections (
        "ID" FLOAT,
        "Handle" VARCHAR(255),
        "Command" VARCHAR(255),
        "Title" VARCHAR(255),
        "Body HTML" TEXT,
        "Sort Order" VARCHAR(255),
        "Template Suffix" VARCHAR(255),
        "Updated At" VARCHAR(255),
        "Published" BOOLEAN,
        "Published At" VARCHAR(255),
        "Published Scope" VARCHAR(255),
        "Image Src" TEXT,
        "Image Width" FLOAT,
        "Image Height" FLOAT,
        "Image Alt Text" VARCHAR(255),
        "Row #" INTEGER,
        "Top Row" VARCHAR(255),
        "Must Match" VARCHAR(255),
        "Rule: Product Column" VARCHAR(255),
        "Rule: Relation" VARCHAR(255),
        "Rule: Condition" VARCHAR(255),
        "Product: ID" FLOAT,
        "Product: Handle" VARCHAR(255)
    );

    -- Add indexes for performance
    CREATE INDEX idx_orders_handle_variant ON orders ("Line: Product Handle", "Line: Variant ID");
    CREATE INDEX idx_products_handle_variant ON products ("Handle", "Variant ID");
    CREATE INDEX idx_products_category ON products ("Category: Name");
    CREATE INDEX idx_smart_collections_handle ON smart_collections ("Product: Handle");

    -- Additional indexes for /chat endpoint
    -- Full-text search index
    CREATE INDEX idx_products_fts ON products USING GIN (
        to_tsvector('english', "Title" || ' ' || "Tags" || ' ' || "Product Description")
    );

    -- Trigram indexes for LIKE queries
    CREATE INDEX idx_products_title_trgm ON products USING GIN ("Title" gin_trgm_ops);
    CREATE INDEX idx_products_tags_trgm ON products USING GIN ("Tags" gin_trgm_ops);

    -- Composite index for category and price sorting
    CREATE INDEX idx_products_category_price ON products ("Category: Name", "Variant Price");

    -- Partial index for non-null filters
    CREATE INDEX idx_products_non_null ON products ("Handle", "Variant ID")
    WHERE "Variant ID" IS NOT NULL AND "Variant Price" IS NOT NULL;

    -- Index for price sorting
    CREATE INDEX idx_products_price ON products ("Variant Price");
    """
    try:
        with engine.connect() as connection:
            connection.execute(text(create_tables_sql))
            connection.commit()
        print("Tables and indexes created successfully.")
    except Exception as e:
        print(f"Error creating tables and indexes: {str(e)}")
        raise

def download_file(url, output_path):
    if "docs.google.com" in url:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {output_path}")
        else:
            raise Exception(f"Failed to download {url}: Status code {response.status_code}")
    else:
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded {output_path}")

def import_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download files
        order_file = os.path.join(tmpdir, "OrderExport230425.xlsx")
        product_file = os.path.join(tmpdir, "ProductExport230425.xlsx")
        smart_file = os.path.join(tmpdir, "Smart Collection Export.csv")

        download_file(ORDER_URL, order_file)
        download_file(PRODUCT_URL, product_file)
        download_file(SMART_URL, smart_file)

        # Read files
        orders = pd.read_excel(order_file)
        products = pd.read_excel(product_file)
        smart = pd.read_csv(smart_file)

        # Handle data types
        orders['Created At'] = pd.to_datetime(orders['Created At'], errors='coerce')
        smart['Published'] = smart['Published'].astype(bool)
        smart['Updated At'] = smart['Updated At'].astype(str)
        smart['Published At'] = smart['Published At'].astype(str)

        # Import to database
        orders.to_sql('orders', engine, if_exists='append', index=False)
        products.to_sql('products', engine, if_exists='append', index=False)
        smart.to_sql('smart_collections', engine, if_exists='append', index=False)

        print(f"Imported {len(orders)} orders, {len(products)} products, and {len(smart)} smart collections.")

def main():
    print("Setting up database...")
    create_tables()
    import_data()
    print("Database setup complete.")

if __name__ == "__main__":
    main()