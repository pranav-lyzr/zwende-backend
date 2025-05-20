import os
import pandas as pd
import requests
from sqlalchemy import create_engine
import gdown
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded database connection details
DB_USER = "postgres"
DB_PASSWORD = "1VrOfw8<dIE_~mxyp1fBA*buIy~-"
DB_HOST = "zawande-test.cfm46gmg0sf8.eu-west-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "postgres"

# Database connection with SSL
db_url = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
)
try:
    engine = create_engine(db_url)
    with engine.connect() as connection:
        logger.info("Successfully connected to the database")
except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    raise

# Google Drive URL for Smart Collections
SMART_URL = "https://drive.google.com/uc?id=1_dHsJNwBvsqKza2jWuy4Ml1nBTx54eIy"

# Number of rows already imported (update this based on SELECT COUNT(*) FROM smart_collections)
ROWS_ALREADY_IMPORTED = 1116000  # Replace with the actual count from the database

def download_file(url, output_path):
    """Download file from URL to the specified path."""
    try:
        gdown.download(url, output_path, quiet=False)
        logger.info(f"Downloaded {output_path}")
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        raise

def preprocess_dataframe(df, table_name):
    """Preprocess DataFrame to match table schema and handle data types."""
    expected_columns = {
        'smart_collections': {
            'ID': float,
            'Handle': str,
            'Command': str,
            'Title': str,
            'Body HTML': str,
            'Sort Order': str,
            'Template Suffix': str,
            'Updated At': str,
            'Published': bool,
            'Published At': str,
            'Published Scope': str,
            'Image Src': str,
            'Image Width': float,
            'Image Height': float,
            'Image Alt Text': str,
            'Row #': 'Int64',
            'Top Row': str,
            'Must Match': str,
            'Rule: Product Column': str,
            'Rule: Relation': str,
            'Rule: Condition': str,
            'Product: ID': float,
            'Product: Handle': str
        }
    }

    expected = expected_columns[table_name]
    
    # Log the columns in the DataFrame
    logger.info(f"Columns in {table_name} DataFrame: {list(df.columns)}")
    
    # Rename columns to match table schema
    df_columns = {col: col for col in df.columns}
    for expected_col in expected.keys():
        found = False
        for df_col in df.columns:
            if df_col.strip().replace('"', '') == expected_col.strip().replace('"', ''):
                df_columns[df_col] = expected_col
                found = True
                break
        if not found and expected_col not in df.columns:
            logger.warning(f"Column {expected_col} not found in {table_name} data. Adding with NULL values.")
            df[expected_col] = pd.NA

    df.rename(columns=df_columns, inplace=True)

    # Ensure all expected columns are present
    for col in expected.keys():
        if col not in df.columns:
            df[col] = pd.NA

    # Convert data types
    for col, dtype in expected.items():
        if col not in df.columns:
            continue
        try:
            if dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif dtype == bool:
                df[col] = df[col].astype(str).str.lower().map({
                    'true': True,
                    'false': False,
                    '1': True,
                    '0': False,
                    'yes': True,
                    'no': False
                }).fillna(pd.NA)
            elif dtype == float:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype == 'Int64':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            else:
                df[col] = df[col].astype(str).replace('nan', pd.NA)
        except Exception as e:
            logger.error(f"Error converting column {col} in {table_name} to {dtype}: {str(e)}")
            raise

    # Select only the expected columns
    df = df[list(expected.keys())]
    return df

def import_data_in_batches(df, table_name, batch_size=5000, start_row=0, original_start_row=0):
    """Import DataFrame into the database in batches starting from start_row."""
    total_rows = len(df)
    logger.info(f"Importing {total_rows} rows into {table_name} (original rows {original_start_row + 1} to {original_start_row + total_rows}) in batches of {batch_size}")
    
    for i in range(start_row, total_rows, batch_size):
        batch = df[i:i + batch_size]
        original_batch_start = original_start_row + i + 1
        original_batch_end = original_start_row + min(i + batch_size, total_rows)
        try:
            batch.to_sql(table_name, engine, if_exists='append', index=False, method='multi')
            logger.info(f"Imported batch {(i-start_row)//batch_size + 1}: original rows {original_batch_start} to {original_batch_end} into {table_name}")
        except Exception as e:
            logger.error(f"Error importing batch {(i-start_row)//batch_size + 1} into {table_name}: {str(e)}")
            raise

def main():
    logger.info(f"Starting data import for smart_collections from original row {ROWS_ALREADY_IMPORTED + 1}...")
    
    # Download the file to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        smart_file = os.path.join(tmpdir, "Smart Collection Export.csv")
        download_file(SMART_URL, smart_file)
        
        # Specify dtypes to avoid mixed type warnings
        smart_dtypes = {
            'ID': float,
            'Handle': str,
            'Command': str,
            'Title': str,
            'Body HTML': str,
            'Sort Order': str,
            'Template Suffix': str,
            'Updated At': str,
            'Published': str,
            'Published At': str,
            'Published Scope': str,
            'Image Src': str,
            'Image Width': float,
            'Image Height': float,
            'Image Alt Text': str,
            'Row #': float,
            'Top Row': str,
            'Must Match': str,
            'Rule: Product Column': str,
            'Rule: Relation': str,
            'Rule: Condition': str,
            'Product: ID': float,
            'Product: Handle': str
        }
        
        # Read only the remaining rows
        try:
            smart = pd.read_csv(smart_file, dtype=smart_dtypes, skiprows=range(1, ROWS_ALREADY_IMPORTED + 1))
        except Exception as e:
            logger.error(f"Error reading smart_collections file: {str(e)}")
            raise

        # Preprocess data
        smart = preprocess_dataframe(smart, 'smart_collections')

        # Import remaining rows in larger batches
        import_data_in_batches(smart, 'smart_collections', batch_size=5000, start_row=0, original_start_row=ROWS_ALREADY_IMPORTED)

    logger.info("Data import complete")

if __name__ == "__main__":
    main()