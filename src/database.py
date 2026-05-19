

import os
import psycopg2
from dotenv import load_dotenv

if '__file__' in locals() or '__file__' in globals():
    # Running as a regular python script (.py)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
else:
    # Running inside a Jupyter Notebook (.ipynb)
    # Walk upward until we hit the root directory containing your .env
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):
        if '.env' in os.listdir(current_dir):
            break
        current_dir = os.path.dirname(current_dir)
    project_root = current_dir

env_path = os.path.join(project_root, '.env')
# ----------------------------------------------------------------

# Load the local keys into the computer's system memory
load_dotenv(dotenv_path=env_path)

def get_db_connection():
    """
    Securely fetches environment variables to establish and return 
    a raw connection handle using psycopg2.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "bank_reviews"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection critical failure: {e}")
        raise e
    
import pandas as pd

import pandas as pd

def populate_database(csv_path):
    """
    Reads the complete CSV, maps columns to match the SQL schema, 
    populates the banks table, and inserts all reviews into PostgreSQL.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Map (rename) your CSV columns to match the PostgreSQL schema exactly
    df = df.rename(columns={
        'review': 'review_text', 
        'date': 'review_date',
        'hf_sentiment': 'sentiment_label',        
        'hf_directional_score': 'sentiment_score', 
        'themes': 'identified_theme'              
    })
    
    print("Opening database transaction pipeline...")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            
            # 2. Insert unique banks to ensure they exist
            print("Syncing bank metadata...")
            unique_banks = df['bank'].dropna().unique()
            for bank in unique_banks:
                cur.execute(
                    "INSERT INTO banks (bank_name, app_name) VALUES (%s, %s) ON CONFLICT (bank_name) DO NOTHING;",
                    (bank, f"{bank} Mobile App")
                )
            
            # 3. Fetch the newly created Bank IDs to map them to the reviews
            cur.execute("SELECT bank_name, bank_id FROM banks;")
            bank_map = dict(cur.fetchall()) 
            
            # 4. Insert the reviews using the mapped column names
            print(f"Inserting {len(df)} records safely...")
            for _, row in df.iterrows():
                current_bank_id = bank_map.get(row['bank'])
                
                cur.execute(
                    """
                    INSERT INTO reviews 
                    (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, identified_theme, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        current_bank_id,
                        str(row['review_text']),       # Now using the mapped name
                        int(row['rating']), 
                        row['review_date'],            # Now using the mapped name
                        row['sentiment_label'],        # Now using the mapped name
                        float(row['sentiment_score']), # Now using the mapped name
                        str(row['identified_theme']),  # Now using the mapped name
                        str(row['source'])
                    )
                )
                
        # Commit the transaction to save the data permanently
        conn.commit()
        
    print("🚀 Data successfully inserted via modular pipeline!")

