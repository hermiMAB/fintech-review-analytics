import pandas as pd
import re
import os
from google_play_scraper import app, reviews, Sort


def get_app_metadata(app_id, bank_name):
    """Fetches and prints app store stats."""
    info = app(app_id, lang='en', country='et')
    print(f"\n{'='*30}\n{bank_name} Metadata\n{'='*30}")
    print(f"Score: {info['score']:.2f}")
    print(f"Installs: {info['installs']}")
    return info

def scrape_bank_reviews(app_id, bank_name, count=500):
    """Scrapes raw reviews and returns a raw DataFrame."""
    print(f"Scraping {count} reviews for {bank_name}...")
    
    result, _ = reviews(
        app_id,
        lang='en',
        country='et',
        sort=Sort.NEWEST,
        count=count
    )
    
    # Initial extraction into a list of dicts
    raw_list = []
    for r in result:
        raw_list.append({
            'review_id': r.get('reviewId'),
            'review': r.get('content'),
            'rating': r.get('score'),
            'date': r.get('at'),
            'bank': bank_name,
            'source': 'Google Play'
        })
    
    return pd.DataFrame(raw_list)

def clean_reviews_dataframe(df):
    """Applies logic for duplicates, dates, and text cleaning with row tracking."""
    # Track starting count for reporting 
    initial_count = len(df)
    
    # 1. Deduplicate by ID
    df = df.drop_duplicates(subset=['review_id'], keep='first').copy()
    dupes_removed = initial_count - len(df)
    
    # 2. Handle missing critical text 
    before_drop = len(df)
    df = df.dropna(subset=['review', 'rating'])
    missing_removed = before_drop - len(df)
    
    # 3. Text Normalization
    def _clean_text(text):
        if pd.isna(text): return ''
        text = re.sub(r'\s+', ' ', str(text))
        return text.strip()
    
    df['review'] = df['review'].apply(_clean_text)

    # 4. Remove empty strings after cleaning
    before_empty = len(df)
    df = df[df['review'].str.len() > 2]
    empty_removed = before_empty - len(df)
    
    # 5. Format Date to YYYY-MM-DD [cite: 125]
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # 6. Ensure integer ratings [cite: 121]
    df['rating'] = df['rating'].astype(int)

    # --- PRINT SUMMARY FOR TASK 1 DOCUMENTATION ---
    print(f"--- Data Cleaning Report ---")
    print(f"Total raw records    : {initial_count}")
    print(f"Duplicates removed   : {dupes_removed}")
    print(f"Missing text/rating  : {missing_removed}")
    print(f"Empty after cleaning : {empty_removed}")
    print(f"Final clean records  : {len(df)}")
    print(f"----------------------------\n")
    
    return df[['review', 'rating', 'date', 'bank', 'source']]

def save_processed_data(df, filename):
    """Saves the dataframe to the processed data folder."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_path, exist_ok=True)
    final_path = os.path.join(processed_path, filename)
    df.to_csv(final_path, index=False)
    print(f"Successfully saved to: {final_path}")