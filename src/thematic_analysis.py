import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 1. Define global business themes and associated keyword mappings
theme_keywords = {
    'Performance & Stability': ['update', 'slow', 'crash', 'network', 'error', 'bug', 'loading'],
    'Account & Access': ['login', 'password', 'otp', 'pin', 'register', 'number', 'access'],
    'Transactions & Money': ['transfer', 'money', 'payment', 'balance', 'send', 'receive', 'branch'],
    'General UI/UX': ['easy', 'fast', 'simple', 'design', 'interface', 'user friendly']
}

def assign_themes(text):
    """
    Tags a piece of text with predefined business themes based on keywords.
    """
    if not isinstance(text, str):
        return 'Uncategorized'
    
    found_themes = []
    for theme, keywords in theme_keywords.items():
        if any(keyword in text for keyword in keywords):
            found_themes.append(theme)
            
    return ", ".join(found_themes) if found_themes else 'General Feedback'


def extract_themes_per_bank(df, text_column='nlp_ready_text', bank_column='bank', top_n=10):
    """
    Extracts the highest-scoring TF-IDF words and most frequent bigrams 
    for each individual bank, returning a sorted DataFrame for comparison.
    """
    theme_results = {}
    banks = df[bank_column].dropna().unique()

    for bank in banks:
        bank_corpus = df[df[bank_column] == bank][text_column].dropna()
        
        if len(bank_corpus) < 5:
            continue

        # Extract and sort top unigrams via TF-IDF weights
        tfidf_vec = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vec.fit_transform(bank_corpus)
        mean_weights = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_word_indices = mean_weights.argsort()[::-1][:top_n]
        feature_names = tfidf_vec.get_feature_names_out()
        top_words = [feature_names[i] for i in top_word_indices]

        # Extract and sort top phrases via frequency counts
        bigram_vec = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        bigram_matrix = bigram_vec.fit_transform(bank_corpus)
        bigram_counts = np.asarray(bigram_matrix.sum(axis=0)).flatten()
        top_bigram_indices = bigram_counts.argsort()[::-1][:top_n]
        bigram_features = bigram_vec.get_feature_names_out()
        top_phrases = [bigram_features[i] for i in top_bigram_indices]

        theme_results[bank] = {
            'Top Keywords': ", ".join(top_words),
            'Top Phrases': ", ".join(top_phrases)
        }

    return pd.DataFrame(theme_results).T


def plot_themes_per_bank(df, theme_column='business_themes', bank_column='bank'):
    """
    Plots the normalized distribution of business themes for each individual bank
    with zero-division safety checks.
    """
    df_exploded = df.copy()
    
    df_exploded[theme_column] = df_exploded[theme_column].astype(str).str.split(', ')
    df_exploded = df_exploded.explode(theme_column)

    filtered_df = df_exploded[
        ~df_exploded[theme_column].isin(['General Feedback', 'Uncategorized', 'nan'])
    ]

    if filtered_df.empty:
        print("Warning: No specific business themes found to plot.")
        return

    theme_counts = filtered_df.groupby([bank_column, theme_column]).size().unstack(fill_value=0)
    
    row_sums = theme_counts.sum(axis=1).replace(0, 1)
    theme_pct = theme_counts.div(row_sums, axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    theme_pct.plot(kind='barh', stacked=True, ax=ax, colormap='viridis')

    ax.set_title('Actionable Business Theme Share per Bank (%)', fontsize=14, pad=15)
    ax.set_xlabel('Percentage of Theme Mentions (%)', fontsize=12)
    ax.set_ylabel('Bank', fontsize=12)
    ax.legend(title='Identified Business Themes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()