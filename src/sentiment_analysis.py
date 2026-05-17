import warnings, re, os, json
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline

# 1. Download necessary NLTK components
# Note: Added 'punkt_tab' to your list to prevent the LookupError from earlier
for res in ['vader_lexicon', 'stopwords', 'averaged_perceptron_tagger', 'punkt', 'wordnet', 'punkt_tab']:
    nltk.download(res, quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (11, 5)
print("Environment ready ✓")

# 2. Load Local Hugging Face Model (No API Key required)
print("Loading local Hugging Face model (this may take a moment the first time)...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True, 
    max_length=512,
    token=False
)

def modular_nlp_pipeline(text):
    """
    Task 2: Tokenization, Stop-word removal, and Lemmatization using NLTK. 
    """
    if pd.isna(text):
        return ""
    
    # Cleaning & Lowercasing: Keep only alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())

    # Tokenization
    tokens = word_tokenize(text)

    # Stop-word removal & Lemmatization
    lemmatizer = WordNetLemmatizer()
    processed = [
        lemmatizer.lemmatize(t) for t in tokens 
        if t not in stop_words and len(t) > 2
    ]
    
    return " ".join(processed)


def get_vader_label(score):
    """Assigns positive, negative, or neutral based on VADER compound threshold."""
    return 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')


def classify_hf_local(text, threshold=0.65):
    """
    Classifies a single review using the local DistilBERT pipeline.
    Applies a confidence threshold to introduce a 'neutral' category.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "neutral", 0.0

    try:
        # Predict using local model
        result = classifier(text)[0]
        label = result['label'].lower() # 'positive' or 'negative'
        confidence = float(result['score'])
        
        # Rigorous Threshold Check to derive the 'neutral' label
        if confidence < threshold:
            return "neutral", confidence
            
        return label, confidence

    except Exception as e:
        print(f"Prediction error on text sample: {str(e)}")
        return "neutral", 0.0