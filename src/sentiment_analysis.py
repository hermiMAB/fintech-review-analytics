import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
# Ensure the required NLTK resources are downloaded
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab')

def preprocess_for_nlp(text):
    """
    Task 2: Tokenization, Stop-word removal, and Lemmatization using NLTK. 
    """
    if pd.isna(text):
        return ""
    
    # 1. Cleaning & Lowercasing: Keep only alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())

    # 2. Tokenization 
    tokens = word_tokenize(text)

    # 3. Stop-word removal & Lemmatization 
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Process tokens: Lemmatize and drop words that are stop-words or too short
    processed_tokens = [
        lemmatizer.lemmatize(t) 
        for t in tokens 
        if t not in stop_words and len(t) > 2
    ]
    
    return " ".join(processed_tokens)

import os
import pandas as pd
from huggingface_hub import InferenceClient

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
env_path = os.path.join(project_root, '.env')

# Explicitly load the key-value pairs from the root .env file into system environment variables
load_dotenv(dotenv_path=env_path)

# Fetch the hidden token securely from system memory
hf_token = os.getenv("HF_TOKEN")

# Initialize the inference engine using the decoupled asset 
client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token,

)

def classify_review_sentiment(text, threshold=0.65):
    """
    Classifies a single review using DistilBERT via Hugging Face Inference API.
    Applies a confidence threshold to introduce a 'neutral' category.
    """
    # Fallback for empty or invalid strings
    if not isinstance(text, str) or len(text).strip() == 0:
        return "neutral", 0.0

    try:
        # Call the serverless Hugging Face API
        response = client.text_classification(
            text,
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        )
        
        # The API returns a list of dicts, usually sorted by highest score first:
        # [{'label': 'POSITIVE', 'score': 0.9998}]
        top_prediction = response[0]
        label = top_prediction['label'].lower() # 'positive' or 'negative'
        confidence = float(top_prediction['score'])
        
        # Rigorous Threshold Check to derive the 'neutral' label
        # If the model is unsure (e.g., confidence is lower than your threshold), classify as neutral
        if confidence < threshold:
            return "neutral", confidence
            
        return label, confidence

    except Exception as e:
        # Return fallback values if the API rate limits or drops out
        print(f"Prediction error on text sample: {str(e)}")
        return "neutral", 0.0