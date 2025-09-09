import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import os

# Google Drive direct download links
MODEL_URLS = {
    "trained.h5": "https://drive.google.com/uc?export=download&id=1CrLSlZaEeXXBg9fijpdZ8_X0xm3x0efL",
    "mappings.pkl": "https://drive.google.com/uc?export=download&id=1mZNOltuUJr3LjJDkLikcylnQHVcc48nF"
}

# Global variables for caching
model = None
token2idx, idx2token, tag2idx, idx2tag, maxlen = None, None, None, None, None

def download_file(url, filepath):
    """Download a file from URL if it doesn't exist"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Download the file
    print(f"Downloading {url} to {filepath}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print(f"Download completed: {filepath}")
    return filepath

def load_model_and_mappings():
    global model, token2idx, idx2token, tag2idx, idx2tag, maxlen
    
    if model is not None:
        print("Model already loaded, using cached version")
        return
    
    # Use Netlify's temporary directory
    cache_dir = "/tmp/model_cache"
    trained_path = os.path.join(cache_dir, "trained.h5")
    mappings_path = os.path.join(cache_dir, "mappings.pkl")
    
    # Download files if they don't exist in cache
    if not os.path.exists(trained_path):
        print("Downloading trained.h5...")
        download_file(MODEL_URLS["trained.h5"], trained_path)
    
    if not os.path.exists(mappings_path):
        print("Downloading mappings.pkl...")
        download_file(MODEL_URLS["mappings.pkl"], mappings_path)
    
    # Load model and mappings
    print("Loading model into memory...")
    model = load_model(trained_path)
    
    print("Loading mappings into memory...")
    with open(mappings_path, "rb") as f:
        mappings_data = pickle.load(f)
        # Handle different mapping formats
        if isinstance(mappings_data, tuple) and len(mappings_data) == 5:
            token2idx, idx2token, tag2idx, idx2tag, maxlen = mappings_data
        else:
            # Fallback if format is different
            token2idx = mappings_data.get('token2idx', {})
            idx2token = mappings_data.get('idx2token', {})
            tag2idx = mappings_data.get('tag2idx', {})
            idx2tag = mappings_data.get('idx2tag', {})
            maxlen = mappings_data.get('maxlen', 50)
    
    print("Model and mappings loaded successfully")

def predict_ner(sentence):
    load_model_and_mappings()
    words = sentence.strip().split()
    seq = [token2idx.get(w.lower(), 0) for w in words]  # 0 for unknown token
    seq_padded = pad_sequences([seq], maxlen=maxlen, padding='post')
    pred = model.predict(seq_padded)
    pred_labels = np.argmax(pred, axis=-1)[0][:len(words)]
    tags = [idx2tag.get(i, "O") for i in pred_labels]  # Default to "O" if tag not found
    return list(zip(words, tags))

def handler(event, context):
    print("Received event:", json.dumps(event))
    
    # Handle CORS
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS, GET"
    }
    
    # Handle preflight request
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": headers
        }
    
    # Handle GET request (for testing)
    if event.get("httpMethod") == "GET":
        return {
            "statusCode": 200,
            "headers": {**headers, "Content-Type": "application/json"},
            "body": json.dumps({"message": "NER API is working", "status": "ok"})
        }
    
    # Handle POST request
    if event.get("httpMethod") == "POST":
        try:
            body = event.get("body", "{}")
            if isinstance(body, str):
                data = json.loads(body)
            else:
                data = body
                
            sentence = data.get("sentence", "")
            
            if not sentence:
                return {
                    "statusCode": 400,
                    "headers": {**headers, "Content-Type": "application/json"},
                    "body": json.dumps({"error": "No sentence provided"})
                }
            
            result = predict_ner(sentence)
            
            return {
                "statusCode": 200,
                "headers": {**headers, "Content-Type": "application/json"},
                "body": json.dumps(result)
            }
        except Exception as e:
            print("Error:", str(e))
            return {
                "statusCode": 500,
                "headers": {**headers, "Content-Type": "application/json"},
                "body": json.dumps({"error": str(e)})
            }
    
    # Method not allowed
    return {
        "statusCode": 405,
        "headers": {**headers, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Method not allowed"})
    }
