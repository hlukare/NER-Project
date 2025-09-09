import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import os
import time

# Google Drive direct download links (replace with your actual file IDs)
MODEL_URLS = {
    "trained.h5": "https://drive.google.com/uc?export=download&id=1CrLSlZaEeXXBg9fijpdZ8_X0xm3x0efL",
    "mappings.pkl": "https://drive.google.com/uc?export=download&id=1mZNOltuUJr3LjJDkLikcylnQHVcc48nF",
    "model.h5": "https://drive.google.com/uc?export=download&id=1AEE_uNs0nIof4QjaaY0cBztQQzGn4aWQ"
}

# Global variables for caching - these persist between requests to the same instance
model = None
token2idx, idx2token, tag2idx, idx2tag, maxlen = None, None, None, None, None
model_loaded = False
last_activity_time = time.time()
MAX_IDLE_TIME = 3600  # 1 hour idle time before considering reloading

def download_file(url, filepath):
    """Download a file from URL if it doesn't exist or is outdated"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Download the file
    print(f"Downloading {url} to {filepath}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Download completed: {filepath}")
    return filepath

def load_model_and_mappings():
    global model, token2idx, idx2token, tag2idx, idx2tag, maxlen, model_loaded, last_activity_time
    
    # Update activity time
    last_activity_time = time.time()
    
    if model_loaded:
        print("Model already loaded, using cached version")
        return  # Already loaded
    
    # Use Netlify's temporary directory which persists between requests
    cache_dir = "/tmp/model_cache"
    trained_path = os.path.join(cache_dir, "trained.h5")
    mappings_path = os.path.join(cache_dir, "mappings.pkl")
    model_path = os.path.join(cache_dir, "model.h5")
    
    # Download files if they don't exist in cache
    if not os.path.exists(trained_path):
        print("Downloading trained.h5...")
        download_file(MODEL_URLS["trained.h5"], trained_path)
    
    if not os.path.exists(mappings_path):
        print("Downloading mappings.pkl...")
        download_file(MODEL_URLS["mappings.pkl"], mappings_path)
    
    if not os.path.exists(model_path):
        print("Downloading model.h5...")
        download_file(MODEL_URLS["model.h5"], model_path)
    
    # Load model and mappings
    print("Loading model into memory...")
    try:
        model = load_model(trained_path)
        print("Main model loaded successfully")
    except:
        print("Falling back to model.h5")
        model = load_model(model_path)
    
    print("Loading mappings into memory...")
    with open(mappings_path, "rb") as f:
        mappings_data = pickle.load(f)
        token2idx, idx2token, tag2idx, idx2tag, maxlen = mappings_data
    
    model_loaded = True
    print("Model and mappings loaded successfully")

def predict_ner(sentence):
    load_model_and_mappings()
    words = sentence.strip().split()
    seq = [token2idx.get(w, 0) for w in words]  # 0 for unknown token
    seq_padded = pad_sequences([seq], maxlen=maxlen, padding='post')
    pred = model.predict(seq_padded)
    pred_labels = np.argmax(pred, axis=-1)[0][:len(words)]
    tags = [idx2tag[i] for i in pred_labels]
    return list(zip(words, tags))

def handler(event, context):
    # Handle CORS
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    }
    
    # Handle preflight request
    if event["httpMethod"] == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": headers
        }
    
    # Handle POST request
    if event["httpMethod"] == "POST":
        try:
            body = json.loads(event["body"])
            sentence = body.get("sentence", "")
            
            if not sentence:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "No sentence provided"})
                }
            
            result = predict_ner(sentence)
            
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps(result)
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": headers,
                "body": json.dumps({"error": str(e)})
            }
    
    # Method not allowed
    return {
        "statusCode": 405,
        "headers": headers,
        "body": json.dumps({"error": "Method not allowed"})
    }