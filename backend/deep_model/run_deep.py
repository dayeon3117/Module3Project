import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from deep_model.deep_model import preprocess_yelp_data, encode_descriptions, recommend_restaurants

def recommend_deep(data):
    user_query = data.get("query", "")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("JesseFWarrenV/Yelp-Restaurants", split="train")
    df_raw = pd.DataFrame(dataset)
    
    # Preprocess raw data to produce a DataFrame with a 'description' column
    df_processed = preprocess_yelp_data(df_raw)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the Sentence Transformer model
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Skip caching to disk — compute embeddings directly in memory
    df_final, embedding_matrix = encode_descriptions(df_processed, transformer_model)
    
    # Use cosine similarity to find top recommended restaurants
    recommendations_df = recommend_restaurants(user_query, transformer_model, df_final, embedding_matrix, top_k=5)
    
    return recommendations_df.to_dict(orient="records")
