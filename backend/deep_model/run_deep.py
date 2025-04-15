import os
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
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Define paths to save cached embeddings
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    embed_file = os.path.join(cache_dir, "restaurant_embeddings.npy")
    data_file = os.path.join(cache_dir, "restaurants_with_embeddings.csv")
    
    # Encode or load
    df_final, embedding_matrix = encode_descriptions(df_processed, transformer_model, embed_file, data_file)
    
    # Recommend
    recommendations_df = recommend_restaurants(user_query, transformer_model, df_final, embedding_matrix, top_k=5)
    return recommendations_df.to_dict(orient="records")
