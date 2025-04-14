import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from deep_model import load_json_lines, preprocess_yelp_data, encode_descriptions, recommend_restaurants

def recommend_deep(data):
    # Get the user's query (e.g., a text string like "Vietnamese cheap")
    user_query = data.get("query", "")
    
    # Compute file paths relative to this file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes your Yelp JSON file is stored one level up in the yelp_data folder
    json_file = os.path.join(current_dir, "yelp_data/yelp_academic_dataset_business.json")
    embed_file = os.path.join(current_dir, "restaurant_embeddings.npy")
    data_file = os.path.join(current_dir, "restaurants_with_embeddings.csv")
    
    # Load raw Yelp data using a helper that loads a JSON Lines file
    df_raw = load_json_lines(json_file)
    
    # Preprocess raw data to produce a DataFrame with a 'description' column
    df_processed = preprocess_yelp_data(df_raw)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the Sentence Transformer model
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Generate or load cached embeddings and processed DataFrame
    df_final, embedding_matrix = encode_descriptions(df_processed, transformer_model, embed_file, data_file)
    
    # Use cosine similarity to find the top recommended restaurants based on the user query
    recommendations_df = recommend_restaurants(user_query, transformer_model, df_final, embedding_matrix, top_k=5)
    
    # Return the recommendations as a list of dictionaries
    return recommendations_df.to_dict(orient="records")
