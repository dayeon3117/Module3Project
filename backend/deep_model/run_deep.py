import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from deep_model.deep_model import preprocess_yelp_data, encode_descriptions, recommend_restaurants

def recommend_deep(data):
    user_query = data.get("query", "")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Download from Hugging Face Hub
    embed_file = hf_hub_download(repo_id="JesseFWarrenV/Yelp-Restaurants", filename="restaurant_embeddings.npy")
    data_file = hf_hub_download(repo_id="JesseFWarrenV/Yelp-Restaurants", filename="restaurants_with_embeddings.csv")

    # Load cached embeddings and metadata
    df_final = pd.read_csv(data_file)
    embedding_matrix = np.load(embed_file)

    # Recommend
    recommendations_df = recommend_restaurants(user_query, transformer_model, df_final, embedding_matrix, top_k=20)
    return recommendations_df.to_dict(orient="records")
