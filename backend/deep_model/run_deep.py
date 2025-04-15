import torch
import numpy as np
import pandas as pd
import ast
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from deep_model.deep_model import recommend_restaurants

def safe_parse_embedding(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return None

def recommend_deep(data):
    user_query = data.get("query", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    dataset = load_dataset(
        "JesseFWarrenV/Yelp-Restaurants",
        data_files="restaurants_with_embeddings.csv",
        split="train"
    )
    df_final = pd.DataFrame(dataset)

    df_final['embedding'] = df_final['embedding'].apply(safe_parse_embedding)
    df_final = df_final[df_final['embedding'].notnull()]

    # Extra safety check before vstack
    if len(df_final) == 0:
        print("⚠️ No valid embeddings to stack. Returning fallback.")
        return [{
            "name": "No valid embeddings found",
            "category": "",
            "rating": 0,
            "price": 0,
            "address": ""
        }]

    try:
        embedding_matrix = np.vstack(df_final['embedding'].values)
    except ValueError as e:
        print(f"Embedding stacking failed: {e}")
        return [{
            "name": "Error parsing embeddings",
            "category": "",
            "rating": 0,
            "price": 0,
            "address": ""
        }]

    recommendations_df = recommend_restaurants(
        user_query, transformer_model, df_final, embedding_matrix, top_k=20
    )
    return recommendations_df.to_dict(orient="records")
