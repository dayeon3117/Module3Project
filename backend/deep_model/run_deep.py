import torch
import numpy as np
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from deep_model.deep_model import recommend_restaurants

def safe_parse_embedding(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return None

def recommend_deep(data):
    food = data.get("food", "").lower()
    price = data.get("price", "$")
    price_level = len(price)

    # Use only pandas to reduce memory usage
    df_final = pd.read_csv(
        "restaurants_with_embeddings.csv",
        nrows=5000
    )

    df_final['embedding'] = df_final['embedding'].apply(safe_parse_embedding)
    df_final = df_final[df_final['embedding'].notnull()]

    df_final['category'] = df_final['category'].fillna('').str.lower()
    df_final['price'] = pd.to_numeric(df_final['price'], errors='coerce').fillna(2).astype(int)

    df_filtered = df_final[
        df_final['category'].str.contains(food, case=False, na=False) &
        (df_final['price'] == price_level)
    ]

    if df_filtered.empty:
        return [{
            "name": "No matching restaurants found",
            "category": "",
            "rating": 0,
            "price": 0,
            "address": ""
        }]

    try:
        embedding_matrix = np.vstack(df_filtered['embedding'].values)
    except ValueError:
        return [{
            "name": "Error stacking embeddings",
            "category": "",
            "rating": 0,
            "price": 0,
            "address": ""
        }]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    recommendations_df = recommend_restaurants(
        query=food,
        transformer_model=transformer_model,
        df=df_filtered,
        embedding_matrix=embedding_matrix,
        top_k=20
    )

    return recommendations_df.to_dict(orient="records")
