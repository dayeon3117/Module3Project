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
    food = data.get("food", "").lower()
    price = data.get("price", "$")
    price_level = len(price)

    # Load from Hugging Face dataset (only first 3k rows for safety)
    dataset = load_dataset(
        "JesseFWarrenV/Yelp-Restaurants",
        data_files="restaurants_with_embeddings.csv",
        split="train"
    )
    df_final = pd.DataFrame(dataset).head(3000)

    # Clean + filter embeddings
    df_final['embedding'] = df_final['embedding'].apply(safe_parse_embedding)
    df_final.dropna(subset=['embedding'], inplace=True)

    # Standardize and filter
    df_final['category'] = df_final['categories'].fillna('').str.lower()
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
    except Exception as e:
        return [{
            "name": f"Embedding stack failed: {str(e)}",
            "category": "",
            "rating": 0,
            "price": 0,
            "address": ""
        }]

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    except Exception as e:
        return [{
            "name": f"Model load failed: {str(e)}",
            "category": "",
            "rating": 0,
            "price": 0,
            "address": ""
        }]

    recommendations_df = recommend_restaurants(
        query=food,
        transformer_model=transformer_model,
        df=df_filtered,
        embedding_matrix=embedding_matrix,
        top_k=20
    )

    return recommendations_df.to_dict(orient="records")
