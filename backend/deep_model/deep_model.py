import pandas as pd
import numpy as np
import torch
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

def safe_parse_dict(x):
    if isinstance(x, dict):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return {}


def preprocess_yelp_data(df):
    df['attributes'] = df['attributes'].apply(safe_parse_dict)
    df['price'] = df['attributes'].apply(lambda x: x.get('RestaurantsPriceRange2') if isinstance(x, dict) else None)
    df = df[(df['is_open'] == 1) & (df['categories'].notnull())]
    df = df[df['categories'].str.contains("Restaurants", case=False)]
    df['categories'] = df['categories'].str.replace(',', ' ')
    df['price'] = df['price'].fillna("unknown")
    df['description'] = df['categories'] + " price:" + df['price'].astype(str)
    return df


def encode_descriptions(df, model, embed_file, data_file):
    if os.path.exists(embed_file) and os.path.exists(data_file):
        print("Loading cached embeddings and data...")
        df = pd.read_csv(data_file)
        embedding_matrix = np.load(embed_file)
    else:
        print("Encoding and caching new embeddings...")
        descriptions = df['description'].tolist()
        embedding_matrix = model.encode(descriptions, batch_size=64, show_progress_bar=True)
        df['embedding'] = list(embedding_matrix)

        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(embed_file), exist_ok=True)
        os.makedirs(os.path.dirname(data_file), exist_ok=True)

        np.save(embed_file, embedding_matrix)
        df.to_csv(data_file, index=False)

    return df, embedding_matrix



def recommend_restaurants(query, model, df, embedding_matrix, top_k=5):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embedding_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_df = df.iloc[top_indices]

    return top_df.apply(lambda row: {
        "name": row.get("name", ""),
        "category": row.get("categories", ""),
        "rating": row.get("stars", 0),
        "price": "$" * int(row.get("price", 2)),
        "address": f"{row.get('city', '')}, {row.get('state', '')}".strip(', ')
    }, axis=1)
