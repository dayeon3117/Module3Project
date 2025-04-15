import pandas as pd
import numpy as np
import torch
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


def encode_descriptions(df, model):
    descriptions = df['description'].tolist()
    embedding_matrix = model.encode(descriptions, batch_size=64, show_progress_bar=True)
    return df, np.array(embedding_matrix)


def recommend_restaurants(query, model, df, embedding_matrix, top_k=5):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embedding_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][['name', 'categories', 'price', 'stars', 'review_count']]
