import numpy as np
import pandas as pd
from classical_model.recommender import recommendations
import os

def recommend_classical(data):
    # Extract and preprocess inputs
    food = data.get('food', '').lower()
    price = data.get('price', '$')
    price_level = len(price)

    # Load model data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "embeddings.npy")
    embeddings = np.load(model_path, allow_pickle=True)
    metadata_path = os.path.join(current_dir, "metadata.csv")
    df = pd.read_csv(metadata_path)

    # Clean and prepare data
    df['categories'] = df['categories'].fillna('').str.lower()
    df['price_range'] = pd.to_numeric(df['price_range'], errors='coerce').fillna(2).astype(int)

    # Filter by food type and price level
    filtered_df = df[
        df['categories'].str.contains(food) &
        (df['price_range'] == price_level)
    ]

    print("Filtered total rows:", len(filtered_df))

    if filtered_df.empty:
        print("No exact matches. Returning message.")
        return [{
            "name": "No matches found",
            "category": "",
            "rating": 0,
            "price": 0,
            "address": ""
        }]

    # Recommend based on first match in filtered set
    query_idx = filtered_df.index[0]
    recs = recommendations(embeddings, df, query_idx=query_idx, n=20)

    return [
        {
            "name": r.get('name', 'Unknown'),
            "category": r.get('categories', 'Unknown'),
            "rating": r.get('stars', 4.0),
            "price": r.get('price_range', 2),
            "address": f"{r.get('city', '')}, {r.get('state', '')}".strip(', ')
        } for _, r in recs.iterrows()
    ]
