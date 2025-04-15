import numpy as np
import pandas as pd
from classical_model.recommender import recommendations
import os

def recommend_from_saved(query_idx=0, n=5):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "embeddings.npy")
    embeddings = np.load(model_path, allow_pickle=True)
    metadata_path = os.path.join(current_dir, "metadata.csv")
    df = pd.read_csv(metadata_path)

    recs = recommendations(embeddings, df, query_idx=query_idx, n=n)

    return [
        {
            "name": r.get('name', 'Unknown'),
            "category": r.get('categories', 'Unknown'),
            "rating": r.get('stars', 4.0),
            "price": r.get('price_range', 2),
            "address": f"{r.get('city', '')}, {r.get('state', '')}".strip(', ')
        } for _, r in recs.iterrows()
    ]