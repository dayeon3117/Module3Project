import numpy as np
import pandas as pd
from classical_model.recommender import recommendations

def recommend_from_saved(query_idx=0, n=5):
    model_path = os.path.join(os.path.dirname(__file__), "embeddings.npy")
    embeddings = np.load(model_path, allow_pickle=True)
    df = pd.read_csv("backend/classical_model/metadata.csv")

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