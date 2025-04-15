import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from deep_model.deep_model import recommend_restaurants

def recommend_deep(data):
    user_query = data.get("query", "")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Load from Hugging Face (CSV already has descriptions + embeddings)
    dataset = load_dataset("JesseFWarrenV/Yelp-Restaurants", data_files="restaurants_with_embeddings.csv", split="train")
    df_final = pd.DataFrame(dataset)

    # Parse stringified embeddings
    df_final['embedding'] = df_final['embedding'].apply(eval)
    embedding_matrix = np.vstack(df_final['embedding'].values)

    # Recommend
    recommendations_df = recommend_restaurants(user_query, transformer_model, df_final, embedding_matrix, top_k=20)
    return recommendations_df.to_dict(orient="records")
