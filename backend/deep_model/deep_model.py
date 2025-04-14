import pandas as pd
import numpy as np
import torch
import json
import ast
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_json_lines(json_path):
    """
    Loads a JSONL file where each line is a valid JSON object.
    
    Parameters:
    - json_path (str): Path to the JSONL file.

    Returns:
    - DataFrame containing all parsed records.
    """
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
    return pd.DataFrame(data)


def safe_parse_dict(x):
    """
    Safely parses a string representation of a dictionary if needed.
    
    Parameters:
    - x (str or dict): Input value to parse.

    Returns:
    - A dictionary object or an empty dict on failure.
    """
    if isinstance(x, dict):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return {}


def preprocess_yelp_data(df):
    """
    Processes raw Yelp business DataFrame for restaurant recommendations.
    
    Parameters:
    - df (DataFrame): Raw Yelp business data.

    Returns:
    - DataFrame with processed descriptions for embedding.
    """
    df['attributes'] = df['attributes'].apply(safe_parse_dict)
    df['price'] = df['attributes'].apply(lambda x: x.get('RestaurantsPriceRange2') if isinstance(x, dict) else None)
    df = df[(df['is_open'] == 1) & (df['categories'].notnull())]
    df = df[df['categories'].str.contains("Restaurants", case=False)]
    df['categories'] = df['categories'].str.replace(',', ' ')
    df['price'] = df['price'].fillna("unknown")
    df['description'] = df['categories'] + " price:" + df['price'].astype(str)
    return df


def encode_descriptions(df, model, embed_file, data_file):
    """
    Encodes restaurant descriptions into embeddings and optionally caches them.

    Parameters:
    - df (DataFrame): DataFrame with a 'description' column.
    - model (SentenceTransformer): The transformer model for encoding.
    - embed_file (str): File path to store/load the embeddings.
    - data_file (str): File path to store/load the processed DataFrame.

    Returns:
    - Tuple of (DataFrame, numpy.ndarray of embeddings)
    """
    if os.path.exists(embed_file) and os.path.exists(data_file):
        print("Loading cached embeddings and data...")
        df = pd.read_csv(data_file)
        embedding_matrix = np.load(embed_file)
    else:
        print("Encoding restaurant descriptions...")
        descriptions = df['description'].tolist()
        embedding_matrix = model.encode(descriptions, batch_size=64, show_progress_bar=True)
        df['embedding'] = list(embedding_matrix)
        np.save(embed_file, embedding_matrix)
        df.to_csv(data_file, index=False)
    return df, embedding_matrix


def recommend_restaurants(query, model, df, embedding_matrix, top_k=5):
    """
    Recommends restaurants based on user query.

    Parameters:
    - query (str): The user's food type and price preference query.
    - model (SentenceTransformer): The transformer model used for embedding.
    - df (DataFrame): The dataset of restaurants.
    - embedding_matrix (ndarray): Matrix of restaurant description embeddings.
    - top_k (int): Number of top results to return.

    Returns:
    - DataFrame of top_k recommended restaurants.
    """
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embedding_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][['name', 'categories', 'price', 'stars', 'review_count']]


def main():
    # Define file paths
    json_file = "yelp_academic_dataset_business.json"
    embed_file = "restaurant_embeddings.npy"
    data_file = "restaurants_with_embeddings.csv"

    # Load raw Yelp data
    df_raw = load_json_lines(json_file)

    # Preprocess Yelp restaurant data
    df_processed = preprocess_yelp_data(df_raw)

    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Generate or load embeddings
    df_final, embedding_matrix = encode_descriptions(df_processed, model, embed_file, data_file)

    # Command-line loop for user queries
    while True:
        user_query = input("Enter food type and price (e.g. 'Vietnamese cheap'): ")
        print("\nTop Recommendations:\n")
        print(recommend_restaurants(user_query, model, df_final, embedding_matrix))
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
