import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from classical_model.data_loader import load_json_to_dataframe
from classical_model.autoencoder import denoising_autoencoder
from classical_model.autoencoder_training import train_autoencoder
from classical_model.autoencoder_testing import test_model
from classical_model.recommender import recommendations
import torch

def run_pipeline(query_idx=0, n=5, save_outputs=True):
    # Load data
    directory = os.path.join(os.path.dirname(__file__), "yelp_data")
    filename = "yelp_academic_dataset_business.json"
    business_df = load_json_to_dataframe(directory, filename)
    restaurant_df = business_df[business_df['categories'].str.contains('Restaurants', case=False, na=False)]

    # Select and preprocess features
    features = restaurant_df[['categories', 'stars', 'city', 'state', 'attributes']].copy()

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '), binary=True, token_pattern=None)
    category_features = vectorizer.fit_transform(features['categories'].fillna(''))

    features['stars'] = MinMaxScaler().fit_transform(features[['stars']])

    encoder = OneHotEncoder(sparse_output=False)
    location_features = encoder.fit_transform(features[['city', 'state']].fillna('Unknown'))

    features['price_range'] = features['attributes'].apply(
        lambda x: x.get('RestaurantsPriceRange2', 2) if isinstance(x, dict) else 2
    )
    features['price_range'] = pd.to_numeric(features['price_range'], errors='coerce')
    features['price_range'] = features['price_range'].fillna(features['price_range'].median())
    price_features = MinMaxScaler().fit_transform(features[['price_range']])

    X = scipy.sparse.hstack([
        category_features,
        location_features,
        price_features,
        features['stars'].values.reshape(-1, 1)
    ]).toarray()

    # Train/test split (fix: split both features and original DataFrame)
    X_train, X_test, df_train, df_test = train_test_split(
        X, restaurant_df, test_size=0.2, random_state=0
    )

    print(f"Shape of X_train: {X_train.shape}")
    input_dim = X_train.shape[1]
    print(f"Number of features (input_dim): {input_dim}")

    # Model training
    encoding_dimensions = 64
    model = denoising_autoencoder(input_dim, encoding_dimensions)
    training = train_autoencoder(model, X_train, epochs=50, batch_size=256)

    # Embedding generation
    embedding_results = test_model(training, X_test)

    # Save everything for deployment
    if save_outputs:
        save_dir = os.path.dirname(__file__)
        torch.save(training.state_dict(), os.path.join(save_dir, "model_weights.pth"))
        np.save(os.path.join(save_dir, "embeddings.npy"), embedding_results)
        # Recalculate price_range for saving in df_test
        df_test['price_range'] = df_test['attributes'].apply(
            lambda x: x.get('RestaurantsPriceRange2', 2) if isinstance(x, dict) else 2
        )
        df_test.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)
        print("Model, embeddings, and metadata saved.")

    # Return formatted recommendations for testing
    recs = recommendations(embedding_results, df_test, query_idx=query_idx, n=n)
    return recs.to_dict(orient="records")

if __name__ == "__main__":
    run_pipeline()
