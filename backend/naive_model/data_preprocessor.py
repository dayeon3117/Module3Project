import pandas as pd
from naive_model.data_loader import load_json_to_dataframe

def preprocess_data(directory, filename):
    df = load_json_to_dataframe(directory, filename)

    # Filter only restaurant businesses
    restaurant_df = df[df['categories'].str.contains('Restaurants', case=False, na=False)].copy()

    restaurant_df = restaurant_df[[
        'business_id', 'name', 'stars', 'review_count', 'categories', 'city', 'state'
    ]]

    # Drop rows with missing star ratings
    restaurant_df.dropna(subset=['stars'], inplace=True)

    return restaurant_df
