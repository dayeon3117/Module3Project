import pandas as pd
from naive_model.recommender import naive_recommend
from datasets import load_dataset

def load_yelp_data():
    dataset = load_dataset("JesseFWarrenV/Yelp-Restaurants", split="train")
    return dataset.to_pandas()

def recommend_naive(data):
    food = data.get('food', '').lower()
    price = data.get('price', '$')
    price_level = len(price)

    df = load_yelp_data()
    df['categories'] = df['categories'].fillna('').str.lower()

    if 'attributes' in df.columns:
        df['price_range'] = df['attributes'].apply(
            lambda x: x.get('RestaurantsPriceRange2', 2) if isinstance(x, dict) else 2
        )
    else:
        df['price_range'] = 2

    df['price_range'] = pd.to_numeric(df['price_range'], errors='coerce').fillna(2).astype(int)

    # Filter by category + price
    filtered_df = df[
        df['categories'].str.contains(food) &
        (df['price_range'] == price_level)
    ]

    print("Filtered total rows:", len(filtered_df))

    if filtered_df.empty:
        print("No exact matches. Returning fallback.")
        filtered_df = df[df['categories'].str.contains(food)]

    if filtered_df.empty:
        return [{
            "name": "No matches found",
            "category": "",
            "rating": 0,
            "price": 0,
            "address": ""
        }]

    top_recs = naive_recommend(filtered_df, n=20)

    return [
        {
            "name": r.get('name', 'Unknown'),
            "category": r.get('categories', 'Unknown'),
            "rating": r.get('stars', 4.0),
            "price": r.get('price_range', 2),
            "address": f"{r.get('city', '')}, {r.get('state', '')}".strip(', ')
        }
        for _, r in top_recs.iterrows()
    ]
