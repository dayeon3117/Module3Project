from data_preprocessor import preprocess_data
from recommender import naive_recommend
import os

# Use relative path inside the project
current_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(current_dir, "yelp_data")
filename = "yelp_academic_dataset_business.json"

restaurant_df = preprocess_data(directory, filename)

# Generate top recommendations
top_restaurants = naive_recommend(restaurant_df, n=5)

print(top_restaurants[['name', 'stars', 'review_count', 'categories', 'city', 'state']])
