def naive_recommend(restaurants_df, n=5):
    top_restaurants = restaurants_df.sort_values(by=['stars', 'review_count'], ascending=False)
    return top_restaurants.head(n)
