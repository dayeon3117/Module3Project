import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from data_loader import load_json_to_dataframe
from Module3Project.backend.classical_model.autoencoder_training import train_autoencoder
from Module3Project.backend.classical_model.autoencoder_testing import test_model
from Module3Project.backend.classical_model.autoencoder import denoising_autoencoder
from Module3Project.backend.classical_model.recommender import recommendations

directory= "C:/Users/santo/Documents/AIPI540/Assignment3a/Yelp-JSON/Yelp JSON/yelp_dataset"
filename="yelp_academic_dataset_business.json"
business_df = load_json_to_dataframe(directory, filename)


restaurant_df = business_df[business_df['categories'].str.contains('Restaurants', case=False, na=False)]

# Select features for the autoencoder
features = restaurant_df[['categories', 'stars', 'city', 'state', 'attributes']].copy()

# Preprocess features:

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '), binary=True, token_pattern=None)
category_features = vectorizer.fit_transform(features['categories'].fillna(''))


features['stars'] = MinMaxScaler().fit_transform(features[['stars']])


encoder = OneHotEncoder(sparse_output=False)
location_features = encoder.fit_transform(features[['city', 'state']].fillna('Unknown'))


features['price_range'] = features['attributes'].apply(
    lambda x: x.get('RestaurantsPriceRange2', 2) if isinstance(x, dict) else 2
)
# Changing empty strings to Nans
features['price_range'] = pd.to_numeric(features['price_range'], errors='coerce')  
# Fill nans with median of column
features['price_range'] = features['price_range'].fillna(features['price_range'].median())
price_features = MinMaxScaler().fit_transform(features[['price_range']])
X = scipy.sparse.hstack([category_features, location_features, price_features, features['stars'].values.reshape(-1, 1)]).toarray()
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

# Generate model
datainput = X_train.shape[1]

# Debugging
# Checking shape of X_train
print(f"Shape of X_train: {X_train.shape}")

# Extracting the number of features
input_dim = X_train.shape[1]
print(f"Number of features (input_dim): {input_dim}")

## Parameters
encoding_dimensions=64
model=denoising_autoencoder(datainput,encoding_dimensions)

# Train model
## Parameters
epochs=50
batch_size=256
training=train_autoencoder(model, X_train, epochs, batch_size)


# Test model embeddings
embedding_results=test_model(training, X_test)

recommendation_results = recommendations(embedding_results, restaurant_df, query_idx=0, n=5)

print(recommendation_results[['categories', 'stars', 'city', 'state', 'attributes']])