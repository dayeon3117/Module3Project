# Restaurant Recommender System

Welcome to our Module 3 Project! This app helps users find restaurants based on user preferences like food type and price. We use three different approaches: a simple ranking method, a machine learning model and a deep learning transformer model.

## Live App

The app is deployed and running on Render:  
**https://module3project.onrender.com**

## Data Pipeline

All data was sourced from the Yelp Open Dataset (https://business.yelp.com/data/resources/open-dataset/): real-world data from Yelp related to restaurant businesses. This data includes customer reviews, price range, photos, check-ins, hours of operation, parking availability, and ambiance. We downloaded the JSON provided on this website, and we scraped a subset of that data as a separate JSON to train our models. Then, the data was cleaned optimally and individually for each of our three approaches. Due to the highly detailed nature of the dataset, we chose to focus on features that would be helpful in rudimentary recommendation systems.

## Models Included

### 1. Naive Recommender
- Filters restaurants by food category and price level
- Ranks them by Yelp star rating and number of reviews
- Fast and simple with no training required
- Great for quickly finding the top-rated places for a specific food type
  
### 2. Classical Machine Learning Model (Denoising Autoencoder)
- Learns compressed embeddings of restaurant features using a denoising autoencoder built in PyTorch
- Trained on processed metadata including categories, price and star ratings
- Recommends similar restaurants by comparing embeddings with cosine similarity

### 3. Deep Learning Recommender (Sentence Transformer)
- Uses a Sentence Transformer (all-MiniLM-L6-v2) to embed both user queries and restaurant descriptions
- Compares embeddings using cosine similarity to find the most relevant results
- Handles vague/casual queries better than other models by focusing on meaning rather than exact word matches

## Evaluation
We evaluated all three different recommendation approaches: Naive, Classical Machine Learning (Denoising Autoencoder) and Deep Learning (Transformer-based embeddings). Each model was tested with real user queries in the deployed web application. We compared how well each model returned relevant results based on food type and price. Below is a summary of the observed performance for each method.

### Naive Model
- Approach: The naive model filters the dataset based on whether the restaurant’s categories contain the user’s food input and if the price matches the selected range. It then ranks the filtered results by star rating and review count, bringing the highest rated spots to the top
  
- Performance: This model is very fast and works well when the food input exactly matches terms in the data. For clear and common queries like “brunch” or “sushi,” it almost always gives back strong results with popular places near the top
  
- Observation: While the approach is reliable, it only checks for direct keyword matches in the category field. If a user types something more casual or subjective like 'dinner spots' or 'good Korean BBQ,' the model may fall back to looser matches or return no results if it finds nothing at all. It also doesn’t handle multiple word phrases well if they don’t appear exactly in the data
  
- Best Use Case: The naive model is best when the user has a clear idea of what food they want and just wants to find top-rated places that match. It’s useful for fast lookups of common cuisines and meal types

### Classical Machine Learning Model (Denoising Autoencoder)
- Approach: This model uses a denoising autoencoder built with PyTorch. It is trained on structured restaurant features like category, city, state, price, and star ratings. These features are vectorized and normalized using CountVectorizer, OneHotEncoder, and MinMaxScaler. The autoencoder learns a compressed vector for each restaurant, capturing its core features in a way that allows comparison to other restaurants. When a user provides a query, the system compares its embedding to others using cosine similarity
  
- Performance: The model performed well for broad categories like “brunch” and “sushi.” It returned a mix of options that felt diverse but still relevant. The recommendations were based on similarity of features, not just popularity. So while many results were good matches, they didn’t always include the highest rated places. This is because the model doesn’t directly use star ratings during similarity comparison. It focuses on overall feature embeddings instead
  
- Observation: Compared to the naive model, this one was more flexible and returned a wider range of results. It was better at handling inputs with multiple or slightly varied keywords. But it still struggled with subjective or niche phrases like “dinner spots,” since those don’t map cleanly to structured features
  
- Best Use Case: This model works best when users are exploring and want to see more than just the top rated places. It’s useful when someone is open to different kinds of restaurants and wants suggestions that match the mood or occasion, even if they aren’t sure exactly what to search for

### Deep Learning Model (Sentence Transformer)
- Approach: This model uses a pre-trained Sentence Transformer (all-MiniLM-L6-v2) to turn category and price information into short descriptions, which are then embedded into vectors for matching. It compares these embeddings using cosine similarity, returning results that match the meaning of the query rather than just the specific words. This makes it much better at understanding user intent and matching restaurants based on overall context. Before matching, the model filters the dataset by price and whether the restaurant's categories mention the food type.

- Performance: This model consistently delivered strong results for queries like “brunch” and “Italian,” even when the phrasing didn’t exactly match the categories in the dataset. It handled vague, broad or slightly subjective inputs better than the other models. The recommendations usually felt relevant and reflected the spirit of what the user was asking, not just the literal terms

- Observation: The deep learning model felt the most thoughtful out of the three. It understood the intent behind the query and returned restaurants that made sense even when the wording was a bit fuzzy. For example, it still worked when the query didn’t exactly match the restaurant tags. However, like the other models, it struggled when there was very little data in the category. Queries like “good Korean BBQ” or “dinner spots” returned no results, likely because those specific phrases weren’t well represented in the dataset

- Best Use Case: This model is best when users want more personalized recommendations and don’t want to worry about using the exact right words. It works well for users who know the kind of food or vibe they’re looking for even if they don’t phrase it the same way Yelp does
  
## Ethics Statement
We used the public Yelp Open Dataset and made sure not to include any personal information. The data we used, like restaurant names, categories, and ratings, is already public and was not changed or edited.

Our system does not save any user input or track what people type. All recommendations are made in real time and nothing is stored after.

This tool is meant to help users discover restaurants they might like. It should not be used to make decisions about allergies, dietary needs or anything related to health. We recommend checking directly with restaurants for that kind of info.

## Project Structure
```
Module3Project/  
├── backend/  
│   ├── app.py  
│   ├── requirements.txt  
│  
│   ├── classical_model/  
│   │   ├── autoencoder.py  
│   │   ├── autoencoder_testing.py  
│   │   ├── autoencoder_training.py  
│   │   ├── data_loader.py  
│   │   ├── data_preprocessor.py  
│   │   ├── embeddings.npy  
│   │   ├── metadata.csv  
│   │   ├── model_weights.pth  
│   │   ├── recommender.py  
│   │   ├── run_classical.py  
│   │   └── serve_model.py  
│  
│   ├── deep_model/  
│   │   └── deep_model.py  
│  
│   ├── naive_model/  
│   │   ├── data_loader.py  
│   │   ├── data_preprocessor.py  
│   │   ├── naive_model.py  
│   │   ├── recommender.py  
│   │   └── run_naive.py  
│  
│   ├── utils/  
│   │   └── gdrive_downloader.py  
│  
│   └── yelp_data/  
│       └── yelp_academic_dataset_business.json  
│  
├── frontend/  
│   ├── public/  
│   │   ├── favicon.ico  
│   │   ├── index.html  
│   │   ├── logo192.png  
│   │   ├── logo512.png  
│   │   ├── manifest.json  
│   │   └── robots.txt  
│   ├── src/  
│   │   ├── components/  
│   │   │   ├── InputForm.js  
│   │   │   └── RestaurantCard.js  
│   │   ├── App.css  
│   │   ├── App.js  
│   │   ├── App.test.js  
│   │   ├── Chatbot.js  
│   │   ├── index.css  
│   │   ├── index.js  
│   │   ├── logo.svg  
│   │   ├── reportWebVitals.js  
│   │   ├── setupTests.js  
│   │   └── styles.css  
│   ├── package.json  
│   ├── package-lock.json  
│   └── .gitignore  
│  
├── README.md  
└── .gitignore
```

