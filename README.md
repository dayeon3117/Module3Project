# Restaurant Recommender System

Welcome to our Module 3 Project! This app helps users find restaurants based on user preferences like location, food type and price. We use three different approaches: a simple ranking method, a machine learning model and a deep learning transformer model.

## Live App

The app is deployed and running on Render:  
**https://module3project.onrender.com**

## Models Included

### 1. Naive Recommender
- Ranks restaurants by Yelp star ratings and review counts
- Simple and fast
- Good for general high-rated suggestions

### 2. Classical Machine Learning Model
- Built using a denoising autoencoder in PyTorch
- Trained on a subset of Yelp data
- Recommends similar restaurants using learned embeddings and cosine similarity

### 3. Deep Learning Recommender (Transformers4Rec)
- Uses a Sentence Transformer to embed restaurant categories
- Accepts a user query and returns relevant results based on cosine similarity
- Strong performance in understanding food type preferences

## Project Structure
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

