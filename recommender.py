from sklearn.metrics.pairwise import cosine_similarity

def recommendations(embeddings, dataframe_df, query_idx, n=5):

    query_embedding = embeddings[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities.argsort()[0][-n-1:-1][::-1]
    return dataframe_df.iloc[top_indices]