from backend.classical_model.data_preprocessor import run_pipeline

def recommend_classical(data):
    # Currently unused, but available for future filters
    mood = data.get('mood')
    food = data.get('food')
    price = data.get('price')

    # Run the autoencoder pipeline and get recs
    results = run_pipeline(query_idx=0, n=5)

    # Format results for frontend
    formatted_results = []
    for r in results:
        formatted_results.append({
            "name": r.get('name', 'Unknown'),
            "category": r.get('categories', 'Unknown'),
            "rating": r.get('stars', 4.0),
            "price": r.get('price_range', 2),  # pulled from processed features
            "address": f"{r.get('city', '')}, {r.get('state', '')}".strip(', ')
        })

    return formatted_results
