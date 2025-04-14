from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.classical_model.run_classical import recommend_classical
from backend.naive_model.run_naive import recommend_naive
from backend.classical_model.run_classical import recommend_classical

# from deep_model.run_deep import recommend_deep 

app = Flask(__name__)
CORS(app)  

@app.route("/")
def home():
    return "Restaurant recommender is live!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    model_type = data.get("model")

    try:
        if model_type == "naive":
            return jsonify({"results": recommend_naive(data)})
        elif model_type == "classical":
            return jsonify({"results": recommend_classical(data)})
        elif model_type == "deep":
            return jsonify({"results": recommend_deep(data)})
        else:
            return jsonify({"error": "Unknown model type"}), 400
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
