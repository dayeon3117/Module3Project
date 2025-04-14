import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.gdrive_downloader import download_from_gdrive

FILE_MAP = {
    "embeddings": {
        "id": "1EiTQYAEXhL9WN-pdL949ZQJurvpuHCTg",
        "dest": "backend/classical_model/embeddings.npy"
    },
    "metadata": {
        "id": "1RctNK1Qld3W9rCLjnptJxylewv7IMfnL",
        "dest": "backend/classical_model/metadata.csv"
    },
    "model_weights": {
        "id": "1ZbJa_aP8puYifenXKTVWevEhNFZZ1hKG",
        "dest": "backend/classical_model/model_weights.pth"
    },
    "yelp_data": {
        "id": "1ZEoaLub5AKWTq-9BoBoeBPplb9epzxtA",
        "dest": "backend/naive_model/yelp_data/yelp_academic_dataset_business.json"
    }
}

# Download missing files
for key, val in FILE_MAP.items():
    if not os.path.exists(val["dest"]):
        os.makedirs(os.path.dirname(val["dest"]), exist_ok=True)
        print(f"Downloading {key} from Google Drive...")
        download_from_gdrive(val["id"], val["dest"])

from classical_model.run_classical import recommend_classical
from naive_model.run_naive import recommend_naive
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
