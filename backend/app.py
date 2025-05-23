import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.gdrive_downloader import download_from_gdrive

FILE_MAP = {
    "metadata": {
        "id": "1RctNK1Qld3W9rCLjnptJxylewv7IMfnL",
        "dest": "classical_model/metadata.csv"
    },
    "model_weights": {
        "id": "1ZbJa_aP8puYifenXKTVWevEhNFZZ1hKG",
        "dest": "classical_model/model_weights.pth"
    },
    "naive_yelp_data": {
        "id": "1ZEoaLub5AKWTq-9BoBoeBPplb9epzxtA",
        "dest": "naive_model/yelp_data/yelp_academic_dataset_business.json"
    }
}

for key, val in FILE_MAP.items():
    if not os.path.exists(val["dest"]):
        os.makedirs(os.path.dirname(val["dest"]), exist_ok=True)
        print(f"Downloading {key} from Google Drive...")
        download_from_gdrive(val["id"], val["dest"])

files_to_check = [
    "classical_model/embeddings.npy",
    "classical_model/metadata.csv",
    "classical_model/model_weights.pth",
    "naive_model/yelp_data/yelp_academic_dataset_business.json"
]

print("Checking required files...")

for rel_path in files_to_check:
    abs_path = os.path.abspath(rel_path)
    if os.path.exists(abs_path):
        print(f"File exists: {abs_path}")
    else:
        print(f"File NOT found: {abs_path}")

from classical_model.run_classical import recommend_classical
from naive_model.run_naive import recommend_naive
from deep_model.run_deep import recommend_deep

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
