"""
Legacy runner for baseline feature extraction.

NOTE: This script performs offline feature extraction using a pre-trained network.
It does not perform model training.
For the new modular workflow, please use:
    python scripts/generate_baseline_embeddings.py
"""

from scripts.generate_baseline_embeddings import build_baseline_embeddings
from config import load_config

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    dataset_path = config.get("paths", {}).get("dataset_dir", "dataset")
    model_output_path = config.get("paths", {}).get("embeddings_path", "trained_model/face_encodings.pkl")

    print("[INFO] Invoking baseline embedding extraction pipeline...")
    build_baseline_embeddings(dataset_path, model_output_path)
