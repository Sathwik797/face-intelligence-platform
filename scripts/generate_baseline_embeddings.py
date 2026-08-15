import os
import sys
import pickle
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from ml.detector import DlibHOGDetector
from ml.embedder import DlibEmbedder

def build_baseline_embeddings(
    dataset_dir: str,
    output_path: str
) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
    """
    Extracts baseline dlib 128D face embeddings from the dataset and saves them to disk.

    NOTE: This is offline feature extraction using a pre-trained network, NOT model training.

    Args:
        dataset_dir (str): Path to directory containing identity subfolders.
        output_path (str): Filepath to save the serialized (encodings, names) tuple.

    Returns:
        Tuple[List[np.ndarray], List[str], Dict[str, Any]]: (encodings, names, summary_stats)
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    detector = DlibHOGDetector()
    embedder = DlibEmbedder()

    known_encodings: List[np.ndarray] = []
    known_names: List[str] = []

    stats = {
        "identities_processed": 0,
        "images_scanned": 0,
        "embeddings_extracted": 0,
        "skipped_no_face": 0,
        "multiple_faces_found": 0,
        "invalid_images": 0
    }

    identity_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"[INFO] Found {len(identity_folders)} identity folders in {dataset_dir}")

    for person_name in identity_folders:
        person_path = os.path.join(dataset_dir, person_name)
        stats["identities_processed"] += 1
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        image_files = [f for f in os.listdir(person_path) if os.path.splitext(f)[1].lower() in valid_extensions]

        for image_name in image_files:
            image_path = os.path.join(person_path, image_name)
            stats["images_scanned"] += 1

            # Read image with OpenCV
            bgr_image = cv2.imread(image_path)
            if bgr_image is None:
                print(f"[WARNING] Skipping unreadable image: {image_path}")
                stats["invalid_images"] += 1
                continue

            # Convert BGR to RGB for dlib / face_recognition
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # Detect face
            face_locations = detector.detect(rgb_image)

            if len(face_locations) == 0:
                print(f"[WARNING] No face detected in {image_path}. Skipping.")
                stats["skipped_no_face"] += 1
                continue

            if len(face_locations) > 1:
                print(f"[WARNING] Multiple faces ({len(face_locations)}) detected in {image_path}. Using primary (largest) face.")
                stats["multiple_faces_found"] += 1
                # Sort by area (bottom - top) * (right - left) descending
                face_locations = sorted(
                    face_locations,
                    key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
                    reverse=True
                )[:1]

            # Extract 128D embedding
            embeddings = embedder.embed(rgb_image, face_locations)
            if len(embeddings) > 0:
                known_encodings.append(embeddings[0])
                known_names.append(person_name)
                stats["embeddings_extracted"] += 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Save to disk
    with open(output_path, "wb") as f:
        pickle.dump((known_encodings, known_names), f)

    print(f"\n[SUMMARY] Embedding Extraction Complete:")
    print(f"  - Identities: {stats['identities_processed']}")
    print(f"  - Images scanned: {stats['images_scanned']}")
    print(f"  - Encodings extracted: {stats['embeddings_extracted']}")
    print(f"  - Skipped (no face): {stats['skipped_no_face']}")
    print(f"  - Multi-face warnings: {stats['multiple_faces_found']}")
    print(f"  - Saved to: {output_path}")

    return known_encodings, known_names, stats


if __name__ == "__main__":
    config = load_config()
    dataset_path = config.get("paths", {}).get("dataset_dir", "dataset")
    model_output_path = config.get("paths", {}).get("embeddings_path", "trained_model/face_encodings.pkl")

    build_baseline_embeddings(dataset_path, model_output_path)
