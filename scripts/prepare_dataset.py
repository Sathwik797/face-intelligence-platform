import os
import sys
import shutil
import hashlib
import random
import json
from typing import Dict, List, Tuple, Any
from PIL import Image
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config

def calculate_sha256(filepath: str) -> str:
    """Calculates SHA256 checksum for a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_lfw_pairs(pairs_path: str, lfw_dir: str) -> pd.DataFrame:
    """
    Parses the official LFW 10-fold pairs.txt file into a structured DataFrame.
    """
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    records = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # First line contains: num_folds (10), pairs_per_fold (300)
    header = lines[0].split()
    num_folds = int(header[0])
    num_pairs_per_fold = int(header[1])

    current_line = 1
    pair_id = 1

    for fold in range(1, num_folds + 1):
        # 1. Matched pairs (same identity)
        for _ in range(num_pairs_per_fold):
            parts = lines[current_line].split()
            current_line += 1
            name = parts[0]
            num1 = int(parts[1])
            num2 = int(parts[2])
            img1_rel = os.path.join(name, f"{name}_{num1:04d}.jpg")
            img2_rel = os.path.join(name, f"{name}_{num2:04d}.jpg")
            img1_full = os.path.join(lfw_dir, img1_rel)
            img2_full = os.path.join(lfw_dir, img2_rel)

            records.append({
                "pair_id": pair_id,
                "fold": fold,
                "identity1": name,
                "image1_rel_path": img1_rel.replace("\\", "/"),
                "image1_exists": os.path.exists(img1_full),
                "identity2": name,
                "image2_rel_path": img2_rel.replace("\\", "/"),
                "image2_exists": os.path.exists(img2_full),
                "is_same": 1
            })
            pair_id += 1

        # 2. Mismatched pairs (different identities)
        for _ in range(num_pairs_per_fold):
            parts = lines[current_line].split()
            current_line += 1
            name1 = parts[0]
            num1 = int(parts[1])
            name2 = parts[2]
            num2 = int(parts[3])
            img1_rel = os.path.join(name1, f"{name1}_{num1:04d}.jpg")
            img2_rel = os.path.join(name2, f"{name2}_{num2:04d}.jpg")
            img1_full = os.path.join(lfw_dir, img1_rel)
            img2_full = os.path.join(lfw_dir, img2_rel)

            records.append({
                "pair_id": pair_id,
                "fold": fold,
                "identity1": name1,
                "image1_rel_path": img1_rel.replace("\\", "/"),
                "image1_exists": os.path.exists(img1_full),
                "identity2": name2,
                "image2_rel_path": img2_rel.replace("\\", "/"),
                "image2_exists": os.path.exists(img2_full),
                "is_same": 0
            })
            pair_id += 1

    df_pairs = pd.DataFrame(records)
    return df_pairs


def prepare_identification_splits(
    lfw_dir: str,
    output_eval_dir: str,
    min_faces: int = 20,
    enrollment_samples: int = 2,
    val_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects identities meeting min_faces and partitions images into
    Enrollment, Validation, and Test splits with zero identity leakage.
    """
    rng = random.Random(seed)

    # Clean existing evaluation directory to prevent stale files
    if os.path.exists(output_eval_dir):
        shutil.rmtree(output_eval_dir)
    os.makedirs(output_eval_dir, exist_ok=True)

    enrollment_dir = os.path.join(output_eval_dir, "enrollment")
    validation_dir = os.path.join(output_eval_dir, "validation")
    test_dir = os.path.join(output_eval_dir, "test")

    os.makedirs(enrollment_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Find qualified identities
    all_identities = sorted([d for d in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, d))])
    qualified_identities = []

    for name in all_identities:
        folder = os.path.join(lfw_dir, name)
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) >= min_faces:
            qualified_identities.append((name, sorted(images)))

    print(f"[INFO] Qualified identities with >= {min_faces} images: {len(qualified_identities)}")

    identities_summary = []
    splits_records = []
    image_id_counter = 1

    for identity, images in qualified_identities:
        # Deterministic shuffle
        shuffled_images = list(images)
        rng.shuffle(shuffled_images)

        enroll_imgs = shuffled_images[:enrollment_samples]
        remaining = shuffled_images[enrollment_samples:]

        num_val = max(1, int(len(remaining) * val_ratio))
        val_imgs = remaining[:num_val]
        test_imgs = remaining[num_val:]

        identities_summary.append({
            "identity": identity,
            "total_images": len(images),
            "enrollment_count": len(enroll_imgs),
            "validation_count": len(val_imgs),
            "test_count": len(test_imgs)
        })

        split_map = [
            ("enrollment", enroll_imgs, enrollment_dir),
            ("validation", val_imgs, validation_dir),
            ("test", test_imgs, test_dir),
        ]

        for split_name, img_list, dest_root in split_map:
            dest_folder = os.path.join(dest_root, identity)
            os.makedirs(dest_folder, exist_ok=True)

            for img_name in img_list:
                src_path = os.path.join(lfw_dir, identity, img_name)
                dest_path = os.path.join(dest_folder, img_name)
                shutil.copy2(src_path, dest_path)

                # Compute image properties
                with Image.open(dest_path) as img:
                    width, height = img.size

                file_hash = calculate_sha256(dest_path)
                rel_path = os.path.relpath(dest_path, PROJECT_ROOT).replace("\\", "/")

                splits_records.append({
                    "image_id": image_id_counter,
                    "identity": identity,
                    "split": split_name,
                    "image_name": img_name,
                    "relative_path": rel_path,
                    "file_hash": file_hash,
                    "width": width,
                    "height": height
                })
                image_id_counter += 1

    df_identities = pd.DataFrame(identities_summary)
    df_splits = pd.DataFrame(splits_records)
    return df_identities, df_splits


def run_dataset_preparation():
    """Main preparation entrypoint."""
    config = load_config("config/config.yaml")

    lfw_raw_dir = config.get("paths", {}).get("raw_lfw_dir", "data/raw/lfw")
    # Subfolder where scikit-learn stores images
    lfw_extracted_dir = os.path.join(lfw_raw_dir, "lfw_home", "lfw_funneled")
    if not os.path.exists(lfw_extracted_dir):
        lfw_extracted_dir = lfw_raw_dir

    pairs_path = os.path.join(lfw_raw_dir, "lfw_home", "pairs.txt")
    eval_dir = config.get("paths", {}).get("evaluation_dir", "data/evaluation")
    meta_dir = config.get("paths", {}).get("metadata_dir", "data/metadata")
    os.makedirs(meta_dir, exist_ok=True)

    min_faces = config.get("dataset", {}).get("min_faces_per_person", 20)
    enroll_samples = config.get("dataset", {}).get("enrollment_samples", 2)
    val_ratio = config.get("dataset", {}).get("val_ratio", 0.5)
    seed = config.get("dataset", {}).get("random_seed", 42)

    print("="*60)
    print("PHASE 2: DATASET PREPARATION & EVALUATION SETUP")
    print("="*60)
    print(f"LFW Image Source: {lfw_extracted_dir}")
    print(f"Evaluation Target: {eval_dir}")
    print(f"Random Seed: {seed}")
    print(f"Min Images / Person: {min_faces}")

    # 1. Track A: Verification Pairs Metadata
    if os.path.exists(pairs_path):
        print("\n[TRACK A] Processing Official LFW 10-Fold Verification Pairs...")
        df_pairs = parse_lfw_pairs(pairs_path, lfw_extracted_dir)
        pairs_out = os.path.join(meta_dir, "verification_pairs.csv")
        df_pairs.to_csv(pairs_out, index=False)
        print(f"  -> Saved {len(df_pairs)} verification pairs across 10 folds to {pairs_out}")
    else:
        print(f"\n[WARNING] Pairs file not found at {pairs_path}. Skipping Track A.")

    # 2. Track B: Identification Dataset Splits
    print("\n[TRACK B] Generating Identification Splits (Enrollment / Validation / Test)...")
    df_identities, df_splits = prepare_identification_splits(
        lfw_dir=lfw_extracted_dir,
        output_eval_dir=eval_dir,
        min_faces=min_faces,
        enrollment_samples=enroll_samples,
        val_ratio=val_ratio,
        seed=seed
    )

    identities_out = os.path.join(meta_dir, "identities.csv")
    splits_out = os.path.join(meta_dir, "splits.csv")
    df_identities.to_csv(identities_out, index=False)
    df_splits.to_csv(splits_out, index=False)

    print(f"  -> Saved {len(df_identities)} selected identities to {identities_out}")
    print(f"  -> Saved {len(df_splits)} image split records to {splits_out}")
    print(f"  - Enrollment Images: {len(df_splits[df_splits['split'] == 'enrollment'])}")
    print(f"  - Validation Images: {len(df_splits[df_splits['split'] == 'validation'])}")
    print(f"  - Test Images: {len(df_splits[df_splits['split'] == 'test'])}")
    print("\n[SUCCESS] Dataset preparation complete!")


if __name__ == "__main__":
    run_dataset_preparation()
