import os
import sys
import json
import hashlib
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from ml.detector import DlibHOGDetector

def calculate_sha256(filepath: str) -> str:
    """Calculates SHA256 checksum for a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_dataset_validation(
    run_detector_check: bool = True,
    max_detector_samples: int = 300
) -> Dict[str, Any]:
    """
    Validates dataset integrity, checks for cross-split leakage,
    verifies file readability, audits hashes, checks face detectability,
    and writes data/metadata/dataset_summary.json.

    Returns:
        Dict[str, Any]: Validation report and summary metrics.
    """
    config = load_config("config/config.yaml")
    meta_dir = config.get("paths", {}).get("metadata_dir", "data/metadata")
    splits_csv = os.path.join(meta_dir, "splits.csv")
    identities_csv = os.path.join(meta_dir, "identities.csv")

    if not os.path.exists(splits_csv) or not os.path.exists(identities_csv):
        raise FileNotFoundError(f"Metadata files missing! Run scripts/prepare_dataset.py first.")

    df_splits = pd.read_csv(splits_csv)
    df_identities = pd.read_csv(identities_csv)

    print("="*60)
    print("PHASE 2: DATASET INTEGRITY & LEAKAGE VALIDATION")
    print("="*60)

    errors = []
    warnings = []

    # 1. Check Missing / Unreadable Files & Image Dimensions
    unreadable_count = 0
    dimensions_set = set()
    hashes_by_split = {"enrollment": set(), "validation": set(), "test": set()}
    hashes_to_images = {}

    for _, row in df_splits.iterrows():
        filepath = os.path.join(PROJECT_ROOT, row["relative_path"])
        split = row["split"]

        if not os.path.exists(filepath):
            errors.append(f"Missing file: {filepath}")
            continue

        try:
            with Image.open(filepath) as img:
                img.verify()
            with Image.open(filepath) as img:
                width, height = img.size
                dimensions_set.add((width, height))
        except Exception as e:
            errors.append(f"Corrupted/Unreadable image: {filepath} ({e})")
            unreadable_count += 1
            continue

        # Hash audit
        fhash = calculate_sha256(filepath)
        hashes_by_split[split].add(fhash)
        if fhash not in hashes_to_images:
            hashes_to_images[fhash] = []
        hashes_to_images[fhash].append((split, row["identity"], row["image_name"]))

    # 2. Check for Duplicate Files within and Cross-Split
    cross_split_leakage = []
    enroll_hashes = hashes_by_split["enrollment"]
    val_hashes = hashes_by_split["validation"]
    test_hashes = hashes_by_split["test"]

    # Intersections
    leak_enroll_val = enroll_hashes.intersection(val_hashes)
    leak_enroll_test = enroll_hashes.intersection(test_hashes)
    leak_val_test = val_hashes.intersection(test_hashes)

    if leak_enroll_val:
        errors.append(f"Cross-split leakage detected between Enrollment and Validation: {len(leak_enroll_val)} duplicate hashes!")
    if leak_enroll_test:
        errors.append(f"Cross-split leakage detected between Enrollment and Test: {len(leak_enroll_test)} duplicate hashes!")
    if leak_val_test:
        errors.append(f"Cross-split leakage detected between Validation and Test: {len(leak_val_test)} duplicate hashes!")

    # 3. Check Insufficient Samples per Identity
    min_faces = config.get("dataset", {}).get("min_faces_per_person", 20)
    for _, row in df_identities.iterrows():
        if row["total_images"] < min_faces:
            errors.append(f"Identity {row['identity']} has only {row['total_images']} images (< {min_faces}).")
        if row["enrollment_count"] < 1:
            errors.append(f"Identity {row['identity']} has zero enrollment images.")
        if row["validation_count"] < 1:
            errors.append(f"Identity {row['identity']} has zero validation images.")
        if row["test_count"] < 1:
            errors.append(f"Identity {row['identity']} has zero test images.")

    # 4. Detectability Check using Phase 1 DlibHOGDetector
    detector_stats = {
        "images_checked": 0,
        "single_face_detected": 0,
        "zero_face_detected": 0,
        "multiple_faces_detected": 0
    }

    if run_detector_check:
        print("\n[AUDIT] Running Phase 1 dlib face detectability check on split samples...")
        detector = DlibHOGDetector()
        
        # Sample images across all splits up to max_detector_samples
        sample_subset = df_splits.sample(n=min(len(df_splits), max_detector_samples), random_state=42)
        for _, row in sample_subset.iterrows():
            filepath = os.path.join(PROJECT_ROOT, row["relative_path"])
            try:
                with Image.open(filepath) as img:
                    rgb_img = np.array(img.convert("RGB"))
                locs = detector.detect(rgb_img)
                detector_stats["images_checked"] += 1
                if len(locs) == 1:
                    detector_stats["single_face_detected"] += 1
                elif len(locs) == 0:
                    detector_stats["zero_face_detected"] += 1
                else:
                    detector_stats["multiple_faces_detected"] += 1
            except Exception as e:
                warnings.append(f"Detector failed on {filepath}: {e}")

    # 5. Build Summary Statistics
    summary = {
        "status": "PASSED" if len(errors) == 0 else "FAILED",
        "total_identities": int(len(df_identities)),
        "total_images": int(len(df_splits)),
        "splits_distribution": {
            "enrollment": int(len(df_splits[df_splits["split"] == "enrollment"])),
            "validation": int(len(df_splits[df_splits["split"] == "validation"])),
            "test": int(len(df_splits[df_splits["split"] == "test"]))
        },
        "images_per_identity_min": int(df_identities["total_images"].min()),
        "images_per_identity_max": int(df_identities["total_images"].max()),
        "images_per_identity_mean": round(float(df_identities["total_images"].mean()), 2),
        "resolutions_found": [f"{w}x{h}" for w, h in sorted(dimensions_set)],
        "cross_split_leakage_count": len(leak_enroll_val) + len(leak_enroll_test) + len(leak_val_test),
        "unreadable_images_count": unreadable_count,
        "detector_detectability_audit": detector_stats,
        "errors": errors,
        "warnings": warnings
    }

    # Save summary JSON
    summary_path = os.path.join(meta_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    print(f"Status: {summary['status']}")
    print(f"Identities: {summary['total_identities']}")
    print(f"Total Images: {summary['total_images']}")
    print(f"  - Enrollment: {summary['splits_distribution']['enrollment']}")
    print(f"  - Validation: {summary['splits_distribution']['validation']}")
    print(f"  - Test: {summary['splits_distribution']['test']}")
    print(f"Cross-Split Leakage: {summary['cross_split_leakage_count']}")
    print(f"Unreadable Images: {summary['unreadable_images_count']}")
    print(f"Face Detectability Audit ({detector_stats['images_checked']} images):")
    print(f"  - 1 Face Detected: {detector_stats['single_face_detected']}")
    print(f"  - 0 Faces Detected: {detector_stats['zero_face_detected']}")
    print(f"  - >1 Faces Detected: {detector_stats['multiple_faces_detected']}")
    print(f"Saved dataset summary to: {summary_path}")

    if errors:
        print(f"\n[CRITICAL ERRORS FOUND ({len(errors)})]:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.")
        sys.exit(1)

    print("\n[SUCCESS] Dataset integrity verified with zero leakage!")
    return summary


if __name__ == "__main__":
    run_dataset_validation()
