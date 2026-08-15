import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from ml.detector import ModernFaceDetector
from ml.aligner import FaceAligner
from ml.embedder import ArcFaceEmbedder
from ml.gallery import IdentityGallery

def build_enrollment_gallery(
    output_path: Optional[str] = None
) -> IdentityGallery:
    """
    Constructs the modern IdentityGallery using ONLY the Phase 2 enrollment split.
    Guarantees that no validation or test images are used to build the gallery.
    """
    config = load_config("config/config.yaml")
    meta_dir = config.get("paths", {}).get("metadata_dir", "data/metadata")
    splits_csv = os.path.join(meta_dir, "splits.csv")

    if not os.path.exists(splits_csv):
        raise FileNotFoundError("Metadata splits.csv missing! Run scripts/prepare_dataset.py first.")

    target_gallery_path = output_path or config.get("paths", {}).get(
        "gallery_path", "data/embeddings/arcface_gallery.npz"
    )

    df_splits = pd.read_csv(splits_csv)
    enrollment_subset = df_splits[df_splits["split"] == "enrollment"]

    print("="*60)
    print("PHASE 5: BUILDING ENROLLED IDENTITY GALLERY (EXPERIMENT E2)")
    print("="*60)
    print(f"Enrollment Images: {len(enrollment_subset)}")
    print(f"Unique Identities: {enrollment_subset['identity'].nunique()}")
    print(f"Output Gallery Path: {target_gallery_path}\n")

    t_start = time.perf_counter()

    detector = ModernFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()

    gallery = IdentityGallery()

    identities_grouped = enrollment_subset.groupby("identity")
    skipped_images = 0
    enrolled_templates = 0

    for identity, group in identities_grouped:
        id_crops = []
        id_paths = []

        for _, row in group.iterrows():
            img_path = os.path.join(PROJECT_ROOT, row["relative_path"])
            with Image.open(img_path) as img:
                rgb_img = np.array(img.convert("RGB"))

            faces = detector.detect_faces(rgb_img)
            if not faces:
                print(f"[WARNING] No face detected in reference image: {img_path}. Skipping.")
                skipped_images += 1
                continue

            primary_face = max(faces, key=lambda d: d.confidence)
            if primary_face.landmarks is not None:
                aligned_face = aligner.align(rgb_img, primary_face.landmarks)
                id_crops.append(aligned_face)
                id_paths.append(row["relative_path"])

        if id_crops:
            embeddings = embedder.embed_batch(id_crops)
            gallery.add_templates(
                identity=identity,
                embeddings=embeddings,
                source_paths=id_paths
            )
            enrolled_templates += len(id_crops)

    t_end = time.perf_counter()
    build_time_ms = (t_end - t_start) * 1000.0

    # Validate gallery integrity
    val_report = gallery.validate()
    print("="*60)
    print("GALLERY BUILD SUMMARY")
    print("="*60)
    print(f"Build Time: {build_time_ms:.2f} ms ({build_time_ms / 1000.0:.2f} s)")
    print(f"Enrolled Unique Identities: {val_report['unique_identities']}")
    print(f"Total Enrolled Templates: {val_report['total_templates']}")
    print(f"Embedding Dimension: {val_report['embedding_dim']}D")
    print(f"Validation Status: {'PASSED' if val_report['valid'] else 'FAILED'}")
    if not val_report["valid"]:
        print(f"Validation Errors: {val_report['errors']}")
        raise ValueError(f"Gallery validation failed: {val_report['errors']}")

    # Serialize gallery to disk
    gallery.save(target_gallery_path)
    print(f"Saved verified gallery artifact to: {target_gallery_path}")

    return gallery


if __name__ == "__main__":
    build_enrollment_gallery()
