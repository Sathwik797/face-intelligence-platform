import os
import sys
import time
from typing import Dict, Any, List
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

def run_embedding_benchmark(
    num_samples: int = 300,
    split_name: str = "validation"
) -> Dict[str, Any]:
    """
    Executes the modern ArcFace embedding pipeline over a representative LFW subset.
    Performs rigorous embedding sanity checks and benchmarks CPU latency.
    """
    config = load_config("config/config.yaml")
    meta_dir = config.get("paths", {}).get("metadata_dir", "data/metadata")
    splits_csv = os.path.join(meta_dir, "splits.csv")

    if not os.path.exists(splits_csv):
        raise FileNotFoundError("Metadata splits.csv missing! Run scripts/prepare_dataset.py first.")

    df_splits = pd.read_csv(splits_csv)
    subset = df_splits[df_splits["split"] == split_name]

    if len(subset) > num_samples:
        subset = subset.sample(n=num_samples, random_state=42)

    arcface_model_path = config.get("model", {}).get("arcface_model_path", "ml/models/arcface_w600k_r50.onnx")

    print("="*60)
    print("PHASE 4: PRETRAINED ARCFACE EMBEDDING BENCHMARK")
    print("="*60)
    print(f"Embedding Backbone: ResNet-50 (w600k_r50 ONNX)")
    print(f"Model Path: {arcface_model_path}")
    print(f"Target Dimensionality: 512D (L2 Normalized)")
    print(f"Samples Evaluated: {len(subset)} ({split_name} split)\n")

    # 1. Model Loading Latency
    t_load_0 = time.perf_counter()
    embedder = ArcFaceEmbedder(model_path=arcface_model_path)
    t_load_1 = time.perf_counter()
    load_time_ms = (t_load_1 - t_load_0) * 1000.0
    print(f"[INIT] Model loaded in {load_time_ms:.2f} ms")

    detector = ModernFaceDetector()
    aligner = FaceAligner()

    # Pre-extract aligned face crops
    aligned_crops: List[np.ndarray] = []
    skipped_no_detection = 0

    print("[INFO] Preprocessing face crops with YuNet + 5-point alignment...")
    for _, row in subset.iterrows():
        img_path = os.path.join(PROJECT_ROOT, row["relative_path"])
        with Image.open(img_path) as img:
            rgb_img = np.array(img.convert("RGB"))

        faces = detector.detect_faces(rgb_img)
        if not faces:
            skipped_no_detection += 1
            continue

        primary_face = max(faces, key=lambda d: d.confidence)
        if primary_face.landmarks is not None:
            aligned_face = aligner.align(rgb_img, primary_face.landmarks)
            aligned_crops.append(aligned_face)

    total_crops = len(aligned_crops)
    print(f"[INFO] Aligned crops ready: {total_crops} / {len(subset)} (Skipped {skipped_no_detection} undetected)\n")

    # 2. Embedding Extraction & Latency Profiling
    single_latencies = []
    embeddings_list = []

    print("[BENCHMARK] Running single-crop embedding extraction...")
    for crop in aligned_crops:
        t0 = time.perf_counter()
        emb = embedder.embed(crop)
        t1 = time.perf_counter()

        single_latencies.append((t1 - t0) * 1000.0)
        embeddings_list.append(emb)

    embeddings_matrix = np.vstack(embeddings_list) if embeddings_list else np.empty((0, 512))

    # 3. Batch Embedding Latency (Batch size 16)
    batch_latencies_per_item = []
    batch_size = 16
    if total_crops >= batch_size:
        print(f"[BENCHMARK] Running batch embedding extraction (batch size = {batch_size})...")
        for i in range(0, total_crops - batch_size + 1, batch_size):
            batch_slice = aligned_crops[i:i + batch_size]
            t0 = time.perf_counter()
            batch_embs = embedder.embed_batch(batch_slice)
            t1 = time.perf_counter()
            batch_latencies_per_item.append(((t1 - t0) * 1000.0) / batch_size)

    # 4. Rigorous Embedding Sanity Checks
    print("\n" + "="*60)
    print("EMBEDDING SANITY CHECKS")
    print("="*60)

    # Check 1: Output Dimension
    check_dim = (embeddings_matrix.shape[1] == 512)
    print(f"  [1] Output Dimension == 512: {check_dim} ({embeddings_matrix.shape[1]}D)")

    # Check 2: Finite Values (no NaN/Inf)
    check_finite = bool(np.all(np.isfinite(embeddings_matrix)))
    print(f"  [2] Finite Values (No NaN/Inf): {check_finite}")

    # Check 3: Unit L2 Norm
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))
    check_unit_norm = bool(np.allclose(norms, 1.0, atol=1e-4))
    print(f"  [3] Unit L2 Norm: {check_unit_norm} (Min: {norm_min:.6f}, Max: {norm_max:.6f})")

    # Check 4: Determinism on Identical Inputs
    emb_a = embedder.embed(aligned_crops[0])
    emb_b = embedder.embed(aligned_crops[0])
    check_deterministic = bool(np.allclose(emb_a, emb_b, atol=1e-6))
    print(f"  [4] Deterministic on Identical Inputs: {check_deterministic}")

    # Check 5: Discriminability on Distinct Faces
    if total_crops >= 2:
        sim_distinct = float(np.dot(embeddings_matrix[0], embeddings_matrix[1]))
        check_discriminable = (sim_distinct < 0.99)
        print(f"  [5] Distinct Embeddings for Different Images: {check_discriminable} (Cosine Sim: {sim_distinct:.4f})")
    else:
        check_discriminable = True

    # Check 6: Batch vs Single Consistency
    if total_crops >= 4:
        sub_batch = aligned_crops[:4]
        b_embs = embedder.embed_batch(sub_batch)
        s_embs = np.vstack([embedder.embed(c) for c in sub_batch])
        check_batch_consistency = bool(np.allclose(b_embs, s_embs, atol=1e-5))
        print(f"  [6] Single vs Batch Inference Consistency: {check_batch_consistency}")
    else:
        check_batch_consistency = True

    avg_single_latency = float(np.mean(single_latencies))
    min_single_latency = float(np.min(single_latencies))
    max_single_latency = float(np.max(single_latencies))
    avg_batch_latency = float(np.mean(batch_latencies_per_item)) if batch_latencies_per_item else avg_single_latency

    results = {
        "model_name": embedder.model_name,
        "embedding_dim": embedder.embedding_dim,
        "model_loading_time_ms": round(load_time_ms, 2),
        "total_images_processed": len(subset),
        "successful_embeddings": total_crops,
        "avg_single_inference_latency_ms": round(avg_single_latency, 2),
        "min_single_inference_latency_ms": round(min_single_latency, 2),
        "max_single_inference_latency_ms": round(max_single_latency, 2),
        "avg_batch_latency_per_item_ms": round(avg_batch_latency, 2),
        "checks_passed": {
            "dimension_512": check_dim,
            "finite_values": check_finite,
            "unit_l2_norm": check_unit_norm,
            "deterministic_inference": check_deterministic,
            "distinct_embeddings": check_discriminable,
            "batch_single_consistency": check_batch_consistency
        }
    }

    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    print(f"Model Load Time: {load_time_ms:.2f} ms")
    print(f"Single-Image Latency (CPU): {avg_single_latency:.2f} ms (Min: {min_single_latency:.2f} ms, Max: {max_single_latency:.2f} ms)")
    print(f"Batch Latency per Face (CPU): {avg_batch_latency:.2f} ms")
    print(f"Single-Image Embedding Throughput: {1000.0 / avg_single_latency:.1f} FPS")

    return results


if __name__ == "__main__":
    run_embedding_benchmark(num_samples=300, split_name="validation")
