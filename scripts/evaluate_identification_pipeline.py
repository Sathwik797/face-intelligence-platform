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
from ml.gallery import IdentityGallery
from ml.pipeline import ModernRecognitionPipeline, ModernRecognitionResult

def run_identification_pipeline_verification(
    num_known_queries: int = 50,
    num_unknown_queries: int = 50
) -> Dict[str, Any]:
    """
    Validates the end-to-end ModernRecognitionPipeline (Experiment E2).
    Evaluates both Known Identity queries and Unknown/Open-Set queries against the provisional threshold.
    Measures component-level latency breakdown.
    """
    config = load_config("config/config.yaml")
    meta_dir = config.get("paths", {}).get("metadata_dir", "data/metadata")
    splits_csv = os.path.join(meta_dir, "splits.csv")
    gallery_path = config.get("paths", {}).get("gallery_path", "data/embeddings/arcface_gallery.npz")
    threshold = config.get("model", {}).get("cosine_threshold", 0.45)

    print("="*65)
    print("PHASE 5: MODERN RECOGNITION PIPELINE VERIFICATION (EXPERIMENT E2)")
    print("="*65)
    print(f"Gallery Artifact: {gallery_path}")
    print(f"Provisional Cosine Threshold: {threshold} (Will be calibrated in Phase 7)")
    print(f"Multi-face Policy: {config.get('model', {}).get('multi_face_policy', 'highest_confidence')}\n")

    # 1. Gallery Load Time
    t_load_0 = time.perf_counter()
    pipeline = ModernRecognitionPipeline.from_config(config, gallery_path=gallery_path)
    t_load_1 = time.perf_counter()
    gallery_load_ms = (t_load_1 - t_load_0) * 1000.0

    val_report = pipeline.gallery.validate()
    print(f"[INIT] Gallery loaded in {gallery_load_ms:.2f} ms")
    print(f"[INIT] Gallery Enrolled Identities: {val_report['unique_identities']} ({val_report['total_templates']} templates)\n")

    df_splits = pd.read_csv(splits_csv)

    # 2. Prepare Known Queries (from validation split of enrolled identities)
    known_pool = df_splits[df_splits["split"] == "validation"]
    if len(known_pool) > num_known_queries:
        known_subset = known_pool.sample(n=num_known_queries, random_state=42)
    else:
        known_subset = known_pool

    # 3. Prepare Unknown Queries (identities not in enrolled 59 identities)
    enrolled_names = set(df_splits["identity"].unique())
    raw_lfw_root = config.get("paths", {}).get("raw_lfw_dir", "data/raw/lfw")
    lfw_funneled = os.path.join(raw_lfw_root, "lfw_home", "lfw_funneled")

    unknown_paths = []
    if os.path.exists(lfw_funneled):
        for folder in os.listdir(lfw_funneled):
            if folder not in enrolled_names and os.path.isdir(os.path.join(lfw_funneled, folder)):
                imgs = [f for f in os.listdir(os.path.join(lfw_funneled, folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    unknown_paths.append((folder, os.path.join(lfw_funneled, folder, imgs[0])))
                    if len(unknown_paths) >= num_unknown_queries:
                        break

    print(f"[EVALUATION] Evaluating {len(known_subset)} Known queries and {len(unknown_paths)} Unknown queries...")

    # Profiling containers
    detection_latencies = []
    alignment_latencies = []
    embedding_latencies = []
    search_latencies = []
    total_latencies = []

    known_results = []
    for _, row in known_subset.iterrows():
        img_path = os.path.join(PROJECT_ROOT, row["relative_path"])
        with Image.open(img_path) as img:
            rgb_img = np.array(img.convert("RGB"))

        # Component breakdown profiling
        t0 = time.perf_counter()
        faces = pipeline.detector.detect_faces(rgb_img)
        t1 = time.perf_counter()

        if faces and faces[0].landmarks is not None:
            primary_face = max(faces, key=lambda d: d.confidence)
            t_al0 = time.perf_counter()
            aligned = pipeline.aligner.align(rgb_img, primary_face.landmarks)
            t_al1 = time.perf_counter()

            t_em0 = time.perf_counter()
            emb = pipeline.embedder.embed(aligned)
            t_em1 = time.perf_counter()

            t_se0 = time.perf_counter()
            rec_id, best_cand, best_sim, is_rec = pipeline.gallery.search(emb, threshold=threshold)
            t_se1 = time.perf_counter()

            detection_latencies.append((t1 - t0) * 1000.0)
            alignment_latencies.append((t_al1 - t_al0) * 1000.0)
            embedding_latencies.append((t_em1 - t_em0) * 1000.0)
            search_latencies.append((t_se1 - t_se0) * 1000.0)
            total_latencies.append((t_se1 - t0) * 1000.0)

            known_results.append({
                "true_identity": row["identity"],
                "predicted_identity": rec_id,
                "best_candidate": best_cand,
                "similarity": best_sim,
                "recognized": is_rec,
                "correct_candidate": (best_cand == row["identity"])
            })

    unknown_results = []
    for true_unknown_id, img_path in unknown_paths:
        with Image.open(img_path) as img:
            rgb_img = np.array(img.convert("RGB"))

        res: ModernRecognitionResult = pipeline.recognize(rgb_img)
        unknown_results.append({
            "true_identity": f"UNKNOWN ({true_unknown_id})",
            "predicted_identity": res.identity,
            "best_candidate": res.best_candidate,
            "similarity": res.similarity,
            "recognized": res.recognized,
            "reason": res.reason
        })

    # Summary calculations
    known_correct_candidate = sum(1 for r in known_results if r["correct_candidate"])
    known_accepted = sum(1 for r in known_results if r["recognized"])
    known_avg_sim = float(np.mean([r["similarity"] for r in known_results]))

    unknown_rejected = sum(1 for r in unknown_results if not r["recognized"])
    unknown_avg_sim = float(np.mean([r["similarity"] for r in unknown_results])) if unknown_results else 0.0

    print("\n" + "="*65)
    print("RECOGNITION PIPELINE VERIFICATION RESULTS")
    print("="*65)
    print(f"Known Queries Evaluated: {len(known_results)}")
    print(f"  - Correct Top-1 Candidate Identity: {known_correct_candidate} / {len(known_results)} ({known_correct_candidate / len(known_results) * 100:.2f}%)")
    print(f"  - Accepted at Provisional Threshold ({threshold}): {known_accepted} / {len(known_results)} ({known_accepted / len(known_results) * 100:.2f}%)")
    print(f"  - Average Known Similarity Score: {known_avg_sim:.4f}")

    print(f"\nUnknown Queries Evaluated: {len(unknown_results)}")
    print(f"  - Successfully Rejected as Unknown: {unknown_rejected} / {len(unknown_results)} ({unknown_rejected / len(unknown_results) * 100:.2f}%)")
    print(f"  - Average Unknown Similarity Score: {unknown_avg_sim:.4f}")
    print(f"  - Score Margin Separation: {known_avg_sim - unknown_avg_sim:+.4f}")

    print("\n" + "="*65)
    print("LATENCY BREAKDOWN PER QUERY (CPU INFERENCE)")
    print("="*65)
    print(f"1. Face Detection (YuNet):         {np.mean(detection_latencies):6.2f} ms")
    print(f"2. 5-Point Affine Alignment:       {np.mean(alignment_latencies):6.2f} ms")
    print(f"3. ArcFace Embedding (ResNet-50):  {np.mean(embedding_latencies):6.2f} ms")
    print(f"4. Exact Gallery Search (118 vecs):{np.mean(search_latencies):6.2f} ms")
    print(f"--------------------------------------------------")
    print(f"TOTAL END-TO-END RECOGNITION:      {np.mean(total_latencies):6.2f} ms ({1000.0 / np.mean(total_latencies):.1f} FPS)")

    return {
        "known_queries_evaluated": len(known_results),
        "known_correct_candidate_pct": round(known_correct_candidate / len(known_results) * 100, 2),
        "known_accepted_pct": round(known_accepted / len(known_results) * 100, 2),
        "known_avg_similarity": round(known_avg_sim, 4),
        "unknown_queries_evaluated": len(unknown_results),
        "unknown_rejected_pct": round(unknown_rejected / len(unknown_results) * 100, 2),
        "unknown_avg_similarity": round(unknown_avg_sim, 4),
        "score_margin": round(known_avg_sim - unknown_avg_sim, 4),
        "latency_breakdown_ms": {
            "detection": round(float(np.mean(detection_latencies)), 2),
            "alignment": round(float(np.mean(alignment_latencies)), 2),
            "embedding": round(float(np.mean(embedding_latencies)), 2),
            "gallery_search": round(float(np.mean(search_latencies)), 2),
            "total": round(float(np.mean(total_latencies)), 2)
        }
    }


if __name__ == "__main__":
    run_identification_pipeline_verification(num_known_queries=50, num_unknown_queries=50)
