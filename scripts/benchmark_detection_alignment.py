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
from ml.detector import ModernFaceDetector, DlibHOGDetector, FaceDetection
from ml.aligner import FaceAligner

def benchmark_preprocessing(
    num_samples: int = 300,
    split_name: str = "validation"
) -> Dict[str, Any]:
    """
    Evaluates modern face detection and 5-point alignment over a representative LFW subset.
    Measures detection success rate, alignment success rate, and per-stage latency breakdown.
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

    model_path = config.get("model", {}).get("yunet_model_path", "ml/models/face_detection_yunet_2023mar.onnx")
    score_thresh = config.get("model", {}).get("score_threshold", 0.6)
    nms_thresh = config.get("model", {}).get("nms_threshold", 0.3)
    aligned_size = tuple(config.get("model", {}).get("aligned_face_size", [112, 112]))
    multi_face_policy = config.get("model", {}).get("multi_face_policy", "highest_confidence")

    print("="*60)
    print("PHASE 3: MODERN DETECTION & 5-POINT ALIGNMENT BENCHMARK")
    print("="*60)
    print(f"Detector: OpenCV YuNet (ONNX)")
    print(f"Model Path: {model_path}")
    print(f"Score Threshold: {score_thresh} | NMS: {nms_thresh}")
    print(f"Aligned Crop Target: {aligned_size}")
    print(f"Multi-face Policy: {multi_face_policy}")
    print(f"Samples Evaluated: {len(subset)} ({split_name} split)\n")

    detector = ModernFaceDetector(
        model_path=model_path,
        score_threshold=score_thresh,
        nms_threshold=nms_thresh
    )
    aligner = FaceAligner(output_size=aligned_size)

    # Warm up detector
    dummy = np.zeros((250, 250, 3), dtype=np.uint8)
    detector.detect_faces(dummy)

    total_images = len(subset)
    detected_count = 0
    failed_detection_count = 0
    multi_face_count = 0
    aligned_count = 0
    failed_alignment_count = 0

    detection_latencies = []
    alignment_latencies = []
    total_latencies = []

    for _, row in subset.iterrows():
        img_path = os.path.join(PROJECT_ROOT, row["relative_path"])
        with Image.open(img_path) as img:
            rgb_img = np.array(img.convert("RGB"))

        t0 = time.perf_counter()
        detections = detector.detect_faces(rgb_img)
        t1 = time.perf_counter()

        det_time_ms = (t1 - t0) * 1000.0
        detection_latencies.append(det_time_ms)

        if not detections:
            failed_detection_count += 1
            total_latencies.append(det_time_ms)
            continue

        detected_count += 1
        if len(detections) > 1:
            multi_face_count += 1

        # Apply multi-face selection policy
        if multi_face_policy == "highest_confidence":
            primary_face = max(detections, key=lambda d: d.confidence)
        else:
            primary_face = detections[0]

        # 5-point facial landmark alignment
        t2 = time.perf_counter()
        try:
            if primary_face.landmarks is not None:
                aligned_face = aligner.align(rgb_img, primary_face.landmarks)
                t3 = time.perf_counter()
                align_time_ms = (t3 - t2) * 1000.0
                alignment_latencies.append(align_time_ms)

                if aligned_face.shape == (aligned_size[1], aligned_size[0], 3):
                    aligned_count += 1
                else:
                    failed_alignment_count += 1
            else:
                failed_alignment_count += 1
                align_time_ms = 0.0
        except Exception as e:
            failed_alignment_count += 1
            align_time_ms = 0.0

        total_latencies.append(det_time_ms + align_time_ms)

    det_rate = (detected_count / total_images) * 100.0 if total_images > 0 else 0.0
    align_rate = (aligned_count / detected_count) * 100.0 if detected_count > 0 else 0.0

    avg_det_latency = float(np.mean(detection_latencies))
    avg_align_latency = float(np.mean(alignment_latencies)) if alignment_latencies else 0.0
    avg_total_latency = float(np.mean(total_latencies))

    results = {
        "total_images_processed": total_images,
        "successful_detections": detected_count,
        "failed_detections": failed_detection_count,
        "multiple_face_cases": multi_face_count,
        "successful_alignments": aligned_count,
        "failed_alignments": failed_alignment_count,
        "detection_rate_pct": round(det_rate, 2),
        "alignment_success_rate_pct": round(align_rate, 2),
        "avg_detection_latency_ms": round(avg_det_latency, 3),
        "avg_alignment_latency_ms": round(avg_align_latency, 3),
        "avg_total_preprocessing_latency_ms": round(avg_total_latency, 3)
    }

    print("="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(f"Total Processed: {total_images}")
    print(f"Successful Detections: {detected_count} ({det_rate:.2f}%)")
    print(f"Failed Detections: {failed_detection_count}")
    print(f"Multi-face Images: {multi_face_count}")
    print(f"Successful Alignments: {aligned_count} ({align_rate:.2f}%)")
    print(f"Failed Alignments: {failed_alignment_count}")
    print("\nLatency Breakdown (CPU Inference):")
    print(f"  - Average Detection Latency: {avg_det_latency:.2f} ms")
    print(f"  - Average 5-Point Alignment Latency: {avg_align_latency:.2f} ms")
    print(f"  - Average Total Preprocessing Latency: {avg_total_latency:.2f} ms")
    print(f"  - Preprocessing Throughput: {1000.0 / avg_total_latency:.1f} FPS")

    return results


if __name__ == "__main__":
    benchmark_preprocessing(num_samples=300, split_name="validation")
