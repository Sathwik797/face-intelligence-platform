import os
import time
import json
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageFile
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ml.detector import DlibHOGDetector, ModernFaceDetector
from ml.aligner import FaceAligner
from ml.embedder import DlibEmbedder, ArcFaceEmbedder
from ml.evaluation.metrics import (
    calculate_roc_curve,
    calculate_roc_auc,
    calculate_eer,
    calculate_far_frr,
    find_optimal_threshold,
    calculate_fold_aware_metrics,
    calculate_score_statistics
)

# Global worker handles for multiprocessing
_g_detector = None
_g_aligner = None
_g_embedder = None
_g_exp_id = None

def _init_worker(exp_id: str, config: Dict[str, Any]):
    global _g_detector, _g_aligner, _g_embedder, _g_exp_id
    _g_exp_id = exp_id
    if exp_id == "E1":
        _g_detector = DlibHOGDetector()
        _g_embedder = DlibEmbedder()
        _g_aligner = None
    elif exp_id == "E2":
        _g_detector = ModernFaceDetector(
            model_path=config.get("model", {}).get("yunet_model_path"),
            score_threshold=config.get("model", {}).get("score_threshold", 0.6),
            nms_threshold=config.get("model", {}).get("nms_threshold", 0.3)
        )
        _g_aligner = FaceAligner(
            output_size=tuple(config.get("model", {}).get("aligned_face_size", [112, 112]))
        )
        _g_embedder = ArcFaceEmbedder(
            model_path=config.get("model", {}).get("arcface_model_path")
        )

def _extract_worker_fn(path: str) -> Tuple[str, Optional[np.ndarray]]:
    global _g_detector, _g_aligner, _g_embedder, _g_exp_id
    if not os.path.exists(path):
        return path, None
    try:
        bgr = cv2.imread(path)
        if bgr is None:
            return path, None
        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return path, None

    if _g_exp_id == "E1":
        try:
            locs = _g_detector.detect(rgb_img)
            if not locs:
                h, w = rgb_img.shape[:2]
                locs = [(0, w, h, 0)]
            embs = _g_embedder.embed(rgb_img, locs)
            if len(embs) > 0:
                return path, embs[0].astype(np.float64)
            return path, None
        except Exception:
            return path, None

    elif _g_exp_id == "E2":
        try:
            faces = _g_detector.detect_faces(rgb_img)
            if not faces or faces[0].landmarks is None:
                return path, None
            primary_face = max(faces, key=lambda d: d.confidence)
            aligned = _g_aligner.align(rgb_img, primary_face.landmarks)
            emb = _g_embedder.embed(aligned)[0].astype(np.float32)
            return path, emb
        except Exception:
            return path, None

    return path, None


class VerificationEvaluator:
    """
    High-Performance Robust Face Verification Evaluator for LFW 10-Fold Benchmark.
    Supports Experiment E1 (dlib baseline) and Experiment E2 (modern ArcFace).
    Performs rigorous 10-fold cross-validated threshold selection (train on 9 folds, test on 1).
    """

    def __init__(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        provisional_threshold: Optional[float] = None,
        num_workers: int = 8
    ):
        self.experiment_id = experiment_id.upper()
        self.config = config
        self.num_workers = min(num_workers, os.cpu_count() or 4)
        self.embedding_cache: Dict[str, Optional[np.ndarray]] = {}

        self.cache_dir = "reports/evaluation/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{self.experiment_id.lower()}_embeddings.npz")

        if self.experiment_id == "E1":
            self.score_direction = "distance"
            self.provisional_threshold = provisional_threshold if provisional_threshold is not None else 0.6
            self.model_desc = "dlib_hog + dlib_128d + euclidean"
        elif self.experiment_id == "E2":
            self.score_direction = "similarity"
            self.provisional_threshold = provisional_threshold if provisional_threshold is not None else 0.45
            self.model_desc = "yunet + 5pt_align + arcface_512d + cosine"
        else:
            raise ValueError(f"Unsupported experiment_id: '{experiment_id}'. Must be 'E1' or 'E2'.")

        self._load_disk_cache()

    def _load_disk_cache(self):
        """Loads precomputed embeddings from disk cache if available."""
        if os.path.exists(self.cache_file):
            try:
                data = np.load(self.cache_file, allow_pickle=True)
                keys = list(data["keys"])
                embs = data["embeddings"]
                valid_mask = data["valid_mask"]
                for k, emb, is_valid in zip(keys, embs, valid_mask):
                    self.embedding_cache[str(k)] = emb if is_valid else None
                print(f"[{self.experiment_id}] Loaded {len(self.embedding_cache)} embeddings from cache: {self.cache_file}", flush=True)
            except Exception as e:
                print(f"[{self.experiment_id}] Cache load error: {e}. Recomputing.", flush=True)

    def _save_disk_cache(self):
        """Saves memory embedding cache to disk archive."""
        keys = list(self.embedding_cache.keys())
        dim = 128 if self.experiment_id == "E1" else 512
        embs = np.zeros((len(keys), dim), dtype=np.float32)
        valid_mask = np.zeros(len(keys), dtype=bool)

        for i, k in enumerate(keys):
            val = self.embedding_cache[k]
            if val is not None:
                embs[i] = val
                valid_mask[i] = True

        np.savez_compressed(
            self.cache_file,
            keys=np.array(keys, dtype=object),
            embeddings=embs,
            valid_mask=valid_mask
        )
        print(f"[{self.experiment_id}] Saved {len(keys)} embeddings to disk cache: {self.cache_file}", flush=True)

    def precompute_embeddings(self, image_paths: List[str]):
        """Pre-extracts embeddings using multi-core multiprocessing pool."""
        unique_paths = sorted(list(set(image_paths)))
        missing_paths = [p for p in unique_paths if p not in self.embedding_cache]
        total_missing = len(missing_paths)

        if total_missing == 0:
            print(f"[{self.experiment_id}] All {len(unique_paths)} unique image embeddings are loaded from cache.", flush=True)
            return

        print(f"[{self.experiment_id}] Extracting embeddings for {total_missing} images using {self.num_workers} parallel worker processes...", flush=True)
        t0 = time.perf_counter()

        with mp.Pool(processes=self.num_workers, initializer=_init_worker, initargs=(self.experiment_id, self.config)) as pool:
            results = pool.map(_extract_worker_fn, missing_paths, chunksize=50)

        for p, emb in results:
            self.embedding_cache[p] = emb

        elapsed = time.perf_counter() - t0
        speed = total_missing / elapsed if elapsed > 0 else 0.0
        print(f"[{self.experiment_id}] Extracted {total_missing} images in {elapsed:.2f} s ({speed:.1f} imgs/s)", flush=True)

        self._save_disk_cache()

    def evaluate_pairs(
        self,
        pairs_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes verification evaluation over a dataframe of pair records.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (pair_results_df, summary_dict)
        """
        t0_total = time.perf_counter()

        # Step 1: Precompute embeddings (or load from disk cache)
        all_imgs = list(pairs_df["image1_rel_path"].values) + list(pairs_df["image2_rel_path"].values)
        self.precompute_embeddings(all_imgs)

        # Step 2: Vectorized Pairwise Scoring
        print(f"[{self.experiment_id}] Scoring {len(pairs_df)} verification pairs...", flush=True)
        results = []
        for idx, row in pairs_df.iterrows():
            p1 = str(row["image1_rel_path"])
            p2 = str(row["image2_rel_path"])

            emb1 = self.embedding_cache.get(p1)
            emb2 = self.embedding_cache.get(p2)

            pair_status = "success"
            raw_score = None

            if emb1 is not None and emb2 is not None:
                if self.score_direction == "distance":
                    raw_score = float(np.linalg.norm(emb1 - emb2))
                else:
                    raw_score = float(np.dot(emb1, emb2))
            else:
                pair_status = "failed_embedding"

            results.append({
                "experiment_id": self.experiment_id,
                "pair_id": int(row.get("pair_id", idx + 1)),
                "fold": int(row["fold"]),
                "identity1": str(row.get("identity1", "")),
                "identity2": str(row.get("identity2", "")),
                "image1_rel_path": p1,
                "image2_rel_path": p2,
                "is_same": int(row["is_same"]),
                "raw_score": raw_score,
                "score_direction": self.score_direction,
                "status": pair_status
            })

        t1_total = time.perf_counter()
        total_eval_time_s = t1_total - t0_total

        results_df = pd.DataFrame(results)
        valid_mask = results_df["raw_score"].notnull()
        valid_df = results_df[valid_mask]
        failed_count = len(results_df) - len(valid_df)

        # Step 3: Official 10-Fold Cross-Validated Threshold Selection
        fold_aware_summary = calculate_fold_aware_metrics(valid_df, self.score_direction)

        # Step 4: Global Distribution & Discrimination Metrics
        all_labels = valid_df["is_same"].values
        all_scores = valid_df["raw_score"].values
        global_fpr, global_tpr, global_thresh = calculate_roc_curve(all_labels, all_scores, self.score_direction)
        global_auc = calculate_roc_auc(global_fpr, global_tpr)
        global_eer, global_eer_thresh = calculate_eer(global_fpr, global_tpr, global_thresh)
        global_score_stats = calculate_score_statistics(all_labels, all_scores)

        # Provisional threshold reference metrics
        prov_acc, prov_far, prov_frr = calculate_far_frr(
            all_labels, all_scores, self.provisional_threshold, self.score_direction
        )

        summary = {
            "experiment_id": self.experiment_id,
            "model_description": self.model_desc,
            "score_direction": self.score_direction,
            "total_pairs": len(results_df),
            "successful_pairs": len(valid_df),
            "failed_pairs": failed_count,
            "total_evaluation_time_seconds": round(total_eval_time_s, 2),
            "avg_pair_latency_ms": round((total_eval_time_s * 1000.0) / len(results_df), 2),
            "discrimination_metrics": {
                "global_roc_auc": round(global_auc, 4),
                "fold_roc_auc_mean": fold_aware_summary["roc_auc_mean"],
                "fold_roc_auc_std": fold_aware_summary["roc_auc_std"],
                "global_eer": round(global_eer, 4),
                "global_eer_threshold": round(global_eer_thresh, 4),
                "fold_eer_mean": fold_aware_summary["eer_mean"],
                "fold_eer_std": fold_aware_summary["eer_std"],
                "score_distribution": global_score_stats
            },
            "fold_calibrated_metrics": {
                "accuracy_mean": fold_aware_summary["calibrated_accuracy_mean"],
                "accuracy_std": fold_aware_summary["calibrated_accuracy_std"],
                "far_mean": fold_aware_summary["calibrated_far_mean"],
                "far_std": fold_aware_summary["calibrated_far_std"],
                "frr_mean": fold_aware_summary["calibrated_frr_mean"],
                "frr_std": fold_aware_summary["calibrated_frr_std"],
                "threshold_mean": fold_aware_summary["calibrated_threshold_mean"],
                "threshold_std": fold_aware_summary["calibrated_threshold_std"]
            },
            "provisional_threshold_reference": {
                "provisional_threshold": self.provisional_threshold,
                "accuracy": round(prov_acc, 4),
                "far": round(prov_far, 4),
                "frr": round(prov_frr, 4)
            },
            "folds": fold_aware_summary["folds"]
        }

        return results_df, summary
