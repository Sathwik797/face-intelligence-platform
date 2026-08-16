import os
import time
import json
import itertools
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageFile
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ml.detector import ModernFaceDetector
from ml.aligner import FaceAligner
from ml.embedder import ArcFaceEmbedder
from ml.evaluation.metrics import calculate_roc_curve, calculate_roc_auc, calculate_eer

# Global worker handles for multi-process feature extraction
_g_detector = None
_g_aligner = None
_g_embedder = None

def _init_calib_worker(config: Dict[str, Any]):
    global _g_detector, _g_aligner, _g_embedder
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

def _extract_calib_worker_fn(path: str) -> Tuple[str, Optional[np.ndarray]]:
    global _g_detector, _g_aligner, _g_embedder
    if not os.path.exists(path):
        return path, None
    try:
        bgr = cv2.imread(path)
        if bgr is None:
            return path, None
        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = _g_detector.detect_faces(rgb_img)
        if not faces or faces[0].landmarks is None:
            return path, None
        primary_face = max(faces, key=lambda d: d.confidence)
        aligned = _g_aligner.align(rgb_img, primary_face.landmarks)
        emb = _g_embedder.embed(aligned)[0].astype(np.float32)
        return path, emb
    except Exception:
        return path, None


class ThresholdCalibrator:
    """
    Production Threshold Calibrator for ArcFace Recognition Pipeline (E2).
    Operates strictly on the project's independent validation split to select,
    evaluate, and recommend production recognition thresholds.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_workers: int = 8
    ):
        self.config = config
        self.num_workers = min(num_workers, os.cpu_count() or 4)
        self.cache_dir = "reports/calibration/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "validation_embeddings.npz")
        self.embedding_cache: Dict[str, Optional[np.ndarray]] = {}
        self._load_disk_cache()

    def _load_disk_cache(self):
        """Loads cached validation embeddings if available."""
        if os.path.exists(self.cache_file):
            try:
                data = np.load(self.cache_file, allow_pickle=True)
                keys = list(data["keys"])
                embs = data["embeddings"]
                valid_mask = data["valid_mask"]
                for k, emb, is_valid in zip(keys, embs, valid_mask):
                    self.embedding_cache[str(k)] = emb if is_valid else None
                print(f"[CALIBRATOR] Loaded {len(self.embedding_cache)} validation embeddings from cache.", flush=True)
            except Exception as e:
                print(f"[CALIBRATOR] Cache load error: {e}. Recomputing.", flush=True)

    def _save_disk_cache(self):
        """Persists validation embeddings to disk cache."""
        keys = list(self.embedding_cache.keys())
        embs = np.zeros((len(keys), 512), dtype=np.float32)
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
        print(f"[CALIBRATOR] Saved {len(keys)} validation embeddings to cache: {self.cache_file}", flush=True)

    def extract_validation_embeddings(self, image_paths: List[str]):
        """Extracts 512D ArcFace embeddings for validation images in parallel."""
        unique_paths = sorted(list(set(image_paths)))
        missing_paths = [p for p in unique_paths if p not in self.embedding_cache]
        total_missing = len(missing_paths)

        if total_missing == 0:
            print(f"[CALIBRATOR] All {len(unique_paths)} validation images are cached in memory.", flush=True)
            return

        print(f"[CALIBRATOR] Extracting ArcFace embeddings for {total_missing} validation images using {self.num_workers} processes...", flush=True)
        t0 = time.perf_counter()

        with mp.Pool(processes=self.num_workers, initializer=_init_calib_worker, initargs=(self.config,)) as pool:
            results = pool.map(_extract_calib_worker_fn, missing_paths, chunksize=30)

        for p, emb in results:
            self.embedding_cache[p] = emb

        elapsed = time.perf_counter() - t0
        speed = total_missing / elapsed if elapsed > 0 else 0.0
        print(f"[CALIBRATOR] Extracted {total_missing} validation embeddings in {elapsed:.2f} s ({speed:.1f} imgs/s)", flush=True)
        self._save_disk_cache()

    def generate_validation_pairs(
        self,
        splits_df: pd.DataFrame,
        max_genuine_per_identity: int = 150,
        max_impostor_pairs: int = 50000,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Constructs validation genuine and impostor pairs strictly from the validation split.

        Args:
            splits_df (pd.DataFrame): Splits metadata dataframe containing 'split', 'identity', 'relative_path'.
            max_genuine_per_identity (int): Upper bound of genuine pairs per identity to avoid domination.
            max_impostor_pairs (int): Sampled impostor pairs.
            random_seed (int): Reproducibility seed.

        Returns:
            Tuple[np.ndarray, np.ndarray, pd.DataFrame]: (labels, scores, pairs_df)
        """
        # Strict validation filter (ZERO test split access)
        val_df = splits_df[splits_df["split"] == "validation"].copy()
        if len(val_df) == 0:
            raise ValueError("No validation samples found in splits dataframe.")

        # Extract embeddings for all validation images
        val_paths = list(val_df["relative_path"].values)
        self.extract_validation_embeddings(val_paths)

        rng = np.random.RandomState(random_seed)
        identities = sorted(list(val_df["identity"].unique()))
        id_to_paths = {identity: list(grp["relative_path"].values) for identity, grp in val_df.groupby("identity")}

        # 1. Genuine pairs (within-identity)
        genuine_pairs = []
        for identity, paths in id_to_paths.items():
            valid_paths = [p for p in paths if self.embedding_cache.get(p) is not None]
            if len(valid_paths) < 2:
                continue
            all_gen = list(itertools.combinations(valid_paths, 2))
            if len(all_gen) > max_genuine_per_identity:
                chosen_indices = rng.choice(len(all_gen), size=max_genuine_per_identity, replace=False)
                sampled_gen = [all_gen[i] for i in chosen_indices]
            else:
                sampled_gen = all_gen
            for p1, p2 in sampled_gen:
                genuine_pairs.append((identity, identity, p1, p2, 1))

        # 2. Impostor pairs (cross-identity)
        impostor_pairs = []
        identity_list = [id_name for id_name, paths in id_to_paths.items() if any(self.embedding_cache.get(p) is not None for p in paths)]
        num_identities = len(identity_list)

        # Uniform random cross-identity sampling
        while len(impostor_pairs) < max_impostor_pairs:
            idx1, idx2 = rng.choice(num_identities, size=2, replace=False)
            id1, id2 = identity_list[idx1], identity_list[idx2]
            p1_candidates = [p for p in id_to_paths[id1] if self.embedding_cache.get(p) is not None]
            p2_candidates = [p for p in id_to_paths[id2] if self.embedding_cache.get(p) is not None]
            if not p1_candidates or not p2_candidates:
                continue
            p1 = rng.choice(p1_candidates)
            p2 = rng.choice(p2_candidates)
            impostor_pairs.append((id1, id2, p1, p2, 0))

        all_pair_records = genuine_pairs + impostor_pairs
        print(f"[CALIBRATOR] Generated {len(genuine_pairs)} genuine pairs and {len(impostor_pairs)} impostor pairs ({len(all_pair_records)} total).", flush=True)

        pair_data = []
        labels_list = []
        scores_list = []

        for id1, id2, p1, p2, label in all_pair_records:
            emb1 = self.embedding_cache.get(p1)
            emb2 = self.embedding_cache.get(p2)
            if emb1 is not None and emb2 is not None:
                sim = float(np.dot(emb1, emb2))
                pair_data.append({
                    "identity1": id1,
                    "identity2": id2,
                    "image1": p1,
                    "image2": p2,
                    "is_same": label,
                    "cosine_similarity": sim
                })
                labels_list.append(label)
                scores_list.append(sim)

        pairs_df_res = pd.DataFrame(pair_data)
        labels = np.array(labels_list, dtype=int)
        scores = np.array(scores_list, dtype=float)

        return labels, scores, pairs_df_res

    @staticmethod
    def calculate_distribution_percentiles(scores: np.ndarray) -> Dict[str, float]:
        """Calculates detailed distribution statistics including percentiles."""
        if len(scores) == 0:
            return {}
        return {
            "count": int(len(scores)),
            "mean": round(float(np.mean(scores)), 4),
            "median": round(float(np.median(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "p1": round(float(np.percentile(scores, 1)), 4),
            "p5": round(float(np.percentile(scores, 5)), 4),
            "p25": round(float(np.percentile(scores, 25)), 4),
            "p50": round(float(np.percentile(scores, 50)), 4),
            "p75": round(float(np.percentile(scores, 75)), 4),
            "p95": round(float(np.percentile(scores, 95)), 4),
            "p99": round(float(np.percentile(scores, 99)), 4)
        }

    @staticmethod
    def sweep_thresholds(
        labels: np.ndarray,
        scores: np.ndarray,
        threshold_range: Tuple[float, float] = (-0.2, 0.95),
        step: float = 0.001
    ) -> pd.DataFrame:
        """
        Executes a fine-grained threshold sweep calculating confusion matrix, accuracy,
        FAR, FRR, Precision, Recall, F1, and Acceptance Rate for every candidate threshold.
        """
        labels_arr = np.asarray(labels, dtype=int)
        scores_arr = np.asarray(scores, dtype=float)

        thresholds = np.arange(threshold_range[0], threshold_range[1] + step / 2, step)
        total = len(labels_arr)
        num_pos = int(np.sum(labels_arr == 1))
        num_neg = int(np.sum(labels_arr == 0))

        records = []
        for tau in thresholds:
            tau = round(float(tau), 4)
            preds = (scores_arr >= tau).astype(int)

            tp = int(np.sum((preds == 1) & (labels_arr == 1)))
            fp = int(np.sum((preds == 1) & (labels_arr == 0)))
            tn = int(np.sum((preds == 0) & (labels_arr == 0)))
            fn = int(np.sum((preds == 0) & (labels_arr == 1)))

            accuracy = float((tp + tn) / total) if total > 0 else 0.0
            far = float(fp / num_neg) if num_neg > 0 else 0.0
            frr = float(fn / num_pos) if num_pos > 0 else 0.0
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = float(tp / num_pos) if num_pos > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            acceptance_rate = float((tp + fp) / total) if total > 0 else 0.0

            records.append({
                "threshold": tau,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": round(accuracy, 4),
                "far": round(far, 6),
                "frr": round(frr, 6),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "acceptance_rate": round(acceptance_rate, 4)
            })

        return pd.DataFrame(records)

    def evaluate_calibration_strategies(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        sweep_df: pd.DataFrame,
        target_low_far: float = 0.001  # 0.1% FAR target
    ) -> Dict[str, Any]:
        """
        Evaluates the five calibration strategies on validation data:
        1. EER threshold (ROC continuous)
        2. Maximum accuracy threshold (argmax accuracy)
        3. Security-oriented low-FAR threshold (FAR <= target_low_far)
        4. F1-optimal threshold (argmax F1)
        5. FAR/FRR-balanced operating point (discrete min |FAR - FRR|)
        """
        labels_arr = np.asarray(labels, dtype=int)
        scores_arr = np.asarray(scores, dtype=float)

        # 1. Continuous ROC EER
        fpr, tpr, threshs = calculate_roc_curve(labels_arr, scores_arr, score_direction="similarity")
        roc_auc = calculate_roc_auc(fpr, tpr)
        eer_val, eer_thresh = calculate_eer(fpr, tpr, threshs)

        # Helper to extract metrics from sweep_df closest to a threshold
        def _get_metrics_at_thresh(target_th: float) -> Dict[str, Any]:
            idx = int((sweep_df["threshold"] - target_th).abs().argmin())
            row = sweep_df.iloc[idx]
            return {
                "threshold": round(float(row["threshold"]), 4),
                "accuracy": round(float(row["accuracy"]), 4),
                "far": round(float(row["far"]), 6),
                "frr": round(float(row["frr"]), 6),
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
                "f1": round(float(row["f1"]), 4),
                "acceptance_rate": round(float(row["acceptance_rate"]), 4)
            }

        # 2. Maximum Accuracy Strategy
        max_acc_val = sweep_df["accuracy"].max()
        max_acc_candidates = sweep_df[sweep_df["accuracy"] == max_acc_val]
        # Choose median candidate for stability
        max_acc_row = max_acc_candidates.iloc[len(max_acc_candidates) // 2]
        max_acc_metrics = {
            "threshold": round(float(max_acc_row["threshold"]), 4),
            "accuracy": round(float(max_acc_row["accuracy"]), 4),
            "far": round(float(max_acc_row["far"]), 6),
            "frr": round(float(max_acc_row["frr"]), 6),
            "precision": round(float(max_acc_row["precision"]), 4),
            "recall": round(float(max_acc_row["recall"]), 4),
            "f1": round(float(max_acc_row["f1"]), 4),
            "acceptance_rate": round(float(max_acc_row["acceptance_rate"]), 4)
        }

        # 3. Security-Oriented Low-FAR Strategy (FAR <= target_low_far)
        low_far_candidates = sweep_df[sweep_df["far"] <= target_low_far]
        if len(low_far_candidates) > 0:
            # Pick candidate with best recall/accuracy among those meeting FAR target
            best_low_far_row = low_far_candidates.sort_values(by=["recall", "accuracy"], ascending=False).iloc[0]
            low_far_metrics = {
                "threshold": round(float(best_low_far_row["threshold"]), 4),
                "target_far": target_low_far,
                "achieved_far": round(float(best_low_far_row["far"]), 6),
                "accuracy": round(float(best_low_far_row["accuracy"]), 4),
                "frr": round(float(best_low_far_row["frr"]), 6),
                "precision": round(float(best_low_far_row["precision"]), 4),
                "recall": round(float(best_low_far_row["recall"]), 4),
                "f1": round(float(best_low_far_row["f1"]), 4),
                "acceptance_rate": round(float(best_low_far_row["acceptance_rate"]), 4),
                "supported": True
            }
        else:
            # Fallback if no point meets strict target
            lowest_far_row = sweep_df.sort_values(by="far").iloc[0]
            low_far_metrics = {
                "threshold": round(float(lowest_far_row["threshold"]), 4),
                "target_far": target_low_far,
                "achieved_far": round(float(lowest_far_row["far"]), 6),
                "accuracy": round(float(lowest_far_row["accuracy"]), 4),
                "frr": round(float(lowest_far_row["frr"]), 6),
                "precision": round(float(lowest_far_row["precision"]), 4),
                "recall": round(float(lowest_far_row["recall"]), 4),
                "f1": round(float(lowest_far_row["f1"]), 4),
                "acceptance_rate": round(float(lowest_far_row["acceptance_rate"]), 4),
                "supported": False
            }

        # 4. F1-Optimal Strategy
        max_f1_val = sweep_df["f1"].max()
        max_f1_candidates = sweep_df[sweep_df["f1"] == max_f1_val]
        max_f1_row = max_f1_candidates.iloc[len(max_f1_candidates) // 2]
        f1_optimal_metrics = {
            "threshold": round(float(max_f1_row["threshold"]), 4),
            "accuracy": round(float(max_f1_row["accuracy"]), 4),
            "far": round(float(max_f1_row["far"]), 6),
            "frr": round(float(max_f1_row["frr"]), 6),
            "precision": round(float(max_f1_row["precision"]), 4),
            "recall": round(float(max_f1_row["recall"]), 4),
            "f1": round(float(max_f1_row["f1"]), 4),
            "acceptance_rate": round(float(max_f1_row["acceptance_rate"]), 4)
        }

        # 5. Discrete FAR/FRR-Balanced Strategy (min |FAR - FRR|)
        diffs = (sweep_df["far"] - sweep_df["frr"]).abs()
        min_diff_idx = int(diffs.argmin())
        balanced_row = sweep_df.iloc[min_diff_idx]
        far_frr_balanced_metrics = {
            "threshold": round(float(balanced_row["threshold"]), 4),
            "far_frr_diff": round(float(diffs.iloc[min_diff_idx]), 6),
            "accuracy": round(float(balanced_row["accuracy"]), 4),
            "far": round(float(balanced_row["far"]), 6),
            "frr": round(float(balanced_row["frr"]), 6),
            "precision": round(float(balanced_row["precision"]), 4),
            "recall": round(float(balanced_row["recall"]), 4),
            "f1": round(float(balanced_row["f1"]), 4),
            "acceptance_rate": round(float(balanced_row["acceptance_rate"]), 4)
        }

        eer_operating_metrics = _get_metrics_at_thresh(eer_thresh)
        eer_operating_metrics["eer_continuous"] = round(eer_val, 4)

        return {
            "roc_auc": round(roc_auc, 4),
            "eer_operating_point": eer_operating_metrics,
            "maximum_accuracy": max_acc_metrics,
            "security_oriented_low_far": low_far_metrics,
            "f1_optimal": f1_optimal_metrics,
            "far_frr_balanced": far_frr_balanced_metrics
        }

    def evaluate_threshold_stability(
        self,
        pairs_df: pd.DataFrame,
        num_splits: int = 5,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Evaluates stability of threshold calibration using 5-fold cross-validation
        partitioned across validation identities.
        """
        identities = sorted(list(set(pairs_df["identity1"]).union(set(pairs_df["identity2"]))))
        if len(identities) < num_splits:
            return {"error": "Insufficient identities for cross-validation"}

        rng = np.random.RandomState(random_seed)
        shuffled_ids = np.array(identities)
        rng.shuffle(shuffled_ids)
        folds = np.array_split(shuffled_ids, num_splits)

        cv_eer_threshs = []
        cv_max_acc_threshs = []
        cv_low_far_threshs = []

        for fold_idx, test_ids in enumerate(folds):
            test_set = set(test_ids)
            train_mask = ~(pairs_df["identity1"].isin(test_set) | pairs_df["identity2"].isin(test_set))
            train_df = pairs_df[train_mask]

            if len(train_df) < 100:
                continue

            labels = train_df["is_same"].values
            scores = train_df["cosine_similarity"].values

            fpr, tpr, threshs = calculate_roc_curve(labels, scores, "similarity")
            _, eer_th = calculate_eer(fpr, tpr, threshs)
            cv_eer_threshs.append(eer_th)

            sweep = self.sweep_thresholds(labels, scores, step=0.005)
            max_acc_th = sweep.loc[sweep["accuracy"].idxmax(), "threshold"]
            cv_max_acc_threshs.append(max_acc_th)

            low_far_sub = sweep[sweep["far"] <= 0.001]
            if len(low_far_sub) > 0:
                cv_low_far_threshs.append(low_far_sub.iloc[0]["threshold"])

        def _stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {}
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4)
            }

        return {
            "num_folds": num_splits,
            "eer_threshold_stability": _stats(cv_eer_threshs),
            "max_accuracy_threshold_stability": _stats(cv_max_acc_threshs),
            "low_far_threshold_stability": _stats(cv_low_far_threshs)
        }
