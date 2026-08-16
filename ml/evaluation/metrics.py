import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.metrics import roc_curve, auc

def calculate_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    score_direction: str = "similarity"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes False Positive Rate (FPR), True Positive Rate (TPR), and score thresholds.

    Args:
        labels (np.ndarray): Binary ground truth (1 for genuine/same, 0 for impostor/different).
        scores (np.ndarray): Continuous match scores.
        score_direction (str): 'similarity' (higher score = more similar) or
                               'distance' (lower score = more similar).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (fpr, tpr, thresholds)
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    if score_direction.lower() == "distance":
        # For distance metrics (e.g. Euclidean), smaller distance indicates genuine match.
        # Negate scores so higher values correspond to positive class (label 1).
        roc_scores = -scores_arr
        fpr, tpr, raw_thresholds = roc_curve(labels_arr, roc_scores)
        # Restore thresholds to original distance scale
        thresholds = -raw_thresholds
    elif score_direction.lower() == "similarity":
        # For similarity metrics (e.g. Cosine), higher score indicates genuine match.
        fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    else:
        raise ValueError(f"Unknown score_direction: '{score_direction}'. Must be 'similarity' or 'distance'.")

    return fpr, tpr, thresholds


def calculate_roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Calculates Area Under the ROC Curve (ROC-AUC)."""
    return float(auc(fpr, tpr))


def calculate_eer(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray
) -> Tuple[float, float]:
    """
    Estimates the Equal Error Rate (EER) where FAR (FPR) equals FRR (1 - TPR).

    Returns:
        Tuple[float, float]: (eer_value, eer_threshold)
    """
    fnr = 1.0 - tpr
    diffs = np.abs(fpr - fnr)
    min_idx = int(np.argmin(diffs))

    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    eer_threshold = float(thresholds[min_idx])
    return eer, eer_threshold


def calculate_far_frr(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    score_direction: str = "similarity"
) -> Tuple[float, float, float]:
    """
    Calculates Verification Accuracy, FAR (False Acceptance Rate), and FRR (False Rejection Rate)
    at a specific decision threshold.

    Args:
        labels (np.ndarray): Binary ground truth (1 for genuine, 0 for impostor).
        scores (np.ndarray): Continuous match scores.
        threshold (float): Decision threshold.
        score_direction (str): 'similarity' or 'distance'.

    Returns:
        Tuple[float, float, float]: (accuracy, far, frr)
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    if score_direction.lower() == "similarity":
        predictions = (scores_arr >= threshold).astype(int)
    elif score_direction.lower() == "distance":
        predictions = (scores_arr <= threshold).astype(int)
    else:
        raise ValueError(f"Unknown score_direction: '{score_direction}'.")

    # True Positives, False Positives, True Negatives, False Negatives
    tp = int(np.sum((predictions == 1) & (labels_arr == 1)))
    fp = int(np.sum((predictions == 1) & (labels_arr == 0)))
    tn = int(np.sum((predictions == 0) & (labels_arr == 0)))
    fn = int(np.sum((predictions == 0) & (labels_arr == 1)))

    total = len(labels_arr)
    num_impostor = fp + tn
    num_genuine = tp + fn

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    far = float(fp / num_impostor) if num_impostor > 0 else 0.0
    frr = float(fn / num_genuine) if num_genuine > 0 else 0.0

    return accuracy, far, frr


def find_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    score_direction: str = "similarity",
    num_candidates: int = 1000
) -> float:
    """
    Finds the decision threshold that maximizes verification accuracy on training data.
    Uses candidate grid search across the score range.

    Args:
        labels (np.ndarray): Binary ground truth.
        scores (np.ndarray): Continuous match scores.
        score_direction (str): 'similarity' or 'distance'.
        num_candidates (int): Number of threshold candidate points.

    Returns:
        float: The optimal decision threshold.
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    if len(scores_arr) == 0:
        return 0.0

    min_s = float(np.min(scores_arr))
    max_s = float(np.max(scores_arr))

    candidates = np.linspace(min_s, max_s, num_candidates)
    best_acc = -1.0
    best_thresholds = []

    for thresh in candidates:
        if score_direction.lower() == "similarity":
            preds = (scores_arr >= thresh).astype(int)
        elif score_direction.lower() == "distance":
            preds = (scores_arr <= thresh).astype(int)
        else:
            raise ValueError(f"Unknown score_direction: '{score_direction}'.")

        acc = float(np.mean(preds == labels_arr))
        if acc > best_acc:
            best_acc = acc
            best_thresholds = [thresh]
        elif np.isclose(acc, best_acc, atol=1e-7):
            best_thresholds.append(thresh)

    # Return median optimal threshold for numerical stability
    return float(np.median(best_thresholds))


def calculate_fold_aware_metrics(
    valid_df: pd.DataFrame,
    score_direction: str = "similarity"
) -> Dict[str, Any]:
    """
    Implements the official LFW 10-fold cross-validation threshold calibration.
    For each fold k in [1..10]:
      1. Train threshold on the other 9 folds (F \ {k}).
      2. Evaluate trained threshold on held-out fold k (zero leakage).
      3. Measure held-out accuracy, FAR, FRR.

    Args:
        valid_df (pd.DataFrame): DataFrame with columns ['fold', 'is_same', 'raw_score'].
        score_direction (str): 'similarity' or 'distance'.

    Returns:
        Dict[str, Any]: Comprehensive fold-aware evaluation summary.
    """
    folds = sorted(list(valid_df["fold"].unique()))
    fold_results = {}

    for fold_k in folds:
        # Step 1: Split into 9 training folds and 1 held-out test fold
        train_data = valid_df[valid_df["fold"] != fold_k]
        test_data = valid_df[valid_df["fold"] == fold_k]

        train_labels = train_data["is_same"].values
        train_scores = train_data["raw_score"].values

        test_labels = test_data["is_same"].values
        test_scores = test_data["raw_score"].values

        # Step 2: Select optimal threshold ONLY on training folds
        selected_thresh = find_optimal_threshold(
            train_labels, train_scores, score_direction=score_direction, num_candidates=2000
        )

        # Step 3: Evaluate on the held-out test fold
        test_acc, test_far, test_frr = calculate_far_frr(
            test_labels, test_scores, threshold=selected_thresh, score_direction=score_direction
        )

        # Also compute fold-specific ROC-AUC and EER
        fpr, tpr, threshs = calculate_roc_curve(test_labels, test_scores, score_direction)
        fold_auc = calculate_roc_auc(fpr, tpr)
        fold_eer, fold_eer_thresh = calculate_eer(fpr, tpr, threshs)

        fold_results[int(fold_k)] = {
            "test_pairs": len(test_data),
            "train_pairs": len(train_data),
            "calibrated_threshold": round(selected_thresh, 4),
            "calibrated_accuracy": round(test_acc, 4),
            "calibrated_far": round(test_far, 4),
            "calibrated_frr": round(test_frr, 4),
            "roc_auc": round(fold_auc, 4),
            "eer": round(fold_eer, 4),
            "eer_threshold": round(fold_eer_thresh, 4),
            "score_stats": calculate_score_statistics(test_labels, test_scores)
        }

    # Aggregate cross-fold statistics
    cal_accs = [f["calibrated_accuracy"] for f in fold_results.values()]
    cal_fars = [f["calibrated_far"] for f in fold_results.values()]
    cal_frrs = [f["calibrated_frr"] for f in fold_results.values()]
    cal_threshs = [f["calibrated_threshold"] for f in fold_results.values()]
    fold_aucs = [f["roc_auc"] for f in fold_results.values()]
    fold_eers = [f["eer"] for f in fold_results.values()]

    return {
        "calibrated_accuracy_mean": round(float(np.mean(cal_accs)), 4),
        "calibrated_accuracy_std": round(float(np.std(cal_accs)), 4),
        "calibrated_far_mean": round(float(np.mean(cal_fars)), 4),
        "calibrated_far_std": round(float(np.std(cal_fars)), 4),
        "calibrated_frr_mean": round(float(np.mean(cal_frrs)), 4),
        "calibrated_frr_std": round(float(np.std(cal_frrs)), 4),
        "calibrated_threshold_mean": round(float(np.mean(cal_threshs)), 4),
        "calibrated_threshold_std": round(float(np.std(cal_threshs)), 4),
        "roc_auc_mean": round(float(np.mean(fold_aucs)), 4),
        "roc_auc_std": round(float(np.std(fold_aucs)), 4),
        "eer_mean": round(float(np.mean(fold_eers)), 4),
        "eer_std": round(float(np.std(fold_eers)), 4),
        "folds": fold_results
    }


def calculate_score_statistics(
    labels: np.ndarray,
    scores: np.ndarray
) -> Dict[str, Any]:
    """Computes summary statistics for genuine and impostor score distributions."""
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    gen_scores = scores_arr[labels_arr == 1]
    imp_scores = scores_arr[labels_arr == 0]

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        if len(arr) == 0:
            return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4)
        }

    return {
        "genuine": _stats(gen_scores),
        "impostor": _stats(imp_scores)
    }
