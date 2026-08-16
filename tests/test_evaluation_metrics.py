import pytest
import numpy as np
import pandas as pd
from ml.evaluation.metrics import (
    calculate_roc_curve,
    calculate_roc_auc,
    calculate_eer,
    calculate_far_frr,
    find_optimal_threshold,
    calculate_fold_aware_metrics,
    calculate_score_statistics
)

def test_roc_and_auc_similarity_perfect_separation():
    # Similarity: Genuine have higher scores
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    fpr, tpr, thresholds = calculate_roc_curve(labels, scores, score_direction="similarity")
    auc_val = calculate_roc_auc(fpr, tpr)
    assert auc_val == 1.0


def test_roc_and_auc_distance_perfect_separation():
    # Distance: Genuine have lower scores
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.2])
    fpr, tpr, thresholds = calculate_roc_curve(labels, scores, score_direction="distance")
    auc_val = calculate_roc_auc(fpr, tpr)
    assert auc_val == 1.0


def test_eer_calculation_synthetic():
    # Symmetric crossover
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.8, 0.7, 0.2, 0.1])
    fpr, tpr, thresholds = calculate_roc_curve(labels, scores, score_direction="similarity")
    eer, eer_thresh = calculate_eer(fpr, tpr, thresholds)
    assert 0.0 <= eer <= 1.0
    assert -1.0 <= eer_thresh <= 1.0


def test_far_frr_calculation():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.8, 0.4, 0.6, 0.2])  # threshold 0.5 -> pred [1, 0, 1, 0]
    acc, far, frr = calculate_far_frr(labels, scores, threshold=0.5, score_direction="similarity")
    assert acc == 0.5
    assert far == 0.5  # 1 false accept out of 2 impostors
    assert frr == 0.5  # 1 false reject out of 2 genuine


def test_far_frr_distance_score_direction():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.2, 0.7, 0.4, 0.9])  # threshold 0.5 -> pred [1, 0, 1, 0]
    acc, far, frr = calculate_far_frr(labels, scores, threshold=0.5, score_direction="distance")
    assert acc == 0.5
    assert far == 0.5
    assert frr == 0.5


def test_score_statistics_summary():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.8, 0.6, 0.1, 0.3])
    stats = calculate_score_statistics(labels, scores)
    assert stats["genuine"]["count"] == 2
    assert stats["genuine"]["mean"] == 0.7
    assert stats["impostor"]["count"] == 2
    assert stats["impostor"]["mean"] == 0.2


def test_invalid_score_direction():
    labels = np.array([1, 0])
    scores = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        calculate_roc_curve(labels, scores, score_direction="invalid_mode")
    with pytest.raises(ValueError):
        calculate_far_frr(labels, scores, threshold=0.5, score_direction="invalid_mode")


def test_find_optimal_threshold_similarity():
    # Clean separation between 0.2 (impostor) and 0.8 (genuine)
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.85, 0.8, 0.1, 0.15, 0.2])
    opt_thresh = find_optimal_threshold(labels, scores, score_direction="similarity")
    # Optimal threshold must lie between 0.2 and 0.8
    assert 0.2 < opt_thresh < 0.8
    acc, far, frr = calculate_far_frr(labels, scores, threshold=opt_thresh, score_direction="similarity")
    assert acc == 1.0
    assert far == 0.0
    assert frr == 0.0


def test_find_optimal_threshold_distance():
    # Clean distance separation: genuine around 0.2, impostor around 0.8
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    opt_thresh = find_optimal_threshold(labels, scores, score_direction="distance")
    # Optimal threshold must lie between 0.2 and 0.8
    assert 0.2 < opt_thresh < 0.8
    acc, far, frr = calculate_far_frr(labels, scores, threshold=opt_thresh, score_direction="distance")
    assert acc == 1.0
    assert far == 0.0
    assert frr == 0.0


def test_fold_aware_metrics_no_leakage():
    """Verifies that fold-aware threshold calibration uses only training folds with zero test fold leakage."""
    # Construct synthetic 3-fold dataframe
    records = []
    # Fold 1: genuine 0.9, impostor 0.1
    records += [{"fold": 1, "is_same": 1, "raw_score": 0.9}] * 10
    records += [{"fold": 1, "is_same": 0, "raw_score": 0.1}] * 10
    # Fold 2: genuine 0.85, impostor 0.15
    records += [{"fold": 2, "is_same": 1, "raw_score": 0.85}] * 10
    records += [{"fold": 2, "is_same": 0, "raw_score": 0.15}] * 10
    # Fold 3: genuine 0.80, impostor 0.20
    records += [{"fold": 3, "is_same": 1, "raw_score": 0.80}] * 10
    records += [{"fold": 3, "is_same": 0, "raw_score": 0.20}] * 10

    df = pd.DataFrame(records)
    summary = calculate_fold_aware_metrics(df, score_direction="similarity")

    assert summary["calibrated_accuracy_mean"] == 1.0
    assert summary["calibrated_far_mean"] == 0.0
    assert summary["calibrated_frr_mean"] == 0.0
    assert len(summary["folds"]) == 3

    # Leakage Test: If we corrupt Fold 3's test scores, Fold 1's calibrated threshold must remain IDENTICAL
    # because Fold 1 is trained on {Fold 2, Fold 3}.
    # But if we change Fold 1's test scores, Fold 1's trained threshold (which depends on Fold 2 & Fold 3)
    # must NOT change at all!
    fold1_original_thresh = summary["folds"][1]["calibrated_threshold"]

    df_corrupted = df.copy()
    # Modify Fold 1 test scores drastically
    df_corrupted.loc[df_corrupted["fold"] == 1, "raw_score"] = 999.0
    summary_corrupted = calculate_fold_aware_metrics(df_corrupted, score_direction="similarity")

    # Threshold selected for evaluating Fold 1 must NOT be affected by Fold 1's own data
    fold1_new_thresh = summary_corrupted["folds"][1]["calibrated_threshold"]
    assert fold1_original_thresh == fold1_new_thresh
