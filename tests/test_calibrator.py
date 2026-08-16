import pytest
import numpy as np
import pandas as pd
from ml.evaluation.calibrator import ThresholdCalibrator

def test_sweep_thresholds_metrics_calculation():
    # 4 synthetic samples: 2 genuine (scores 0.8, 0.6), 2 impostor (scores 0.4, 0.2)
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.8, 0.6, 0.4, 0.2])

    sweep_df = ThresholdCalibrator.sweep_thresholds(labels, scores, threshold_range=(0.0, 1.0), step=0.1)
    assert len(sweep_df) == 11

    # At threshold 0.5:
    # predictions: [1, 1, 0, 0] -> TP=2, TN=2, FP=0, FN=0 -> Acc=1.0, FAR=0.0, FRR=0.0, Precision=1.0, Recall=1.0, F1=1.0
    row_05 = sweep_df[np.isclose(sweep_df["threshold"], 0.5)].iloc[0]
    assert row_05["tp"] == 2
    assert row_05["tn"] == 2
    assert row_05["fp"] == 0
    assert row_05["fn"] == 0
    assert row_05["accuracy"] == 1.0
    assert row_05["far"] == 0.0
    assert row_05["frr"] == 0.0
    assert row_05["f1"] == 1.0


def test_distribution_percentiles_calculation():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    stats = ThresholdCalibrator.calculate_distribution_percentiles(scores)
    assert stats["count"] == 10
    assert stats["min"] == 0.1
    assert stats["max"] == 1.0
    assert "p1" in stats
    assert "p5" in stats
    assert "p25" in stats
    assert "p50" in stats
    assert "p75" in stats
    assert "p95" in stats
    assert "p99" in stats
    assert stats["p50"] == stats["median"]


def test_calibration_strategies_synthetic():
    # Construct synthetic distributions
    rng = np.random.RandomState(42)
    gen_scores = rng.normal(loc=0.65, scale=0.08, size=500)
    imp_scores = rng.normal(loc=0.05, scale=0.08, size=2000)

    labels = np.concatenate([np.ones(len(gen_scores)), np.zeros(len(imp_scores))])
    scores = np.concatenate([gen_scores, imp_scores])

    calibrator = ThresholdCalibrator(config={})
    sweep_df = calibrator.sweep_thresholds(labels, scores, threshold_range=(-0.2, 1.0), step=0.01)
    strategies = calibrator.evaluate_calibration_strategies(labels, scores, sweep_df, target_low_far=0.001)

    assert "roc_auc" in strategies
    assert strategies["roc_auc"] > 0.95
    assert "eer_operating_point" in strategies
    assert "maximum_accuracy" in strategies
    assert "security_oriented_low_far" in strategies
    assert "f1_optimal" in strategies
    assert "far_frr_balanced" in strategies

    # Verify low-FAR threshold meets or approximates FAR constraint
    low_far_info = strategies["security_oriented_low_far"]
    assert low_far_info["threshold"] > 0.1
    assert low_far_info["achieved_far"] <= 0.005

    # Verify FAR/FRR balanced point has small difference
    bal_info = strategies["far_frr_balanced"]
    assert bal_info["far_frr_diff"] < 0.05


def test_zero_test_set_leakage_in_validation_pairs():
    """Confirms that generate_validation_pairs strictly filters out test samples and raises error if only test split provided."""
    calibrator = ThresholdCalibrator(config={})
    # Mock split dataframe containing test and validation
    test_only_df = pd.DataFrame([
        {"split": "test", "identity": "Alice", "relative_path": "data/evaluation/test/Alice/01.jpg"},
        {"split": "test", "identity": "Alice", "relative_path": "data/evaluation/test/Alice/02.jpg"}
    ])

    with pytest.raises(ValueError, match="No validation samples found in splits dataframe"):
        calibrator.generate_validation_pairs(test_only_df)
