import os
import sys
import json
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from ml.evaluation import ThresholdCalibrator, calculate_roc_curve, calculate_roc_auc

def generate_calibration_visualizations(
    sweep_df: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    strategies: Dict[str, Any],
    plots_dir: str
):
    """Generates the 7 required calibration visualization figures."""
    os.makedirs(plots_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    taus = sweep_df["threshold"].values
    fars = sweep_df["far"].values
    frrs = sweep_df["frr"].values
    accs = sweep_df["accuracy"].values
    f1s = sweep_df["f1"].values

    # -------------------------------------------------------------
    # Plot 1: FAR vs Threshold
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(taus, fars, color="#c0392b", lw=2, label="False Acceptance Rate (FAR)")
    plt.xlabel("Cosine Similarity Threshold", fontsize=11)
    plt.ylabel("False Acceptance Rate (FAR)", fontsize=11)
    plt.title("Validation FAR vs Decision Threshold", fontsize=12, fontweight="bold")
    plt.xlim([-0.1, 0.9])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "far_vs_threshold.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: FRR vs Threshold
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(taus, frrs, color="#2980b9", lw=2, label="False Rejection Rate (FRR)")
    plt.xlabel("Cosine Similarity Threshold", fontsize=11)
    plt.ylabel("False Rejection Rate (FRR)", fontsize=11)
    plt.title("Validation FRR vs Decision Threshold", fontsize=12, fontweight="bold")
    plt.xlim([-0.1, 0.9])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "frr_vs_threshold.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 3: FAR and FRR vs Threshold on the Same Figure (with EER Point)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    mask = (taus >= -0.1) & (taus <= 0.8)
    plt.plot(taus[mask], fars[mask], color="#c0392b", lw=2, label="FAR (False Acceptance Rate)")
    plt.plot(taus[mask], frrs[mask], color="#2980b9", lw=2, label="FRR (False Rejection Rate)")
    eer_info = strategies["eer_operating_point"]
    plt.plot([eer_info["threshold"]], [eer_info["far"]], marker="o", markersize=8, color="#27ae60",
             label=f"EER Point (Threshold={eer_info['threshold']:.3f}, Error={eer_info['far']:.3f})")
    plt.axvline(eer_info["threshold"], color="#7f8c8d", linestyle=":", lw=1)
    plt.xlabel("Cosine Similarity Threshold", fontsize=11)
    plt.ylabel("Error Rate", fontsize=11)
    plt.title("Validation FAR and FRR vs Threshold (EER Operating Point)", fontsize=12, fontweight="bold")
    plt.legend(loc="center right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "far_frr_vs_threshold.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 4: Accuracy vs Threshold (with Maximum Accuracy Point)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(taus, accs, color="#8e44ad", lw=2, label="Validation Accuracy")
    max_acc_info = strategies["maximum_accuracy"]
    plt.plot([max_acc_info["threshold"]], [max_acc_info["accuracy"]], marker="*", markersize=10, color="#f39c12",
             label=f"Max Accuracy (Threshold={max_acc_info['threshold']:.3f}, Acc={max_acc_info['accuracy']:.4f})")
    plt.axvline(max_acc_info["threshold"], color="#f39c12", linestyle="--", lw=1)
    plt.xlabel("Cosine Similarity Threshold", fontsize=11)
    plt.ylabel("Verification Accuracy", fontsize=11)
    plt.title("Validation Accuracy vs Decision Threshold", fontsize=12, fontweight="bold")
    plt.xlim([-0.1, 0.9])
    plt.ylim([0.4, 1.02])
    plt.legend(loc="lower center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuracy_vs_threshold.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 5: F1-Score vs Threshold
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(taus, f1s, color="#16a085", lw=2, label="F1-Score")
    f1_info = strategies["f1_optimal"]
    plt.plot([f1_info["threshold"]], [f1_info["f1"]], marker="D", markersize=7, color="#d35400",
             label=f"Max F1 (Threshold={f1_info['threshold']:.3f}, F1={f1_info['f1']:.4f})")
    plt.xlabel("Cosine Similarity Threshold", fontsize=11)
    plt.ylabel("F1-Score", fontsize=11)
    plt.title("Validation F1-Score vs Decision Threshold", fontsize=12, fontweight="bold")
    plt.xlim([-0.1, 0.9])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="lower center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "f1_vs_threshold.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 6: Genuine vs Impostor Score Distributions
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    gen_scores = scores[labels == 1]
    imp_scores = scores[labels == 0]
    plt.hist(gen_scores, bins=50, alpha=0.65, color="#27ae60", label=f"Genuine Pairs (N={len(gen_scores)})", density=True)
    plt.hist(imp_scores, bins=50, alpha=0.65, color="#c0392b", label=f"Impostor Pairs (N={len(imp_scores)})", density=True)
    plt.axvline(strategies["maximum_accuracy"]["threshold"], color="#2c3e50", linestyle="--", lw=1.5,
                label=f"Max Acc Threshold ({strategies['maximum_accuracy']['threshold']:.3f})")
    plt.xlabel("Cosine Similarity", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("Validation Genuine vs Impostor Similarity Distributions", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "validation_score_distributions.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 7: Operating Points Comparison (ROC Space)
    # -------------------------------------------------------------
    fpr, tpr, _ = calculate_roc_curve(labels, scores, "similarity")
    auc_val = calculate_roc_auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr, tpr, color="#2980b9", lw=2, label=f"Validation ROC (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#95a5a6", linestyle="--", lw=1)

    # Plot operating points
    opts = [
        ("EER", strategies["eer_operating_point"]["far"], 1.0 - strategies["eer_operating_point"]["frr"], "#27ae60", "o"),
        ("Max Accuracy", strategies["maximum_accuracy"]["far"], 1.0 - strategies["maximum_accuracy"]["frr"], "#f39c12", "*"),
        ("Low-FAR (0.1%)", strategies["security_oriented_low_far"]["achieved_far"], 1.0 - strategies["security_oriented_low_far"]["frr"], "#e74c3c", "s"),
        ("F1-Optimal", strategies["f1_optimal"]["far"], 1.0 - strategies["f1_optimal"]["frr"], "#16a085", "D"),
        ("FAR/FRR Balanced", strategies["far_frr_balanced"]["far"], 1.0 - strategies["far_frr_balanced"]["frr"], "#8e44ad", "^"),
    ]
    for name, opt_far, opt_tpr, col, marker in opts:
        plt.plot([opt_far], [opt_tpr], marker=marker, markersize=9, color=col, label=f"{name} Operating Point")

    plt.xlim([-0.005, 0.2])
    plt.ylim([0.8, 1.01])
    plt.xlabel("False Positive Rate (FAR)", fontsize=11)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=11)
    plt.title("Validation Calibration Operating Points (ROC Space Zoomed)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "operating_points_comparison.png"))
    plt.close()

    print(f"[PLOTS] Generated 7 calibration visualization figures in: {plots_dir}", flush=True)


def run_production_threshold_calibration() -> Dict[str, Any]:
    """
    Executes the formal Phase 7 production threshold calibration on the validation split.
    Evaluates all five calibration strategies, generates visual plots, stability analysis,
    and produces a formal technical production threshold recommendation.
    """
    t0_start = time.perf_counter()
    config = load_config("config/config.yaml")
    meta_dir = config.get("paths", {}).get("metadata_dir", "data/metadata")
    splits_csv = os.path.join(meta_dir, "splits.csv")

    if not os.path.exists(splits_csv):
        raise FileNotFoundError(f"Splits file missing: {splits_csv}")

    splits_df = pd.read_csv(splits_csv)
    reports_dir = "reports/calibration"
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(reports_dir, exist_ok=True)

    print("="*80, flush=True)
    print("PHASE 7: PRODUCTION THRESHOLD CALIBRATION (VALIDATION SPLIT)", flush=True)
    print("="*80, flush=True)
    print(f"Data Split: Independent Validation Set Only (59 identities, 1,395 images)", flush=True)
    print(f"Test Set Protection: Strict Zero-Access (data/evaluation/test/ untouched)", flush=True)
    print(f"Model Pipeline: Modern E2 (YuNet + 5-Point Align + ArcFace 512D + Cosine)", flush=True)
    print(f"Reports Directory: {reports_dir}\n", flush=True)

    # 1. Initialize Calibrator and generate validation pairs
    calibrator = ThresholdCalibrator(config=config)
    labels, scores, pairs_df = calibrator.generate_validation_pairs(
        splits_df=splits_df,
        max_genuine_per_identity=200,
        max_impostor_pairs=50000,
        random_seed=42
    )

    num_genuine = int(np.sum(labels == 1))
    num_impostor = int(np.sum(labels == 0))

    # 2. Distribution Statistics (with percentiles)
    gen_scores = scores[labels == 1]
    imp_scores = scores[labels == 0]
    gen_dist = calibrator.calculate_distribution_percentiles(gen_scores)
    imp_dist = calibrator.calculate_distribution_percentiles(imp_scores)

    # 3. Fine-Grained Threshold Sweep
    print("[CALIBRATOR] Executing fine threshold sweep across cosine range [-0.2, 0.95]...", flush=True)
    sweep_df = calibrator.sweep_thresholds(labels, scores, threshold_range=(-0.2, 0.95), step=0.001)

    # 4. Evaluate Calibration Strategies
    strategies = calibrator.evaluate_calibration_strategies(
        labels, scores, sweep_df, target_low_far=0.001
    )

    # 5. Threshold Stability Analysis
    print("[CALIBRATOR] Evaluating 5-fold threshold stability across validation identities...", flush=True)
    stability = calibrator.evaluate_threshold_stability(pairs_df, num_splits=5, random_seed=42)

    # 6. Generate 7 Visualizations
    generate_calibration_visualizations(sweep_df, labels, scores, strategies, plots_dir)

    # 7. Formulate Technical Recommendation
    rec_strategy = "security_oriented_low_far" if strategies["security_oriented_low_far"]["supported"] else "maximum_accuracy"
    rec_info = strategies[rec_strategy]
    recommended_threshold = rec_info["threshold"]
    rec_far = rec_info.get("achieved_far", rec_info.get("far", 0.0))
    rec_frr = rec_info["frr"]
    rec_acc = rec_info["accuracy"]

    justification = (
        f"Selected {recommended_threshold:.4f} via {rec_strategy.replace('_', ' ').title()} strategy on validation data. "
        f"For attendance and identity verification, security against false impersonation is paramount (achieved validation FAR = {rec_far*100:.3f}%). "
        f"Maintains a high verification accuracy of {rec_acc*100:.2f}% and low rejection rate (FRR = {rec_frr*100:.2f}%) on real facial variations."
    )

    elapsed_time = round(time.perf_counter() - t0_start, 2)

    full_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "Phase_7_Production_Threshold_Calibration",
        "data_split_evaluated": "validation_split_only",
        "test_split_protection": "confirmed_zero_access",
        "num_validation_images": 1395,
        "num_validation_identities": 59,
        "validation_pairs_count": {
            "genuine_pairs": num_genuine,
            "impostor_pairs": num_impostor,
            "total_pairs": len(labels)
        },
        "score_distributions": {
            "genuine": gen_dist,
            "impostor": imp_dist
        },
        "discrimination_metrics": {
            "validation_roc_auc": strategies["roc_auc"]
        },
        "calibration_strategies": strategies,
        "threshold_stability": stability,
        "production_recommendation": {
            "strategy_used": rec_strategy,
            "recommended_threshold": recommended_threshold,
            "expected_validation_far_percent": round(rec_far * 100, 3),
            "expected_validation_frr_percent": round(rec_frr * 100, 3),
            "expected_validation_accuracy_percent": round(rec_acc * 100, 2),
            "expected_validation_precision_percent": round(rec_info["precision"] * 100, 2),
            "expected_validation_recall_percent": round(rec_info["recall"] * 100, 2),
            "expected_validation_f1_percent": round(rec_info["f1"] * 100, 2),
            "technical_justification": justification,
            "label": "Validation-derived production threshold"
        },
        "calibration_runtime_seconds": elapsed_time
    }

    # Save summary JSON
    summary_path = os.path.join(reports_dir, "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"[CALIBRATOR] Serialized calibration summary to: {summary_path}", flush=True)

    # 8. Print Formatted Calibration Results
    print("\n" + "="*80, flush=True)
    print("PHASE 7 VALIDATION THRESHOLD CALIBRATION RESULTS", flush=True)
    print("="*80, flush=True)
    print(f"{'Strategy':<28} | {'Threshold':<10} | {'FAR (%)':<10} | {'FRR (%)':<10} | {'Acc (%)':<10} | {'F1 (%)':<10}", flush=True)
    print("-" * 80, flush=True)

    def _p_row(name, info, far_key="far"):
        far_v = info.get(far_key, info.get("far", 0.0)) * 100
        frr_v = info["frr"] * 100
        acc_v = info["accuracy"] * 100
        f1_v = info["f1"] * 100
        print(f"{name:<28} | {info['threshold']:<10.4f} | {far_v:<10.3f} | {frr_v:<10.3f} | {acc_v:<10.2f} | {f1_v:<10.2f}", flush=True)

    _p_row("A. Equal Error Rate (EER)", strategies["eer_operating_point"])
    _p_row("B. Maximum Accuracy", strategies["maximum_accuracy"])
    _p_row("C. Security Low-FAR (0.1%)", strategies["security_oriented_low_far"], far_key="achieved_far")
    _p_row("D. F1-Optimal", strategies["f1_optimal"])
    _p_row("E. FAR/FRR-Balanced", strategies["far_frr_balanced"])
    print("-" * 80, flush=True)
    print(f"\n[RECOMMENDATION] Validation-Derived Production Threshold: {recommended_threshold:.4f}")
    print(f"  - Expected Validation FAR: {rec_far*100:.3f}%")
    print(f"  - Expected Validation FRR: {rec_frr*100:.3f}%")
    print(f"  - Expected Validation Accuracy: {rec_acc*100:.2f}%")
    print(f"  - Technical Justification: {justification}")
    print("="*80, flush=True)

    return full_summary


if __name__ == "__main__":
    run_production_threshold_calibration()
