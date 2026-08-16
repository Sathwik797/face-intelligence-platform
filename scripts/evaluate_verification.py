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
from ml.evaluation import (
    VerificationEvaluator,
    calculate_roc_curve,
    calculate_roc_auc,
    calculate_eer
)

def generate_evaluation_visualizations(
    df_e1: pd.DataFrame,
    df_e2: pd.DataFrame,
    summary_e1: Dict[str, Any],
    summary_e2: Dict[str, Any],
    plots_dir: str
):
    """Generates the 5 required evaluation plots comparing E1 vs E2."""
    os.makedirs(plots_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Filter valid scores
    valid_e1 = df_e1[df_e1["raw_score"].notnull()]
    valid_e2 = df_e2[df_e2["raw_score"].notnull()]

    fpr_e1, tpr_e1, thresh_e1 = calculate_roc_curve(valid_e1["is_same"].values, valid_e1["raw_score"].values, "distance")
    auc_e1 = calculate_roc_auc(fpr_e1, tpr_e1)

    fpr_e2, tpr_e2, thresh_e2 = calculate_roc_curve(valid_e2["is_same"].values, valid_e2["raw_score"].values, "similarity")
    auc_e2 = calculate_roc_auc(fpr_e2, tpr_e2)

    # -------------------------------------------------------------
    # Plot 1: ROC Curve Comparison (E1 vs E2)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr_e1, tpr_e1, color="#e74c3c", lw=2, label=f"E1: dlib Baseline (AUC = {auc_e1:.4f})")
    plt.plot(fpr_e2, tpr_e2, color="#2980b9", lw=2.5, label=f"E2: ArcFace Modern (AUC = {auc_e2:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", lw=1, label="Random Guess (AUC = 0.5000)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (FAR)", fontsize=12)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    plt.title("LFW 10-Fold Face Verification ROC Curve: E1 vs E2", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "roc_curve_e1_vs_e2.png")
    plt.savefig(p1)
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: Score Distribution E1 (Euclidean Distance)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    gen_e1 = valid_e1[valid_e1["is_same"] == 1]["raw_score"].values
    imp_e1 = valid_e1[valid_e1["is_same"] == 0]["raw_score"].values
    plt.hist(gen_e1, bins=50, alpha=0.65, color="#27ae60", label="Genuine Pairs (Same Person)", density=True)
    plt.hist(imp_e1, bins=50, alpha=0.65, color="#c0392b", label="Impostor Pairs (Different Person)", density=True)
    plt.axvline(summary_e1["discrimination_metrics"]["global_eer_threshold"], color="#2c3e50", linestyle="--", lw=1.5,
                label=f"EER Operating Thresh ({summary_e1['discrimination_metrics']['global_eer_threshold']:.3f})")
    plt.xlabel("Euclidean Distance (Lower = More Similar)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("E1 (dlib 128D) Genuine vs Impostor Score Distributions", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "score_distribution_e1_euclidean.png")
    plt.savefig(p2)
    plt.close()

    # -------------------------------------------------------------
    # Plot 3: Score Distribution E2 (Cosine Similarity)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    gen_e2 = valid_e2[valid_e2["is_same"] == 1]["raw_score"].values
    imp_e2 = valid_e2[valid_e2["is_same"] == 0]["raw_score"].values
    plt.hist(gen_e2, bins=50, alpha=0.65, color="#27ae60", label="Genuine Pairs (Same Person)", density=True)
    plt.hist(imp_e2, bins=50, alpha=0.65, color="#c0392b", label="Impostor Pairs (Different Person)", density=True)
    plt.axvline(summary_e2["discrimination_metrics"]["global_eer_threshold"], color="#2c3e50", linestyle="--", lw=1.5,
                label=f"EER Operating Thresh ({summary_e2['discrimination_metrics']['global_eer_threshold']:.3f})")
    plt.xlabel("Cosine Similarity (Higher = More Similar)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("E2 (ArcFace 512D) Genuine vs Impostor Score Distributions", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "score_distribution_e2_cosine.png")
    plt.savefig(p3)
    plt.close()

    # -------------------------------------------------------------
    # Plot 4: Fold-wise ROC-AUC Comparison
    # -------------------------------------------------------------
    folds = sorted(list(summary_e1["folds"].keys()))
    e1_fold_aucs = [summary_e1["folds"][f]["roc_auc"] for f in folds]
    e2_fold_aucs = [summary_e2["folds"][f]["roc_auc"] for f in folds]

    x = np.arange(len(folds))
    width = 0.38

    plt.figure(figsize=(10, 5), dpi=150)
    plt.bar(x - width/2, e1_fold_aucs, width, label="E1 (dlib)", color="#e74c3c", alpha=0.85)
    plt.bar(x + width/2, e2_fold_aucs, width, label="E2 (ArcFace)", color="#2980b9", alpha=0.85)
    plt.xticks(x, [f"Fold {f}" for f in folds], fontsize=10)
    plt.ylim([0.7, 1.02])
    plt.ylabel("ROC-AUC", fontsize=11)
    plt.title("10-Fold Verification ROC-AUC: E1 (dlib) vs E2 (ArcFace)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "fold_wise_roc_auc.png")
    plt.savefig(p4)
    plt.close()

    # -------------------------------------------------------------
    # Plot 5: FAR vs FRR around EER Region (E2)
    # -------------------------------------------------------------
    fnr_e2 = 1.0 - tpr_e2
    plt.figure(figsize=(8, 5), dpi=150)
    mask = (thresh_e2 >= -0.2) & (thresh_e2 <= 0.8)
    plt.plot(thresh_e2[mask], fpr_e2[mask], color="#c0392b", lw=2, label="FAR (False Acceptance Rate)")
    plt.plot(thresh_e2[mask], fnr_e2[mask], color="#2980b9", lw=2, label="FRR (False Rejection Rate)")
    eer_val = summary_e2["discrimination_metrics"]["global_eer"]
    eer_th = summary_e2["discrimination_metrics"]["global_eer_threshold"]
    plt.plot([eer_th], [eer_val], marker="o", markersize=8, color="#27ae60",
             label=f"EER Point = {eer_val:.4f} at τ = {eer_th:.3f}")
    plt.axvline(eer_th, color="#7f8c8d", linestyle=":", lw=1)
    plt.xlabel("Cosine Similarity Threshold (τ)", fontsize=11)
    plt.ylabel("Error Rate", fontsize=11)
    plt.title("E2 (ArcFace) FAR vs FRR vs Decision Threshold", fontsize=12, fontweight="bold")
    plt.legend(loc="upper center", fontsize=10)
    plt.tight_layout()
    p5 = os.path.join(plots_dir, "far_frr_eer_curve.png")
    plt.savefig(p5)
    plt.close()

    print(f"[PLOTS] Generated 5 verification evaluation figures in: {plots_dir}", flush=True)


def run_full_verification_evaluation() -> Dict[str, Any]:
    """
    Executes the official 10-fold LFW verification benchmark comparing E1 vs E2.
    Uses 10-fold cross-validated threshold selection (train on 9 folds, evaluate on 1).
    Reuses cached embeddings for ultra-fast execution.
    """
    config = load_config("config/config.yaml")
    meta_dir = config.get("paths", {}).get("metadata_dir", "data/metadata")
    pairs_csv = os.path.join(meta_dir, "verification_pairs.csv")

    if not os.path.exists(pairs_csv):
        raise FileNotFoundError(f"Verification pairs file missing: {pairs_csv}")

    pairs_df = pd.read_csv(pairs_csv)
    reports_dir = "reports/evaluation"
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(reports_dir, exist_ok=True)

    print("="*78, flush=True)
    print("PHASE 6: OFFICIAL 10-FOLD LFW FACE VERIFICATION BENCHMARK", flush=True)
    print("="*78, flush=True)
    print(f"Dataset Protocol: Official 10-Fold Cross-Validation (6,000 pairs)", flush=True)
    print(f"Threshold Methodology: 9-Fold Training, 1-Fold Testing (Zero Leakage)", flush=True)
    print(f"Pairs File: {pairs_csv}", flush=True)
    print(f"Reports Directory: {reports_dir}\n", flush=True)

    # 1. Evaluate E1: dlib Baseline
    evaluator_e1 = VerificationEvaluator(experiment_id="E1", config=config, provisional_threshold=0.6)
    df_e1, summary_e1 = evaluator_e1.evaluate_pairs(pairs_df)

    # 2. Evaluate E2: Modern ArcFace
    evaluator_e2 = VerificationEvaluator(experiment_id="E2", config=config, provisional_threshold=0.45)
    df_e2, summary_e2 = evaluator_e2.evaluate_pairs(pairs_df)

    # 3. Generate Visualizations
    generate_evaluation_visualizations(df_e1, df_e2, summary_e1, summary_e2, plots_dir)

    # 4. Save Artifacts
    df_e1.to_csv(os.path.join(reports_dir, "pair_results_e1.csv"), index=False)
    df_e2.to_csv(os.path.join(reports_dir, "pair_results_e2.csv"), index=False)

    full_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "Official_LFW_10_Fold_Cross_Validation",
        "threshold_methodology": "9_fold_training_1_fold_test_cross_validation",
        "total_pairs": len(pairs_df),
        "experiment_e1": summary_e1,
        "experiment_e2": summary_e2
    }

    summary_path = os.path.join(reports_dir, "verification_summary.json")
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    # 5. Print Comprehensive Comparative Table
    print("\n" + "="*78, flush=True)
    print("OFFICIAL 10-FOLD LFW VERIFICATION BENCHMARK RESULTS: E1 vs E2", flush=True)
    print("="*78, flush=True)
    print(f"{'Evaluation Metric':<36} | {'E1: dlib Baseline':<18} | {'E2: ArcFace Modern':<18}", flush=True)
    print("-" * 78, flush=True)
    print(f"{'Fold-Calibrated Accuracy (Mean ± Std)':<36} | {summary_e1['fold_calibrated_metrics']['accuracy_mean']:.4f} ± {summary_e1['fold_calibrated_metrics']['accuracy_std']:.4f}  | {summary_e2['fold_calibrated_metrics']['accuracy_mean']:.4f} ± {summary_e2['fold_calibrated_metrics']['accuracy_std']:.4f}", flush=True)
    print(f"{'Fold-Calibrated FAR (Mean ± Std)':<36} | {summary_e1['fold_calibrated_metrics']['far_mean']:.4f} ± {summary_e1['fold_calibrated_metrics']['far_std']:.4f}  | {summary_e2['fold_calibrated_metrics']['far_mean']:.4f} ± {summary_e2['fold_calibrated_metrics']['far_std']:.4f}", flush=True)
    print(f"{'Fold-Calibrated FRR (Mean ± Std)':<36} | {summary_e1['fold_calibrated_metrics']['frr_mean']:.4f} ± {summary_e1['fold_calibrated_metrics']['frr_std']:.4f}  | {summary_e2['fold_calibrated_metrics']['frr_mean']:.4f} ± {summary_e2['fold_calibrated_metrics']['frr_std']:.4f}", flush=True)
    print(f"{'Fold-Calibrated Thresh (Mean ± Std)':<36} | {summary_e1['fold_calibrated_metrics']['threshold_mean']:.4f} ± {summary_e1['fold_calibrated_metrics']['threshold_std']:.4f}  | {summary_e2['fold_calibrated_metrics']['threshold_mean']:.4f} ± {summary_e2['fold_calibrated_metrics']['threshold_std']:.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"{'Global ROC-AUC':<36} | {summary_e1['discrimination_metrics']['global_roc_auc']:<18.4f} | {summary_e2['discrimination_metrics']['global_roc_auc']:<18.4f}", flush=True)
    print(f"{'Fold ROC-AUC (Mean ± Std)':<36} | {summary_e1['discrimination_metrics']['fold_roc_auc_mean']:.4f} ± {summary_e1['discrimination_metrics']['fold_roc_auc_std']:.4f}  | {summary_e2['discrimination_metrics']['fold_roc_auc_mean']:.4f} ± {summary_e2['discrimination_metrics']['fold_roc_auc_std']:.4f}", flush=True)
    print(f"{'Global Equal Error Rate (EER)':<36} | {summary_e1['discrimination_metrics']['global_eer']:<18.4f} | {summary_e2['discrimination_metrics']['global_eer']:<18.4f}", flush=True)
    print(f"{'Fold EER (Mean ± Std)':<36} | {summary_e1['discrimination_metrics']['fold_eer_mean']:.4f} ± {summary_e1['discrimination_metrics']['fold_eer_std']:.4f}  | {summary_e2['discrimination_metrics']['fold_eer_mean']:.4f} ± {summary_e2['discrimination_metrics']['fold_eer_std']:.4f}", flush=True)
    print(f"{'EER Operating Threshold':<36} | {summary_e1['discrimination_metrics']['global_eer_threshold']:<18.4f} | {summary_e2['discrimination_metrics']['global_eer_threshold']:<18.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"{'Provisional Pipeline Threshold':<36} | {summary_e1['provisional_threshold_reference']['provisional_threshold']:<18.4f} | {summary_e2['provisional_threshold_reference']['provisional_threshold']:<18.4f}", flush=True)
    print(f"{'  Provisional Accuracy':<36} | {summary_e1['provisional_threshold_reference']['accuracy']:<18.4f} | {summary_e2['provisional_threshold_reference']['accuracy']:<18.4f}", flush=True)
    print(f"{'  Provisional FAR':<36} | {summary_e1['provisional_threshold_reference']['far']:<18.4f} | {summary_e2['provisional_threshold_reference']['far']:<18.4f}", flush=True)
    print(f"{'  Provisional FRR':<36} | {summary_e1['provisional_threshold_reference']['frr']:<18.4f} | {summary_e2['provisional_threshold_reference']['frr']:<18.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"{'Evaluated Pairs (Reused Cache)':<36} | {summary_e1['successful_pairs']} / {summary_e1['total_pairs']:<11} | {summary_e2['successful_pairs']} / {summary_e2['total_pairs']:<11}", flush=True)
    print(f"{'Evaluation Execution Time (s)':<36} | {summary_e1['total_evaluation_time_seconds']:<18.2f} | {summary_e2['total_evaluation_time_seconds']:<18.2f}", flush=True)
    print("="*78, flush=True)

    return full_summary


if __name__ == "__main__":
    run_full_verification_evaluation()
