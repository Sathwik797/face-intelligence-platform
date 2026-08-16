# Phase 6 Face Verification Evaluation Artifacts

## Overview
This directory contains experimental evaluation artifacts from the Phase 6 formal 1:1 face verification benchmark conducted on the official Labeled Faces in the Wild (LFW) dataset.

---

## 1. Evaluation Benchmark Protocol

* **Benchmark Dataset**: Official LFW Face Verification Dataset (6,000 pairs partitioned across 10 non-overlapping folds).
* **Fold Protocol**: 10 folds $\times$ 600 pairs per fold (300 genuine pairs + 300 impostor pairs).
* **Threshold Selection Methodology**: Official 10-fold cross-validation. For each test fold $k$, the decision threshold is determined solely on the remaining 9 folds (5,400 training pairs) by maximizing training verification accuracy, and evaluated strictly on the 10th held-out fold (600 test pairs) with zero data leakage.
* **Evaluated Pipelines**:
  - **Experiment E1 (dlib Baseline)**: `dlib HOG Face Detector` + `dlib 128D Embedder` + `Euclidean Distance Matcher` (lower distance indicates greater similarity).
  - **Experiment E2 (ArcFace Modern)**: `OpenCV YuNet Detector` + `5-Point Similarity Aligner` + `ArcFace ResNet-50 512D Embedder` + `Cosine Similarity Matcher` (higher similarity indicates greater similarity).

---

## 2. Directory Contents

### Tracked Evidence Files
* [`verification_summary.json`](./verification_summary.json): Complete machine-readable summary containing global metrics, 10-fold cross-validated aggregates ($\text{Mean} \pm \text{Std}$), fold-by-fold breakdown, and score distribution statistics for both experiments.
* `plots/`: Visual evaluation figures:
  - `roc_curve_e1_vs_e2.png`: Receiver Operating Characteristic (ROC) curves comparing E1 vs E2.
  - `score_distribution_e1_euclidean.png`: Histogram of genuine vs impostor Euclidean distance score distributions for E1.
  - `score_distribution_e2_cosine.png`: Histogram of genuine vs impostor Cosine similarity score distributions for E2.
  - `fold_wise_roc_auc.png`: Fold-by-fold ROC-AUC comparison across all 10 folds.
  - `far_frr_eer_curve.png`: FAR vs FRR vs Decision Threshold curve illustrating the Equal Error Rate (EER) operating point for E2.

### Intentionally Excluded Artifacts (Gitignored)
* `cache/` (`e1_embeddings.npz`, `e2_embeddings.npz`): Precomputed 128D and 512D biometric embedding vectors across all 7,701 unique face images. These are generated biometric-derived representations excluded for privacy and repository cleanliness.
* `pair_results_e1.csv`, `pair_results_e2.csv`: Raw pair-by-pair inference score tables across the 6,000 pairs. These are intermediate reproducible outputs excluded from version control.

---

## 3. Summary of Empirical Results

| Evaluation Metric | E1: dlib Baseline | E2: ArcFace Modern |
|---|---|---|
| **Fold-Calibrated Accuracy ($\text{Mean} \pm \text{Std}$)** | **$97.43\% \pm 0.60\%$** | **$98.50\% \pm 0.72\%$** |
| **Fold-Calibrated FAR ($\text{Mean} \pm \text{Std}$)** | **$1.53\% \pm 0.54\%$** | **$0.03\% \pm 0.10\%$** |
| **Fold-Calibrated FRR ($\text{Mean} \pm \text{Std}$)** | **$3.60\% \pm 1.38\%$** | **$2.97\% \pm 1.40\%$** |
| **Fold-Calibrated Threshold ($\text{Mean} \pm \text{Std}$)** | **$0.6345 \pm 0.0022$** (Euclidean) | **$0.2426 \pm 0.0045$** (Cosine) |
| **Global ROC-AUC** | **0.9941** | **0.9883** |
| **Fold ROC-AUC ($\text{Mean} \pm \text{Std}$)** | **$0.9941 \pm 0.0029$** | **$0.9883 \pm 0.0078$** |
| **Global Equal Error Rate (EER)** | **0.0293** ($2.93\%$) | **0.0268** ($2.68\%$) |
| **Fold EER ($\text{Mean} \pm \text{Std}$)** | **$0.0290 \pm 0.0104$** | **$0.0220 \pm 0.0174$** |
| **Global EER Operating Threshold** | $0.6557$ (Euclidean) | $0.1160$ (Cosine) |
| **Evaluated Pairs** | **6,000 / 6,000 (100.0%)** | **6,000 / 6,000 (100.0%)** |

> **Note on Threshold Terminology:**
> - **Fold-Calibrated Threshold**: Selected per fold from the 9 training folds to measure generalization on the 10th held-out fold.
> - **EER Operating Threshold**: Diagnostic operating point where $\text{FAR} = \text{FRR}$; it does not represent the production decision threshold.
> - **Production Threshold Calibration**: Formal calibration for open-set identification will be performed on the independent validation split in Phase 7.
