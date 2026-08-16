# Phase 7 Production Threshold Calibration

## Overview
This directory contains experimental artifacts from the Phase 7 formal production threshold calibration for the modern ArcFace (E2) pipeline, conducted strictly on the project's **independent validation split** (`data/evaluation/validation/`, 59 identities, 1,395 images).

---

## 1. Calibration Methodology & Data Split

* **Evaluation Split**: Custom 59-identity validation partition (1,395 images).
* **Test Set Protection**: Strict zero-access protocol. The final hold-out test set (`data/evaluation/test/`, 1,423 images) was never accessed or evaluated.
* **Separation from LFW**: Phase 6 evaluated general verification capability on the official 10-fold LFW dataset. Phase 7 selects the decision threshold for actual production system deployment on the project's multi-sample validation identities.
* **Evaluation Pairs**: Generated 6,565 genuine pairs and 50,000 impostor pairs (56,565 total pairs).
* **Score Metric**: Cosine similarity $S \in [-1.0, 1.0]$. Match decision: $S \ge \tau \implies \text{Genuine}$.

---

## 2. Directory Contents

### Tracked Evidence Files
* [`calibration_summary.json`](./calibration_summary.json): Complete machine-readable summary containing distribution percentiles, all five calibration strategy metrics, 5-fold cross-validated stability, and the recommended production threshold.
* `plots/`: Visual calibration evidence:
  - `far_vs_threshold.png`: False Acceptance Rate (FAR) vs Cosine Threshold.
  - `frr_vs_threshold.png`: False Rejection Rate (FRR) vs Cosine Threshold.
  - `far_frr_vs_threshold.png`: Combined FAR and FRR curves highlighting the EER operating point.
  - `accuracy_vs_threshold.png`: Validation accuracy curve highlighting the Maximum Accuracy point.
  - `f1_vs_threshold.png`: F1-Score curve highlighting the Optimal F1 point.
  - `validation_score_distributions.png`: Validation genuine vs impostor Cosine similarity distributions.
  - `operating_points_comparison.png`: ROC operating space visualizing all five candidate operating points.

### Excluded Artifacts (Gitignored)
* `cache/` (`validation_embeddings.npz`): Precomputed 512D validation embeddings (1,395 vectors). Excluded for privacy and repository cleanliness.

---

## 3. Calibration Strategies Summary

| Strategy | Threshold ($\tau$) | FAR (%) | FRR (%) | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|---|---|---|
| **A. Equal Error Rate (EER)** | $0.1280$ | $2.340\%$ | $2.346\%$ | $97.66\%$ | $84.57\%$ | $97.65\%$ | $90.64\%$ |
| **B. Maximum Accuracy** | $0.2800$ | $0.004\%$ | $2.498\%$ | $99.71\%$ | $99.97\%$ | $97.50\%$ | $98.72\%$ |
| **C. Security Low-FAR (0.1% target)** | $\mathbf{0.2400}$ | $\mathbf{0.042\%}$ | $\mathbf{2.376\%}$ | $\mathbf{99.69\%}$ | $\mathbf{99.67\%}$ | $\mathbf{97.62\%}$ | $\mathbf{98.64\%}$ |
| **D. F1-Optimal** | $0.2840$ | $0.002\%$ | $2.498\%$ | $99.71\%$ | $99.98\%$ | $97.50\%$ | $98.73\%$ |
| **E. FAR/FRR-Balanced** | $0.1280$ | $2.340\%$ | $2.346\%$ | $97.66\%$ | $84.57\%$ | $97.65\%$ | $90.64\%$ |

---

## 4. Score Distribution Percentiles

* **Genuine Similarity Distribution ($N = 6,565$)**:
  - Mean: $0.6418 \pm 0.1334$, Median: $0.6656$, Range: $[-0.1151, 0.9549]$
  - Percentiles: $P_1 = 0.0056, P_5 = 0.4483, P_{25} = 0.6000, P_{50} = 0.6656, P_{75} = 0.7180, P_{95} = 0.7854, P_{99} = 0.8259$
* **Impostor Similarity Distribution ($N = 50,000$)**:
  - Mean: $0.0061 \pm 0.0596$, Median: $0.0048$, Range: $[-0.2291, 0.2851]$
  - Percentiles: $P_1 = -0.1278, P_5 = -0.0901, P_{25} = -0.0343, P_{50} = 0.0048, P_{75} = 0.0451, P_{95} = 0.1048, P_{99} = 0.1535$

---

## 5. Recommended Production Threshold

* **Recommended Threshold**: **$\tau = 0.2400$** (Security-Oriented Low-FAR Strategy)
* **Expected Validation FAR**: **$0.042\%$** ($< 1$ false accept per 2,000 impostor attempts)
* **Expected Validation FRR**: **$2.38\%$** ($> 97.6\%$ genuine recognition recall)
* **Expected Validation Accuracy**: **$99.69\%$**
* **Technical Justification**: In identity verification and attendance logging, false acceptance represents a critical security failure (impostor falsely verified). The selected threshold of $0.2400$ guarantees near-zero false acceptance ($\text{FAR} = 0.042\%$) while preserving high genuine convenience ($\text{FRR} = 2.38\%$).
