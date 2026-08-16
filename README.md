# Face Recognition Attendance & Analytics System

## Overview
This repository contains a modular Face Recognition Attendance System designed as an empirical Computer Vision and Machine Learning engineering project.

> **Note on Machine Learning Methodology:**
> All feature extraction models (Phase 1 dlib ResNet-34 128D and Phase 4 ArcFace ResNet-50 512D) utilize **pretrained neural network weights** for inference. Embedding extraction is feature transformation; it is **not** custom model training.
> **Note on Evaluation Protocol & Threshold Calibration:**
> The official 10-fold verification benchmark implements standard leave-one-fold-out threshold selection (training threshold on 9 folds, testing on 1 held-out fold with zero data leakage). Final production decision thresholds for open-set identification will be systematically calibrated on the project's validation split in Phase 7.

---

## 1. Project Architecture

```
face_recognition_attendence_system/
├── app/
│   ├── __init__.py           # Flask application factory
│   └── routes.py             # API endpoints (/ and /recognize)
├── ml/
│   ├── __init__.py           # Package exports
│   ├── detector.py           # BaseDetector, DlibHOGDetector & ModernFaceDetector
│   ├── aligner.py            # FaceAligner (5-point affine transformation)
│   ├── embedder.py           # BaseEmbedder, DlibEmbedder (128D) & ArcFaceEmbedder (512D)
│   ├── matcher.py            # BaseMatcher, EuclideanMatcher & CosineMatcher
│   ├── gallery.py            # IdentityGallery (multi-template enrollment & search)
│   ├── pipeline.py           # Baseline (E1) & Modern (E2) Recognition Pipelines
│   ├── evaluation/           # Formal verification & ML metrics framework
│   │   ├── __init__.py       # Metrics and evaluator exports
│   │   ├── metrics.py        # ROC, AUC, EER, FAR, FRR & 10-fold cross-validation
│   │   └── evaluator.py      # 10-fold verification evaluator with cached embeddings
│   └── models/               # Downloaded ONNX model weights (YuNet, ArcFace)
├── config/
│   ├── __init__.py           # Config loader
│   └── config.yaml           # System paths, model parameters, and thresholds
├── data/                     # Dataset storage (Gitignored raw/eval images & galleries)
│   ├── raw/lfw/              # Full downloaded LFW dataset (5,760 identities)
│   ├── evaluation/           # Partitioned evaluation images
│   │   ├── enrollment/       # Reference gallery templates (59 identities, 118 images)
│   │   ├── validation/       # Validation set for threshold tuning
│   │   └── test/             # Final hold-out test set
│   ├── embeddings/           # Serialized IdentityGallery artifacts (arcface_gallery.npz)
│   └── metadata/             # Versioned split and verification metadata
│       ├── identities.csv    # List of selected evaluation identities
│       ├── splits.csv        # Image-level split mapping and SHA256 hashes
│       ├── verification_pairs.csv # Official LFW 10-fold pairs (6,000 pairs)
│       └── dataset_summary.json   # Full dataset audit summary
├── reports/                  # Generated experiment reports & visualizations
│   └── evaluation/           # 10-fold verification summary JSON & plots
│       ├── README.md         # Detailed Phase 6 evaluation artifact documentation
│       ├── verification_summary.json # Machine-readable 10-fold evaluation metrics
│       └── plots/            # ROC curves, score distributions, EER curves
├── scripts/
│   ├── prepare_dataset.py                 # LFW acquisition & split partitioning
│   ├── validate_dataset.py                # Leakage, hash, and integrity audit
│   ├── benchmark_detection_alignment.py   # Detection & alignment benchmark
│   ├── benchmark_embeddings.py            # ArcFace embedding sanity checks & benchmark
│   ├── build_gallery.py                   # Enrolls reference identities into gallery
│   ├── evaluate_identification_pipeline.py# Open-set identification pipeline verification
│   ├── evaluate_verification.py           # Formal 10-fold LFW verification benchmark
│   ├── generate_baseline_embeddings.py    # Offline baseline feature extraction
│   └── verify_baseline.py                 # Manual & API verification script
├── tests/                    # PyTest test suite (64 tests)
│   ├── test_aligner.py       # 5-point alignment unit tests
│   ├── test_dataset.py       # Dataset partitioning and leakage tests
│   ├── test_detector.py      # Face detector unit tests (Dlib & Modern)
│   ├── test_embedder.py      # Embedding extractor unit tests (Dlib & ArcFace)
│   ├── test_evaluation_metrics.py # ROC, AUC, EER, FAR/FRR & fold-aware metric tests
│   ├── test_gallery.py       # IdentityGallery multi-template & search tests
│   ├── test_matcher.py       # Similarity matcher unit tests (Euclidean & Cosine)
│   ├── test_pipeline.py      # Baseline recognition pipeline tests
│   ├── test_recognition_pipeline.py # Modern recognition pipeline tests
│   └── test_api.py           # Web API integration tests
├── templates/                # Frontend HTML views
├── static/                   # Frontend CSS
├── requirements.txt          # Python dependencies
├── app.py                    # Application entrypoint
└── README.md                 # Documentation
```

---

## 2. Official 10-Fold LFW Face Verification Benchmark (Phase 6)

### Evaluation Protocol
* **Dataset**: Official LFW Verification Benchmark (6,000 pairs partitioned across 10 non-overlapping folds).
* **Composition per Fold**: Exactly 300 genuine pairs (same person) + 300 impostor pairs (different person) = 600 pairs per fold.
* **Threshold Selection Methodology**: Standard leave-one-fold-out cross-validation. For each test fold $k$, the decision threshold is determined solely on the remaining 9 folds (5,400 training pairs) by maximizing training accuracy, and evaluated strictly on the 10th held-out fold (600 test pairs) with zero test-set leakage.
* **Score Directions**:
  - **Experiment E1 (dlib)**: Euclidean distance $d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2$ (smaller distance $\implies$ more similar).
  - **Experiment E2 (ArcFace)**: Cosine similarity $S(\hat{\mathbf{u}}, \hat{\mathbf{v}}) = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$ (larger similarity $\implies$ more similar).

### Empirical 10-Fold Verification Results (E1 vs E2)

| Metric | Experiment E1 (dlib Baseline) | Experiment E2 (ArcFace Modern) |
|---|---|---|
| **Fold-Calibrated Accuracy ($\text{Mean} \pm \text{Std}$)** | **$97.43\% \pm 0.60\%$** ($0.9743 \pm 0.0060$) | **$98.50\% \pm 0.72\%$** ($0.9850 \pm 0.0072$) |
| **Fold-Calibrated FAR ($\text{Mean} \pm \text{Std}$)** | **$1.53\% \pm 0.54\%$** ($0.0153 \pm 0.0054$) | **$0.03\% \pm 0.10\%$** ($0.0003 \pm 0.0010$) |
| **Fold-Calibrated FRR ($\text{Mean} \pm \text{Std}$)** | **$3.60\% \pm 1.38\%$** ($0.0360 \pm 0.0138$) | **$2.97\% \pm 1.40\%$** ($0.0297 \pm 0.0140$) |
| **Fold-Calibrated Threshold ($\text{Mean} \pm \text{Std}$)** | **$0.6345 \pm 0.0022$** (Euclidean) | **$0.2426 \pm 0.0045$** (Cosine) |
| **Global ROC-AUC** | **0.9941** | **0.9883** |
| **Fold ROC-AUC ($\text{Mean} \pm \text{Std}$)** | **$0.9941 \pm 0.0029$** | **$0.9883 \pm 0.0078$** |
| **Global Equal Error Rate (EER)** | **0.0293** ($2.93\%$) | **0.0268** ($2.68\%$) |
| **Fold EER ($\text{Mean} \pm \text{Std}$)** | **$0.0290 \pm 0.0104$** | **$0.0220 \pm 0.0174$** |
| **Global EER Operating Threshold** | $0.6557$ (Euclidean) | $0.1160$ (Cosine) |
| **Evaluated Pairs (Reused Cache)** | **6,000 / 6,000 (100.0%)** | **6,000 / 6,000 (100.0%)** |

### ML Evaluation Evidence
Full machine-readable evaluation summaries, distribution metrics, and generated visualization plots are preserved in [`reports/evaluation/`](./reports/evaluation/README.md):
* [`reports/evaluation/verification_summary.json`](./reports/evaluation/verification_summary.json): Complete 10-fold verification benchmark metrics.
* `reports/evaluation/plots/roc_curve_e1_vs_e2.png`: 10-Fold ROC curves.
* `reports/evaluation/plots/score_distribution_e1_euclidean.png`: E1 genuine vs impostor Euclidean distance histogram.
* `reports/evaluation/plots/score_distribution_e2_cosine.png`: E2 genuine vs impostor Cosine similarity histogram.
* `reports/evaluation/plots/fold_wise_roc_auc.png`: Fold-wise ROC-AUC comparison bar chart.
* `reports/evaluation/plots/far_frr_eer_curve.png`: FAR vs FRR vs Decision Threshold operating curve for E2.

---

## 3. Threshold Definitions & Distinctions

1. **Fold-Calibrated Evaluation Threshold**:
   - Selected per fold using the other 9 training folds to evaluate unbiased generalization on the held-out test fold.
   - E1 Mean: $\tau = 0.6345$ (Euclidean); E2 Mean: $\tau = 0.2426$ (Cosine).
2. **EER Operating Threshold**:
   - The threshold point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR).
   - E1 Global: $\tau = 0.6557$; E2 Global: $\tau = 0.1160$.
   - *Note*: EER is a diagnostic operating point metric and does not represent the calibrated production decision threshold.
3. **Provisional Pipeline Threshold**:
   - Fixed heuristic threshold used in earlier pipeline stages (E1: $0.60$, E2: $0.45$).
   - *Note*: Formal production threshold calibration on the independent validation split will be conducted in Phase 7.

---

## 4. Setup & Execution Commands

### 1. Run 10-Fold LFW Face Verification Benchmark
```bash
python scripts/evaluate_verification.py
```

### 2. Run Complete PyTest Suite (64 tests)
```bash
pytest -v
```

### 3. Start Flask Web Application
```bash
python app.py
```
Access the application in your browser at `http://127.0.0.1:5000/`.
