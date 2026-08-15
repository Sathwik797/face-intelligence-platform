# Face Recognition Attendance & Analytics System

## Overview
This repository contains a modular Face Recognition Attendance System designed as an empirical Computer Vision and Machine Learning engineering project.

> **Note on Machine Learning Methodology:**
> All feature extraction models (Phase 1 dlib ResNet-34 128D and Phase 4 ArcFace ResNet-50 512D) utilize **pretrained neural network weights** for inference. Embedding extraction is feature transformation; it is **not** custom model training.

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
│   ├── pipeline.py           # FaceRecognitionPipeline & RecognitionResult
│   └── models/               # Downloaded ONNX model weights (YuNet, ArcFace)
├── config/
│   ├── __init__.py           # Config loader
│   └── config.yaml           # System paths, model parameters, and thresholds
├── data/                     # Dataset storage (Gitignored raw/eval images)
│   ├── raw/lfw/              # Downloaded raw LFW dataset
│   ├── evaluation/           # Partitioned evaluation images
│   │   ├── enrollment/       # Reference gallery templates
│   │   ├── validation/       # Validation set for threshold tuning
│   │   └── test/             # Final hold-out test set
│   └── metadata/             # Versioned split and verification metadata
│       ├── identities.csv    # List of selected evaluation identities
│       ├── splits.csv        # Image-level split mapping and SHA256 hashes
│       ├── verification_pairs.csv # Official LFW 10-fold pairs
│       └── dataset_summary.json   # Full dataset audit summary
├── scripts/
│   ├── prepare_dataset.py               # LFW acquisition & split partitioning
│   ├── validate_dataset.py              # Leakage, hash, and integrity audit
│   ├── benchmark_detection_alignment.py # Detection & alignment benchmark
│   ├── benchmark_embeddings.py          # ArcFace embedding sanity checks & benchmark
│   ├── generate_baseline_embeddings.py  # Offline baseline feature extraction
│   └── verify_baseline.py               # Manual & API verification script
├── tests/                    # PyTest test suite (44 tests)
│   ├── test_aligner.py       # 5-point alignment unit tests
│   ├── test_dataset.py       # Dataset partitioning and leakage tests
│   ├── test_detector.py      # Face detector unit tests (Dlib & Modern)
│   ├── test_embedder.py      # Embedding extractor unit tests (Dlib & ArcFace)
│   ├── test_matcher.py       # Similarity matcher unit tests (Euclidean & Cosine)
│   ├── test_pipeline.py      # Recognition pipeline tests
│   └── test_api.py           # Web API integration tests
├── templates/                # Frontend HTML views
├── static/                   # Frontend CSS
├── requirements.txt          # Python dependencies
├── app.py                    # Application entrypoint
└── README.md                 # Documentation
```

---

## 2. Pretrained ArcFace Embedding Pipeline (Phase 4)

### Modern Embedding Backbone (`ArcFaceEmbedder`)
* **Architecture**: Deep ResNet-50 backbone with Additive Angular Margin Loss (ArcFace).
* **Source & Checkpoint**: InsightFace `buffalo_l` (`w600k_r50.onnx`, trained on WebFace600k / Glint360k).
* **Input Preprocessing**:
  1. Aligned RGB face crop of resolution $112 \times 112 \times 3$.
  2. Numerical pixel normalization: $\mathbf{x}_{\text{norm}} = (\mathbf{x} - 127.5) / 127.5 \in [-1.0, 1.0]$ (`float32`).
  3. Channel reordering: $(H, W, C) \to (C, H, W)$.
  4. Batch tensor: $(N, 3, 112, 112)$.
* **Embedding Dimensionality**: **512-dimensional vector**.
* **$L_2$ Normalization**: Strict hyperspherical projection:
  $$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\max(\|\mathbf{e}\|_2, 10^{-10})}, \quad \|\hat{\mathbf{e}}\|_2 = 1.0000$$
* **Matching Metric**: Cosine similarity via inner product:
  $$S(\hat{\mathbf{q}}, \hat{\mathbf{k}}) = \hat{\mathbf{q}} \cdot \hat{\mathbf{k}} \in [-1.0, 1.0]$$

---

## 3. Experiment Configuration Profiles

| Feature | Experiment E1 (Baseline) | Experiment E2 (Modern Embedding) |
|---|---|---|
| **Detector** | dlib HOG + Linear SVM | OpenCV YuNet Deep CNN |
| **Preprocessing** | Bounding box crop | 5-Point Affine Landmark Alignment ($112 \times 112$) |
| **Embedding Model** | dlib ResNet-34 (Pretrained) | ArcFace ResNet-50 (Pretrained, w600k) |
| **Embedding Dimension** | 128D | **512D** |
| **Vector Normalization** | $L_2$ Euclidean Space | **Unit $L_2$ Hyperspherical Normalization** |
| **Similarity Metric** | Euclidean Distance ($\|e_1 - e_2\|_2$) | **Cosine Similarity ($\mathbf{e}_1 \cdot \mathbf{e}_2$)** |
| **Decision Rule** | $\min(d) \le \tau_{\text{dist}}$ | $\max(S) \ge \tau_{\text{cosine}}$ |

---

## 4. Measured Embedding Benchmark Results

Evaluated over 300 representative LFW validation images (`scripts/benchmark_embeddings.py`):
* **Model Load Time**: **347.45 ms**
* **Sanity Checks Passed**:
  - [x] Output Dimension == 512
  - [x] Finite Values (No NaN / Inf)
  - [x] Unit $L_2$ Norm ($\|\mathbf{e}\|_2 = 1.000000$)
  - [x] Deterministic on Identical Inputs
  - [x] Discriminability on Distinct Faces (Cosine Sim: $0.0435$)
  - [x] Single vs Batch Inference Consistency ($< 10^{-5}$ tolerance)
* **Single-Crop Inference Latency (CPU)**: **113.85 ms** / face
* **Batch Inference Latency (CPU, Batch Size 16)**: **117.87 ms** / face

---

## 5. Setup & Execution Commands

### 1. Run Embedding Benchmark & Sanity Checks
```bash
python scripts/benchmark_embeddings.py
```

### 2. Run Complete PyTest Suite (44 tests)
```bash
pytest -v
```

### 3. Start Flask Web Application
```bash
python app.py
```
Access the application in your browser at `http://127.0.0.1:5000/`.
