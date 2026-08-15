# Face Recognition Attendance & Analytics System

## Overview
This repository contains a modular Face Recognition Attendance System designed as an empirical Computer Vision and Machine Learning engineering project.

> **Note on Machine Learning Methodology:**
> All feature extraction models (Phase 1 dlib ResNet-34 128D and Phase 4 ArcFace ResNet-50 512D) utilize **pretrained neural network weights** for inference. Embedding extraction is feature transformation; it is **not** custom model training.
> **Note on Threshold Calibration:**
> The cosine similarity threshold (default `0.45`) used in this phase is **provisional** for pipeline verification. Final optimal recognition thresholds will be calibrated using validation data in Phase 7.

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
│   └── models/               # Downloaded ONNX model weights (YuNet, ArcFace)
├── config/
│   ├── __init__.py           # Config loader
│   └── config.yaml           # System paths, model parameters, and thresholds
├── data/                     # Dataset storage (Gitignored raw/eval images & galleries)
│   ├── raw/lfw/              # Downloaded raw LFW dataset
│   ├── evaluation/           # Partitioned evaluation images
│   │   ├── enrollment/       # Reference gallery templates (59 identities, 118 images)
│   │   ├── validation/       # Validation set for threshold tuning
│   │   └── test/             # Final hold-out test set
│   ├── embeddings/           # Serialized IdentityGallery artifacts (arcface_gallery.npz)
│   └── metadata/             # Versioned split and verification metadata
│       ├── identities.csv    # List of selected evaluation identities
│       ├── splits.csv        # Image-level split mapping and SHA256 hashes
│       ├── verification_pairs.csv # Official LFW 10-fold pairs
│       └── dataset_summary.json   # Full dataset audit summary
├── scripts/
│   ├── prepare_dataset.py                 # LFW acquisition & split partitioning
│   ├── validate_dataset.py                # Leakage, hash, and integrity audit
│   ├── benchmark_detection_alignment.py   # Detection & alignment benchmark
│   ├── benchmark_embeddings.py            # ArcFace embedding sanity checks & benchmark
│   ├── build_gallery.py                   # Enrolls reference identities into gallery
│   ├── evaluate_identification_pipeline.py# Open-set identification pipeline verification
│   ├── generate_baseline_embeddings.py    # Offline baseline feature extraction
│   └── verify_baseline.py                 # Manual & API verification script
├── tests/                    # PyTest test suite (54 tests)
│   ├── test_aligner.py       # 5-point alignment unit tests
│   ├── test_dataset.py       # Dataset partitioning and leakage tests
│   ├── test_detector.py      # Face detector unit tests (Dlib & Modern)
│   ├── test_embedder.py      # Embedding extractor unit tests (Dlib & ArcFace)
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

## 2. Modern Face Recognition Pipeline (Phase 5 — Experiment E2)

### End-to-End Recognition Workflow
```
Input Image (RGB)
      │
      ▼
YuNet Face Detection (OpenCV Deep CNN)
      │
      ▼
Deterministic Primary Face Selection (Highest Confidence Policy)
      │
      ▼
5-Point Facial Landmark Alignment (112x112 Canonical ArcFace Crop)
      │
      ▼
ArcFace Embedding Extraction (512D ResNet-50 Backbone)
      │
      ▼
L2 Hyperspherical Normalization (||e||_2 = 1.0)
      │
      ▼
IdentityGallery Multi-Template Cosine Similarity Search
      │
      ▼
Open-Set Decision Thresholding (Provisional τ = 0.45)
      │
      ├── If max(CosineSim) >= τ ──► Recognized Enrolled Identity
      └── If max(CosineSim) < τ  ──► Rejected as "Unknown"
```

### Enrolled Identity Gallery (`IdentityGallery`)
* **Source Split**: Constructed exclusively from the Phase 2 `enrollment/` split (59 identities, 118 reference images, 2 images per person).
* **Multi-Template Matching**:
  $$S_{\text{identity}} = \max_{j \in \text{templates}} (\hat{\mathbf{q}} \cdot \hat{\mathbf{k}}_j)$$
  $$\text{Top-1 Candidate} = \arg\max_{\text{identity}} (S_{\text{identity}})$$
* **Open-Set Decision**:
  $$\text{Decision} = \begin{cases} \text{Top-1 Candidate}, & \text{if } S_{\text{Top-1}} \ge \tau \\ \text{Unknown (None)}, & \text{otherwise} \end{cases}$$

---

## 3. Experiment Configuration Profiles

| Feature | Experiment E1 (Baseline) | Experiment E2 (Modern Recognition) |
|---|---|---|
| **Detector** | dlib HOG + Linear SVM | OpenCV YuNet Deep CNN |
| **Preprocessing** | Bounding box crop | 5-Point Affine Landmark Alignment ($112 \times 112$) |
| **Embedding Model** | Pretrained dlib ResNet-34 | **Pretrained ArcFace ResNet-50** |
| **Embedding Dimension** | 128D | **512D** |
| **Vector Space** | Euclidean Space | **Unit Hyperspherical Manifold ($\|\mathbf{e}\|_2 = 1.0$)** |
| **Gallery Search** | Euclidean Distance ($\|e_1 - e_2\|_2$) | **Multi-Template Cosine Similarity ($\mathbf{e}_1 \cdot \mathbf{e}_2$)** |
| **Decision Rule** | $\min(d) \le 0.6$ | $\max(S) \ge \tau_{\text{cosine}}$ |
| **Open-Set Rejection** | Distance threshold | **Cosine similarity threshold** |

---

## 4. Measured Recognition Pipeline Performance (CPU)

Evaluated over 50 Known validation queries and 50 Open-Set Unknown queries (`scripts/evaluate_identification_pipeline.py`):
* **Known Queries**:
  - Correct Top-1 Candidate: **96.00%** (48 / 50)
  - Accepted at Provisional Threshold (`0.45`): **98.00%** (49 / 50)
  - Average Known Similarity Score: **0.6971**
* **Unknown Queries (Open-Set Rejection)**:
  - Rejected as Unknown: **100.00%** (50 / 50)
  - Average Unknown Similarity Score: **0.1557**
  - **Score Margin Separation**: **+0.5414**
* **Per-Query Latency Breakdown**:
  - YuNet Detection: **4.18 ms**
  - 5-Point Alignment: **0.25 ms**
  - ArcFace Embedding (ResNet-50): **94.88 ms**
  - Exact Gallery Search (118 templates): **0.21 ms**
  - **Total End-to-End Latency**: **99.53 ms** ($\approx \mathbf{10.0\text{ FPS}}$ on CPU)

---

## 5. Setup & Execution Commands

### 1. Build Enrolled Identity Gallery (E2)
```bash
python scripts/build_gallery.py
```

### 2. Run Open-Set Identification Verification
```bash
python scripts/evaluate_identification_pipeline.py
```

### 3. Run Complete PyTest Suite (54 tests)
```bash
pytest -v
```

### 4. Start Flask Web Application
```bash
python app.py
```
Access the application in your browser at `http://127.0.0.1:5000/`.
