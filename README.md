# Face Recognition Attendance & Analytics System

## Overview
This repository contains a modular Face Recognition Attendance System designed as an empirical Computer Vision and Machine Learning engineering project.

> **Note on Machine Learning Methodology:**
> All feature extraction models (Phase 1 dlib ResNet-34 128D and Phase 4 ArcFace ResNet-50 512D) utilize **pretrained neural network weights** for inference. Embedding extraction is feature transformation; it is **not** custom model training.
> **Note on Evaluation Protocol & Threshold Calibration:**
> - Phase 6 benchmarked verification performance on the official 10-fold LFW dataset.
> - Phase 7 calibrated the production decision threshold on the project's **independent validation split** (59 identities, 1,395 images), strictly protecting the final hold-out test set from any access.
> - Phase 8 established the **Face Quality Assessment (FQA)** subsystem, evaluating visual and geometric quality signals before feature extraction.
> - Phase 9 established the **Temporal Identity Stabilization** layer to aggregate multi-frame evidence, suppress identity flicker, and absorb transient dropouts.
> - Phase 10 established the in-memory **Presence & Session Intelligence** state machine to manage presence lifecycles and clean session boundaries.
> - Phase 11 established the **System Integration & Runtime Orchestration** subsystem composing all intelligence layers into an end-to-end fault-tolerant processing engine.

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
│   ├── quality/              # Face Quality Assessment (FQA) subsystem
│   │   ├── __init__.py       # Quality package exports
│   │   ├── schemas.py        # FaceQualityMetrics, QualityThresholds, QualityMode
│   │   ├── metrics.py        # Laplacian blur, brightness, contrast, alignment/pose proxies
│   │   └── assessor.py       # FaceQualityAssessor with Strict/Balanced/Lenient modes
│   ├── temporal/             # Temporal Recognition & Identity Stability subsystem
│   │   ├── __init__.py       # Temporal package exports
│   │   ├── schemas.py        # RecognitionObservation, TemporalRecognitionResult, TemporalPolicyConfig
│   │   └── stabilizer.py     # TemporalIdentityStabilizer (Fast/Balanced/Stable modes)
│   ├── presence/             # Presence & Session Intelligence subsystem
│   │   ├── __init__.py       # Presence package exports
│   │   ├── schemas.py        # PresenceState, PresenceSession, PresenceEvent, PresenceConfig
│   │   ├── state_machine.py  # IdentityPresenceStateMachine (State machine lifecycle)
│   │   └── manager.py        # PresenceManager (Multi-identity orchestrator)
│   ├── runtime/              # System Integration & Runtime Orchestration subsystem
│   │   ├── __init__.py       # Runtime package exports
│   │   ├── schemas.py        # RuntimeStatus, RuntimeConfig, StageLatencyMetrics, RuntimeFrameResult
│   │   ├── frame_source.py   # BaseFrameSource, StaticFrameSource, SyntheticFrameSource, OpenCVFrameSource
│   │   └── orchestrator.py   # FaceIntelligenceRuntime (Lifecycle, stage execution & telemetry)
│   ├── pipeline.py           # Baseline (E1) & Modern (E2) Recognition Pipelines
│   ├── evaluation/           # Verification & threshold calibration framework
│   │   ├── __init__.py       # Exports
│   │   ├── metrics.py        # ROC, AUC, EER, FAR, FRR & 10-fold cross-validation
│   │   ├── evaluator.py      # 10-fold verification benchmark evaluator
│   │   └── calibrator.py     # Production threshold calibrator (validation split)
│   └── models/               # Downloaded ONNX model weights (YuNet, ArcFace)
├── config/
│   ├── __init__.py           # Config loader
│   └── config.yaml           # System paths, model parameters, quality, temporal, presence, runtime, thresholds
├── data/                     # Dataset storage (Gitignored raw/eval images & galleries)
│   ├── raw/lfw/              # Full downloaded LFW dataset (5,760 identities)
│   ├── evaluation/           # Partitioned evaluation images
│   │   ├── enrollment/       # Reference gallery templates (59 identities, 118 images)
│   │   ├── validation/       # Validation set for threshold tuning (59 identities, 1,395 images)
│   │   └── test/             # Final hold-out test set (Protected)
│   ├── embeddings/           # Serialized IdentityGallery artifacts (arcface_gallery.npz)
│   └── metadata/             # Versioned split and verification metadata
│       ├── identities.csv    # List of selected evaluation identities
│       ├── splits.csv        # Image-level split mapping and SHA256 hashes
│       ├── verification_pairs.csv # Official LFW 10-fold pairs (6,000 pairs)
│       └── dataset_summary.json   # Full dataset audit summary
├── reports/                  # Generated experiment reports & visualizations
│   ├── evaluation/           # 10-fold verification summary JSON & plots
│   ├── calibration/          # Production threshold calibration summary JSON & plots
│   ├── quality/              # Face quality analysis summary JSON & plots
│   ├── temporal/             # Temporal stability analysis summary JSON & plots
│   ├── presence/             # Presence & session state policy analysis JSON & plots
│   └── runtime/              # Runtime orchestration summary JSON & documentation
│       ├── README.md         # Runtime architecture, lifecycle, and telemetry documentation
│       └── runtime_analysis_summary.json # Machine-readable runtime execution & latency statistics
├── scripts/
│   ├── prepare_dataset.py                 # LFW acquisition & split partitioning
│   ├── validate_dataset.py                # Leakage, hash, and integrity audit
│   ├── benchmark_detection_alignment.py   # Detection & alignment benchmark
│   ├── benchmark_embeddings.py            # ArcFace embedding sanity checks & benchmark
│   ├── build_gallery.py                   # Enrolls reference identities into gallery
│   ├── evaluate_identification_pipeline.py# Open-set identification pipeline verification
│   ├── evaluate_verification.py           # Formal 10-fold LFW verification benchmark
│   ├── calibrate_threshold.py             # Validation production threshold calibrator
│   ├── evaluate_face_quality.py           # Validation face quality assessment & experiment
│   ├── evaluate_temporal_stability.py     # Validation temporal identity stability evaluation
│   ├── evaluate_presence_session.py       # Controlled presence & session policy evaluation
│   ├── run_runtime_orchestrator.py        # Runtime orchestrator demonstration & benchmark
│   ├── generate_baseline_embeddings.py    # Offline baseline feature extraction
│   └── verify_baseline.py                 # Manual & API verification script
├── tests/                    # PyTest test suite (111 tests)
│   ├── test_aligner.py       # 5-point alignment unit tests
│   ├── test_calibrator.py    # Threshold calibrator & strategy unit tests
│   ├── test_dataset.py       # Dataset partitioning and leakage tests
│   ├── test_detector.py      # Face detector unit tests (Dlib & Modern)
│   ├── test_embedder.py      # Embedding extractor unit tests (Dlib & ArcFace)
│   ├── test_evaluation_metrics.py # ROC, AUC, EER, FAR/FRR & fold-aware metric tests
│   ├── test_gallery.py       # IdentityGallery multi-template & search tests
│   ├── test_matcher.py       # Similarity matcher unit tests (Euclidean & Cosine)
│   ├── test_pipeline.py      # Baseline recognition pipeline tests
│   ├── test_quality.py       # Face Quality Assessment & metric unit tests
│   ├── test_temporal.py      # Temporal stabilization & state machine tests
│   ├── test_presence.py      # Presence state machine & session lifecycle tests
│   ├── test_runtime.py       # End-to-end runtime orchestration tests
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

### Empirical 10-Fold Verification Results (E1 vs E2)

| Metric | Experiment E1 (dlib Baseline) | Experiment E2 (ArcFace Modern) |
|---|---|---|
| **Fold-Calibrated Accuracy ($\text{Mean} \pm \text{Std}$)** | **$97.43\% \pm 0.60\%$** | **$98.50\% \pm 0.72\%$** |
| **Fold-Calibrated FAR ($\text{Mean} \pm \text{Std}$)** | **$1.53\% \pm 0.54\%$** | **$0.03\% \pm 0.10\%$** |
| **Fold-Calibrated FRR ($\text{Mean} \pm \text{Std}$)** | **$3.60\% \pm 1.38\%$** | **$2.97\% \pm 1.40\%$** |
| **Fold-Calibrated Threshold ($\text{Mean} \pm \text{Std}$)** | **$0.6345 \pm 0.0022$** (Euclidean) | **$0.2426 \pm 0.0045$** (Cosine) |
| **Global ROC-AUC** | **0.9941** | **0.9883** |
| **Global Equal Error Rate (EER)** | **0.0293** ($2.93\%$) | **0.0268** ($2.68\%$) |
| **Global EER Operating Threshold** | $0.6557$ (Euclidean) | $0.1160$ (Cosine) |

Full benchmark reports and visualizations are available in [`reports/evaluation/`](./reports/evaluation/README.md).

---

## 3. Production Threshold Calibration (Phase 7)

### Methodology & Validation Results
Conducted strictly on the **independent validation split** (59 identities, 1,395 images, 56,565 evaluated pairs) with zero access to the test set:

| Strategy | Threshold ($\tau$) | FAR (%) | FRR (%) | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|---|---|---|
| **A. Equal Error Rate (EER)** | $0.1280$ | $2.340\%$ | $2.346\%$ | $97.66\%$ | $84.57\%$ | $97.65\%$ | $90.64\%$ |
| **B. Maximum Accuracy** | $0.2800$ | $0.004\%$ | $2.498\%$ | $99.71\%$ | $99.97\%$ | $97.50\%$ | $98.72\%$ |
| **C. Security Low-FAR (Recommended)** | $\mathbf{0.2400}$ | $\mathbf{0.042\%}$ | $\mathbf{2.376\%}$ | $\mathbf{99.69\%}$ | $\mathbf{99.67\%}$ | $\mathbf{97.62\%}$ | $\mathbf{98.64\%}$ |
| **D. F1-Optimal** | $0.2840$ | $0.002\%$ | $2.498\%$ | $99.71\%$ | $99.98\%$ | $97.50\%$ | $98.73\%$ |
| **E. FAR/FRR-Balanced** | $0.1280$ | $2.340\%$ | $2.346\%$ | $97.66\%$ | $84.57\%$ | $97.65\%$ | $90.64\%$ |

* **Recommended Production Threshold**: **$\tau = 0.2400$** (Security-Oriented Low-FAR Strategy).
* Full calibration plots, distribution percentiles, and stability analyses are documented in [`reports/calibration/`](./reports/calibration/README.md).

---

## 4. Face Quality Assessment & Quality-Aware Recognition (Phase 8)

| Operating Mode | Frame/Pair Rejection Rate | Filtered Accuracy (%) | False Acceptance Rate (FAR) | False Rejection Rate (FRR) |
|---|---|---|---|---|
| **Baseline (No FQA)** | $0.00\%$ ($0 / 19,900$) | **$98.50\%$** | $0.070\%$ ($0.000700$) | $2.94\%$ ($0.02939$) |
| **Lenient FQA** | $0.68\%$ ($136 / 19,900$) | **$98.56\%$** | $0.071\%$ ($0.000707$) | $2.82\%$ ($0.02820$) |
| **Balanced FQA (Default)** | $\mathbf{8.72\%}$ ($1,736 / 19,900$) | **$98.61\%$$ | $\mathbf{0.077\%}$ ($0.000771$) | $\mathbf{2.71\%}$ ($0.02709$) |
| **Strict FQA** | $40.34\%$ ($8,028 / 19,900$) | **$99.26\%$** | $0.085\%$ ($0.000855$) | $1.38\%$ ($0.01378$) |

* Full quality plots, percentile distributions, and correlation matrices are documented in [`reports/quality/`](./reports/quality/README.md).

---

## 5. Temporal Recognition & Identity Stability (Phase 9)

### Controlled Temporal Policy Validation Results
Evaluated on simulated temporal sequences derived from the independent validation split (400 sequences, 6,100 observations):

| Operating Mode | Window Size ($W$) | Min Obs ($N_{\text{min}}$) | Simulated Obs Latency (frames) | Transient Recovery (%) | Rogue Blip Suppression (%) |
|---|---|---|---|---|---|
| **Baseline (Frame-Only)** | $1$ | $1$ | **$0.0$** (instant) | **$0.0\%$** ($0 / 200$) | **$0.0\%$** ($0 / 100$) |
| **FAST Mode** | $4$ | $3$ | **$3.1$** | **$97.2\%$** ($194 / 200$) | **$100.0\%$** ($100 / 100$) |
| **BALANCED Mode (Default)** | $\mathbf{7}$ | $\mathbf{4}$ | **$4.3$** (~$143\text{ ms}$ at assumed $30\text{ FPS}$) | **$97.2\%$$ ($194 / 200$) | **$100.0\%$$ ($100 / 100$) |
| **STABLE Mode** | $10$ | $6$ | **$6.8$** | **$63.8\%$** ($128 / 200$) | **$100.0\%$** ($100 / 100$) |

* Full temporal plots and policy documentation are in [`reports/temporal/`](./reports/temporal/README.md).

---

## 6. Presence & Session Intelligence (Phase 10)

### Controlled Presence Policy Validation Results
Evaluated across 8 operational scenarios (400 event sequences):

| Operating Mode | Min Entry Obs | Entry Window (s) | Grace Period (s) | Controlled Synthetic Entry Timing (s) | Session Continuity (%) | Interruption Recovery (%) | Unknown False Entry (%) |
|---|---|---|---|---|---|---|---|
| **FAST Mode** | $2$ | $3.0$ | $5.0$ | **$0.50$** (2 frames at 0.5s interval) | **$100.0\%$** | **$100.0\%$** | **$0.0\%$** |
| **BALANCED Mode (Default)** | $\mathbf{3}$ | $\mathbf{5.0}$ | $\mathbf{10.0}$ | **$1.00$** (3 frames at 0.5s interval) | **$100.0\%$** | **$100.0\%$** | **$0.0\%$** |
| **STRICT Mode** | $5$ | $8.0$ | $20.0$ | **$2.00$** (5 frames at 0.5s interval) | **$100.0\%$** | **$100.0\%$** | **$0.0\%$** |

* Full presence state machine documentation and lifecycle plots are in [`reports/presence/`](./reports/presence/README.md).

---

## 7. System Integration & Runtime Orchestration (Phase 11)

$$\text{BaseFrameSource} \xrightarrow{\text{RGB Frame}} \text{ModernRecognitionPipeline} \xrightarrow{\text{ModernRecognitionResult}} \text{RecognitionObservation} \xrightarrow{\text{TemporalIdentityStabilizer}} \text{TemporalRecognitionResult} \xrightarrow{\text{PresenceManager}} \text{RuntimeFrameResult}$$

* **Modular Orchestration (`FaceIntelligenceRuntime`)**: Cleanly composes detection, FQA, ArcFace embeddings, temporal voting, and presence tracking with full dependency injection.
* **Fault-Tolerant Execution**: Frame anomalies (corrupt images, quality rejections, missing landmarks) are captured gracefully as structured `RuntimeFrameResult` outputs without halting the processing loop.
* **Explicit Lifecycle Control**: Supports deterministic `start()`, `process_frame()`, `tick()`, `stop(reason="runtime_shutdown")`, and `reset()`.
* Full runtime documentation is in [`reports/runtime/`](./reports/runtime/README.md).

---

## 8. Setup & Execution Commands

### 1. Run Runtime Orchestration Demonstration
```bash
python scripts/run_runtime_orchestrator.py
```

### 2. Run Presence & Session Intelligence Evaluation
```bash
python scripts/evaluate_presence_session.py
```

### 3. Run Temporal Identity Stability Evaluation
```bash
python scripts/evaluate_temporal_stability.py
```

### 4. Run Face Quality Assessment Evaluation
```bash
python scripts/evaluate_face_quality.py
```

### 5. Run Production Threshold Calibration
```bash
python scripts/calibrate_threshold.py
```

### 6. Run 10-Fold LFW Face Verification Benchmark
```bash
python scripts/evaluate_verification.py
```

### 7. Run Complete PyTest Suite (111 tests)
```bash
pytest -v
```

### 8. Start Flask Web Application
```bash
python app.py
```
Access the application in your browser at `http://127.0.0.1:5000/`.
