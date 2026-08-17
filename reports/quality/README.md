# Phase 8 Face Quality Assessment & Quality-Aware Recognition

## Overview
This directory contains experimental artifacts, statistical distributions, and visualization plots from the Phase 8 formal evaluation of the **Face Quality Assessment (FQA)** subsystem. The evaluation was conducted strictly on the project's **independent validation split** (`data/evaluation/validation/`, 59 identities, 1,395 images).

---

## 1. Quality Signals & Methodology

Face quality assessment evaluates whether a detected face image exhibits sufficient optical and geometric fidelity to produce a trustworthy embedding vector and recognition decision.

### Implemented Signals:
1. **Face Dimensions & Resolution**: Bounding box width, height, area, and face-to-image area ratio.
2. **Blur / Sharpness**: Variance of Laplacian ($\sigma^2(\nabla^2 I)$) computed on the grayscale face crop.
3. **Illumination / Brightness**: Mean grayscale pixel intensity $\mu(I) \in [0, 255]$.
4. **Contrast**: Standard deviation of grayscale pixel intensity $\sigma(I) \in [0, 255]$.
5. **Detection Confidence**: Detector probability score from YuNet ONNX in $[0.0, 1.0]$.
6. **Alignment Quality**: Landmark geometric structure sanity, eye tilt roll penalty, vertical ordering check, and bounds containment in $[0.0, 1.0]$.
7. **Pose Quality**: Frontal horizontal symmetry proxy from eye-to-nose distances in $[0.0, 1.0]$.
8. **Composite Quality Index**: Normalized weighted multi-factor quality metric in $[0.0, 1.0]$.

---

## 2. Directory Contents

### Tracked Evidence Files
* [`quality_analysis_summary.json`](./quality_analysis_summary.json): Complete machine-readable summary containing distribution percentiles, correlation metrics, threshold presets, and mode comparison tables.
* `plots/`: Visual evidence plots:
  - `blur_distribution.png`: Face sharpness distribution with median and threshold markers.
  - `face_size_distribution.png`: Face width and height histograms.
  - `brightness_distribution.png`: Illumination distribution with under/overexposure boundaries.
  - `contrast_distribution.png`: Grayscale contrast standard deviation distribution.
  - `detection_confidence_distribution.png`: YuNet detection confidence distribution.
  - `quality_vs_recognition_similarity.png`: Genuine pair ArcFace cosine similarity vs overall face quality scatter plot with trend line.
  - `quality_aware_vs_baseline_performance.png`: Comparative verification accuracy across operating modes.
  - `rejection_rate_vs_performance_tradeoff.png`: Rejection rate vs accuracy and error rate trade-off curves.

### Excluded Artifacts (Gitignored)
* `cache/` (`validation_quality.npz`): Intermediate cached quality evaluations across validation images.

---

## 3. Validation Distribution Statistics & Percentiles

| Metric | Mean $\pm$ Std | Median | Min / Max | $P_5$ | $P_{25}$ | $P_{50}$ | $P_{75}$ | $P_{95}$ |
|---|---|---|---|---|---|---|---|---|
| **Sharpness / Blur** | $144.78 \pm 97.70$ | $120.72$ | $[14.64, 789.99]$ | $41.70$ | $80.06$ | $120.72$ | $182.20$ | $334.68$ |
| **Face Width (px)** | $94.67 \pm 8.12$ | $94.00$ | $[45.0, 156.0]$ | $84.00$ | $91.00$ | $94.00$ | $98.00$ | $106.00$ |
| **Face Height (px)** | $127.95 \pm 11.76$ | $127.00$ | $[47.0, 202.0]$ | $111.00$ | $123.00$ | $127.00$ | $133.00$ | $145.00$ |
| **Face Area Ratio** | $0.1950 \pm 0.0331$ | $0.1925$ | $[0.0338, 0.5042]$ | $0.1500$ | $0.1805$ | $0.1925$ | $0.2064$ | $0.2414$ |
| **Brightness** | $131.02 \pm 20.29$ | $132.18$ | $[58.35, 185.45]$ | $95.03$ | $118.34$ | $132.18$ | $145.00$ | $161.94$ |
| **Contrast** | $40.78 \pm 8.57$ | $40.13$ | $[17.20, 75.69]$ | $27.56$ | $34.73$ | $40.13$ | $46.15$ | $55.05$ |
| **Confidence** | $0.9320 \pm 0.0088$ | $0.9332$ | $[0.8815, 0.9530]$ | $0.9162$ | $0.9270$ | $0.9332$ | $0.9380$ | $0.9437$ |
| **Alignment Quality** | $0.9789 \pm 0.0169$ | $0.9818$ | $[0.8049, 1.0000]$ | $0.9502$ | $0.9704$ | $0.9818$ | $0.9914$ | $0.9982$ |
| **Pose Quality** | $0.8406 \pm 0.1020$ | $0.8543$ | $[0.5208, 0.9999]$ | $0.6543$ | $0.7669$ | $0.8543$ | $0.9267$ | $0.9834$ |
| **Overall Quality** | $0.7479 \pm 0.1027$ | $0.7342$ | $[0.5332, 0.9745]$ | $0.6072$ | $0.6641$ | $0.7342$ | $0.8232$ | $0.9303$ |

---

## 4. Quality vs Recognition Correlations (Genuine Pairs)

* **Detection Confidence vs Cosine Similarity**: $+0.2837$
* **Brightness vs Cosine Similarity**: $+0.1782$
* **Overall Quality vs Cosine Similarity**: $+0.1146$
* **Sharpness / Blur vs Cosine Similarity**: $+0.0917$
* **Contrast vs Cosine Similarity**: $+0.0608$

---

## 5. Quality-Aware Recognition Experiment Results ($\tau = 0.24$)

| Operating Mode | Frame/Pair Rejection Rate | Filtered Accuracy (%) | False Acceptance Rate (FAR) | False Rejection Rate (FRR) |
|---|---|---|---|---|
| **Baseline (No FQA)** | $0.00\%$ ($0 / 19,900$) | **$98.50\%$** | $0.070\%$ ($0.000700$) | $2.94\%$ ($0.02939$) |
| **Lenient FQA** | $0.68\%$ ($136 / 19,900$) | **$98.56\%$** | $0.071\%$ ($0.000707$) | $2.82\%$ ($0.02820$) |
| **Balanced FQA (Recommended)** | $\mathbf{8.72\%}$ ($1,736 / 19,900$) | $\mathbf{98.61\%}$ | $\mathbf{0.077\%}$ ($0.000771$) | $\mathbf{2.71\%}$ ($0.02709$) |
| **Strict FQA** | $40.34\%$ ($8,028 / 19,900$) | **$99.26\%$** | $0.085\%$ ($0.000855$) | $1.38\%$ ($0.01378$) |

---

## 6. Technical Recommendation & Operating Trade-offs

* **Default Mode**: **`BALANCED`**
* **Technical Justification**:
  - `BALANCED` mode filters out the bottom $5\text{--}10\%$ of degraded captures (blur $< 40.0$, low confidence, harsh lighting) without unnecessarily penalizing user throughput.
  - Improves genuine recognition consistency ($\text{FRR}$ drops from $2.94\%$ down to $2.71\%$), while preserving strict security ($\text{FAR} \le 0.08\%$).
  - Rejected frames are cleanly reported with diagnostic reasons (e.g. `quality_rejected: blurry`), prompting the client application to request a clean rescan rather than returning an erroneous identity classification.
