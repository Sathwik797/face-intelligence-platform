# Phase 11 System Integration & Runtime Orchestration

## Overview
This directory documents the **FaceIntelligenceRuntime** orchestration subsystem established in Phase 11. It provides an end-to-end, decoupled, and fault-tolerant execution engine connecting the modern recognition pipeline, temporal stabilization, and presence intelligence layers.

> **System Integration Architecture Note**:
> The runtime orchestrator acts purely as a composition engine. It contains **zero machine learning models** itself, delegating frame processing, temporal voting, and presence state machines to their respective established layers while maintaining strict error isolation, timezone-aware timing, and deterministic lifecycle control.

---

## 1. End-to-End Runtime Data Flow

$$\text{BaseFrameSource} \xrightarrow{\text{RGB Frame}} \text{ModernRecognitionPipeline} \xrightarrow{\text{ModernRecognitionResult}} \text{RecognitionObservation} \xrightarrow{\text{TemporalIdentityStabilizer}} \text{TemporalRecognitionResult} \xrightarrow{\text{PresenceManager}} \text{RuntimeFrameResult}$$

```
BaseFrameSource (Synthetic / Static / OpenCV)
      │
      ▼ (RGB Image Array (H, W, 3) + Timezone-Aware UTC Timestamp)
┌─────────────────────────────────────────────────────────────┐
│ 1. ModernRecognitionPipeline (ml/pipeline.py)               │
│    - YuNet Face Detection (ONNX)                            │
│    - Primary Face Selection (Highest Confidence Policy)     │
│    - Face Quality Assessment (Phase 8 FQA)                  │
│    - 5-Point Affine Landmark Alignment (112x112 Canonical) │
│    - ArcFace ResNet-50 512D Embedding Extraction (ONNX)     │
│    - Multi-template Cosine Similarity Search                │
│    - Production Decision Threshold: tau = 0.2400            │
└─────────────────────────────────────────────────────────────┘
      │
      ▼ ModernRecognitionResult
┌─────────────────────────────────────────────────────────────┐
│ 2. TemporalIdentityStabilizer (ml/temporal/)                │
│    - Sliding Window Observation Buffer (W=7, N_min=4)       │
│    - Quality-Weighted Consensus (Good=1.0, Poor=0.0)        │
│    - Challenger Identity Switch Threshold (K=3)             │
│    - Single-Frame Rogue Blip & Transient Dropout Absorption │
└─────────────────────────────────────────────────────────────┘
      │
      ▼ TemporalRecognitionResult (Stable Identity / Unknown)
┌─────────────────────────────────────────────────────────────┐
│ 3. PresenceManager & State Machine (ml/presence/)           │
│    - NOT_PRESENT -> CANDIDATE -> PRESENT -> GRACE -> ABSENT │
│    - Multi-Identity Independent Session Lifecycle           │
│    - Active & Closed PresenceSession Management             │
│    - Timezone-Aware UTC Session Duration Tracking           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
RuntimeFrameResult (Events, Active Sessions, Stage Latencies)
```

---

## 2. Runtime Lifecycle & Graceful Shutdown

* **`start()`**: Sets runtime status to `RUNNING` and initializes clock references and frame counters.
* **`process_frame(rgb_frame, timestamp)`**: Executes full end-to-end pipeline with stage latency tracking and fault isolation.
* **`tick(timestamp)`**: Triggers periodic background sweeps across tracked identities to detect expiring grace timeouts.
* **`stop(reason="runtime_shutdown")`**: Finalizes any open in-memory presence sessions and archives them with the explicit reason `"runtime_shutdown"` (distinguishing runtime stoppage from natural attendance absences).
* **`reset()`**: Clears temporal history, presence state machines, frame counters, and telemetry buffers.

---

## 3. Error Isolation & Fault Tolerance

| Operational Anomaly | Pipeline Reaction | System State Outcome |
|---|---|---|
| **Corrupt / Empty Image Array** | Returns `ModernRecognitionResult(reason="invalid_image")` | Handled gracefully without crash; presence enters/remains in `GRACE` |
| **No Face in Frame** | Returns `ModernRecognitionResult(reason="no_face_detected")` | Recorded as observation gap; active session enters `GRACE` |
| **Quality-Rejected Frame** | Returns `ModernRecognitionResult(reason="quality_rejected")` | Assigned weight $0.0$; session preserved during grace window |
| **Unenrolled Face (Unknown)** | Returns `ModernRecognitionResult(identity=None)` | Reached `TemporalState.UNKNOWN`; 0 false presence sessions created |
| **Unexpected Exception** | Caught by frame try-except block; sets `RuntimeFrameResult.error` | Error isolated to single frame; runtime execution loop continues |

---

## 4. Tracked Evidence Files

* [`runtime_analysis_summary.json`](./runtime_analysis_summary.json): Complete machine-readable summary containing execution statistics, stage latencies, and lifecycle verification.
