# Phase 10 Presence & Session Intelligence

## Overview
This directory contains experimental artifacts, statistical summaries, and visualization plots from the Phase 10 **controlled presence and session policy validation** of the **PresenceManager** and **IdentityPresenceStateMachine**.

> **Important Machine Learning & Systems Evaluation Note**:
> Under the controlled synthetic event scenarios evaluated in Phase 10, the presence state machine achieved 100% session continuity and 100% recovery for interruptions within the configured grace period. These results validate deterministic state-machine behavior under the tested scenarios; they do not establish real-world camera-stream presence accuracy. Real camera-stream presence and physical tracking performance remains to be validated upon live video integration.

---

## 1. Architectural Distinction & Layer Responsibilities

1. **Recognition (Phase 5)**: *"Who does this individual frame crop look like?"*
2. **Temporal Identity Stabilization (Phase 9)**: *"Do consecutive frame observations consistently support this identity over time?"*
3. **Presence Intelligence (Phase 10)**: *"Given a temporally stable identity, is that person physically present, temporarily missing, or departed?"*
4. **Session Lifecycle (Phase 10)**: *"When did this confirmed physical presence period start, how long did it last, and when did it conclude?"*

```
Image
  ↓
YuNet Detection
  ↓
Face Quality Assessment (Phase 8 FQA)
  ↓
ArcFace ResNet-50 512D Embeddings
  ↓
Temporal Identity Stabilization (Phase 9)
  ↓
TemporalRecognitionResult (Stable Identity)
  ↓
IdentityPresenceStateMachine (Phase 10)
  ↓
PresenceSession (In-Memory State)
```

---

## 2. Presence State Machine & Transition Rules

```
       [ NOT_PRESENT ] / [ ABSENT ]
                 │
                 │ (first stable observation)
                 ▼
          [ CANDIDATE ]
                 │
                 │ (min_entry_observations within entry_window)
                 ▼
           [ PRESENT ] ◄─────────────────┐
                 │                       │ (reappears before
                 │ (observation lost)    │  grace timeout)
                 ▼                       │
            [ GRACE ] ───────────────────┘
                 │
                 │ (grace timeout expired)
                 ▼
            [ ABSENT ] ──> (Session Closed & Archived)
```

* **`CANDIDATE`**: Buffers initial observations to prevent single-frame false entries.
* **`PRESENT`**: Confirmed active presence. Updates `last_seen_at` and increments `observation_count`.
* **`GRACE`**: Tolerates temporary observation dropouts (e.g. looking away, momentary occlusion). Keeps the session active without fragmenting into duplicate sessions.
* **`ABSENT`**: Closes the session with timezone-aware `ended_at` and finalized `duration_seconds`. Any subsequent appearance initiates a fresh session.

---

## 3. Controlled Presence Policy Validation Results (400 Sequences)

| Operating Mode | Min Entry Obs | Entry Window (s) | Grace Period (s) | Controlled Synthetic Entry Timing (s) | Session Continuity (%) | Interruption Recovery (%) | Unknown False Entry (%) |
|---|---|---|---|---|---|---|---|
| **FAST Mode** | $2$ | $3.0$ | $5.0$ | **$0.50$** (2 frames at 0.5s interval) | **$100.0\%$** | **$100.0\%$** | **$0.0\%$** |
| **BALANCED Mode (Default)** | $\mathbf{3}$ | $\mathbf{5.0}$ | $\mathbf{10.0}$ | **$1.00$** (3 frames at 0.5s interval) | **$100.0\%$** | **$100.0\%$** | **$0.0\%$** |
| **STRICT Mode** | $5$ | $8.0$ | $20.0$ | **$2.00$** (5 frames at 0.5s interval) | **$100.0\%$** | **$100.0\%$** | **$0.0\%$** |

---

## 4. Interpretation of Experimental Evidence

* **Deterministic State-Machine Validation**: Evaluates the mathematical and logical correctness of state transitions, grace timers, and session duration calculations under synthetic event streams.
* **Controlled Synthetic Timing**: Entry timing metrics ($0.50\text{s}$, $1.00\text{s}$, $2.00\text{s}$) represent controlled synthetic observation/event timing derived from simulated 0.5s frame observation intervals, not measured camera or hardware latency.
* **Synthetic Scenario Results**: Under the controlled synthetic event scenarios evaluated in Phase 10, the presence state machine achieved 100% session continuity and 100% recovery for interruptions within the configured grace period.
* **Safety Safeguard**: The `max_session_duration_seconds: 28800.0` parameter is implemented purely as a configurable safety safeguard against indefinitely open sessions in orphaned camera streams; it is not a fixed universal attendance or workday business rule.
* **Real-World Scope**: These results validate deterministic state-machine behavior under the tested scenarios; they do not establish real-world camera-stream presence accuracy.

---

## 5. Directory Contents

### Tracked Evidence Files
* [`presence_analysis_summary.json`](./presence_analysis_summary.json): Complete machine-readable summary containing scenario evaluations, mode metrics, and transition justifications.
* `plots/`: Visual evidence figures:
  - `state_transition_timeline.png`: Step timeline showing `NOT_PRESENT` $\to$ `CANDIDATE` $\to$ `PRESENT` $\to$ `GRACE` $\to$ `ABSENT`.
  - `session_lifecycle_timeline.png`: Active presence duration vs grace period vs closure timeline.
  - `entry_confirmation_delay_comparison.png`: Controlled synthetic entry confirmation delay across FAST, BALANCED, and STRICT modes.
  - `grace_period_recovery_comparison.png`: Session preservation vs interruption gap duration.
  - `fast_balanced_strict_tradeoff.png`: Operating trade-off comparison across presence modes.
  - `multi_identity_state_timeline.png`: Parallel independent state tracking for multiple identities (Alice & Bob).

---

## 6. Technical Recommendation

* **Default Mode**: **`BALANCED`**
* **Parameters**: `min_entry_observations: 3`, `entry_window_seconds: 5.0`, `grace_period_seconds: 10.0`, `max_session_duration_seconds: 28800.0` ($8\text{ hours configurable safeguard}$).
* **Justification**: Provides rapid entry confirmation ($1.00\text{s}$ at simulated 0.5s observation intervals), complete resilience against momentary $10\text{s}$ occlusions without session fragmentation, and zero false/duplicate sessions under tested synthetic scenarios.
