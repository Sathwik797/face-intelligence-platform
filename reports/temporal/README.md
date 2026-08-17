# Phase 9 Temporal Recognition & Identity Stability

## Overview
This directory contains experimental artifacts, statistical summaries, and visualization plots from the Phase 9 **controlled temporal policy validation** of the **Temporal Identity Stabilizer**. 

> **Important Machine Learning Evaluation Note**:
> Temporal performance was evaluated using **controlled simulated sequences derived from validation observations** (`data/evaluation/validation/`, 59 identities, 1,395 images). Real continuous video-stream recognition performance remains unvalidated and constitutes future empirical work.

---

## 1. Motivation & Architectural Distinction

In practical biometric systems, single-frame face recognition results fluctuate due to momentary optical occlusions, minor pose deviations, or transient detector misses:

$$\text{Frame 1: Alice} \to \text{Frame 2: Alice} \to \text{Frame 3: Unknown} \to \text{Frame 4: Alice} \to \text{Frame 5: Alice}$$

Treating Frame 3 as a hard identity change or attendance loss produces severe identity flicker.

The **Temporal Identity Stabilizer** sits downstream of the modern recognition pipeline:

$$\text{Frame} \xrightarrow{\text{YuNet}} \text{Face Crop} \xrightarrow{\text{FQA}} \text{Quality Check} \xrightarrow{\text{ArcFace}} \text{Recognition Observation} \xrightarrow{\text{Stabilizer}} \text{Stable Identity Decision}$$

---

## 2. Temporal Decision & Evidence Policy

1. **Sliding Observation Window**: Maintains a bounded sliding window of length $W$ (e.g. $W = 7$).
2. **Quality-Weighted Evidence**:
   - High-quality recognized observations contribute full evidence ($w = 1.0$).
   - Quality-rejected observations contribute $w = 0.0$ to prevent optical noise from polluting the temporal window.
3. **Stability Condition**:
   $$\frac{\sum w_{\text{active}}}{\sum w_{\text{valid}}} \ge \theta_{\text{ratio}} \quad \text{and} \quad \sum w_{\text{active}} \ge N_{\text{min}}$$
4. **Transient Unknown Absorption**:
   - Absorbs up to $K_{\text{unknown}}$ consecutive Unknown frames without losing the active stable identity.
   - If consecutive Unknowns exceed $K_{\text{unknown}}$ or the temporal gap exceeds $\Delta t_{\text{max}}$, transitions to `UNKNOWN`.
5. **Challenger Identity Switching**:
   - A competing identity $B \neq A$ must achieve $N_{\text{switch}}$ consecutive observations before displacing active identity $A$.
   - Single rogue impostor blips ($[A, A, A, B, A, A]$) are cleanly suppressed in state `SWITCHING` while preserving stable identity $A$.

---

## 3. Controlled Temporal Policy Validation Results (400 Sequences, 6,100 Observations)

| Operating Mode | Window Size ($W$) | Min Obs ($N_{\text{min}}$) | Simulated Stabilization Latency (frames) | Transient Unknown Recovery (%) | Rogue Blip Suppression (%) | False Switch Rate (%) |
|---|---|---|---|---|---|---|
| **Baseline (Frame-Only)** | $1$ | $1$ | **$0.0$** (instant) | **$0.0\%$** ($0 / 200$) | **$0.0\%$** ($0 / 100$) | **$100.0\%$** |
| **FAST Mode** | $4$ | $3$ | **$3.1$** | **$97.2\%$** ($194 / 200$) | **$100.0\%$** ($100 / 100$) | **$0.0\%$** |
| **BALANCED Mode (Default)** | $\mathbf{7}$ | $\mathbf{4}$ | $\mathbf{4.3}$ | $\mathbf{97.2\%}$ ($194 / 200$) | $\mathbf{100.0\%}$ ($100 / 100$) | $\mathbf{0.0\%}$ |
| **STABLE Mode** | $10$ | $6$ | **$6.8$** | **$63.8\%$** ($128 / 200$) | **$100.0\%$** ($100 / 100$) | **$0.0\%$** |

---

## 4. Interpretation of Experimental Evidence

* **Validation of State-Transition Logic**: This experiment is a controlled validation of the temporal state-transition behavior and policy rules under synthetic operational stress scenarios.
* **Simulated Sequences**: All sequences are constructed by sampling observations from the independent validation split. They do not represent contiguous video streams with continuous physical motion trajectories.
* **Transient Unknown Recovery**: Under the simulated transient single-frame Unknown dropout scenario, the Balanced policy recovered $97.2\%$ of temporary Unknown events while preserving the active identity.
* **Rogue Blip Suppression**: Under the simulated single-frame rogue-identity-blip scenario, the Balanced temporal policy suppressed $100 / 100$ single-frame challenger blips because the policy requires sustained challenger evidence before committing an identity switch.
* **Observation-Level Latency**: The reported latency metrics ($3.1$, $4.3$, $6.8$ frames) measure simulated observation-level stabilization latency. At an assumed $30\text{ FPS}$, $4.3$ frames corresponds to approximately $143\text{ ms}$ of observational delay.
* **Real-World Scope**: These results demonstrate that the configured temporal policies behave mathematically and algorithmically as intended under the tested scenarios, but do not establish real-world video-stream recognition accuracy or physical camera tracking performance.

---

## 5. Directory Contents

### Tracked Evidence Files
* [`temporal_analysis_summary.json`](./temporal_analysis_summary.json): Complete machine-readable summary containing sequence counts, mode configurations, recovery rates, false switch rates, and latency statistics.
* `plots/`: Visual evidence plots:
  - `identity_evidence_over_time.png`: Evidence accumulation curves across operating modes.
  - `similarity_over_time.png`: Cosine similarity timeline showing transient Unknown dropout absorption.
  - `stable_vs_unstable_transitions.png`: State progression timelines (`UNSTABLE` $\to$ `STABLE` $\to$ `SWITCHING`).
  - `stabilization_latency_comparison.png`: Mean simulated observations required to establish a confirmed stable identity.
  - `identity_switch_comparison.png`: Single-frame rogue blip suppression vs verified identity transitions.
  - `configuration_tradeoff_comparison.png`: Accuracy, recovery, and blip suppression across FAST, BALANCED, and STABLE modes.

---

## 6. Technical Recommendation & Operating Trade-offs

* **Default Mode**: **`BALANCED`**
* **Technical Justification**:
  - `BALANCED` mode filters transient single-frame dropout anomalies and suppresses rogue challenger blips while maintaining an optimal simulated observation latency of $4.3$ frames.
