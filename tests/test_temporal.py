import pytest
import time
import numpy as np

from ml.pipeline import ModernRecognitionResult
from ml.quality import FaceQualityMetrics
from ml.temporal import (
    TemporalIdentityStabilizer,
    RecognitionObservation,
    TemporalRecognitionResult,
    TemporalPolicyConfig,
    TemporalMode,
    TemporalState,
    PRESET_POLICIES
)


def _make_obs(identity="Alice", sim=0.75, recognized=True, quality_status="good", frame_idx=0, ts=100.0):
    return RecognitionObservation(
        timestamp=ts,
        frame_index=frame_idx,
        identity=identity if recognized else None,
        best_candidate=identity,
        similarity=sim,
        threshold=0.24,
        recognized=recognized,
        quality_status=quality_status,
        quality_score=0.85
    )


def test_observation_creation_and_dict():
    res = ModernRecognitionResult(
        identity="Alice",
        best_candidate="Alice",
        similarity=0.80,
        threshold=0.24,
        recognized=True,
        bbox=(10, 90, 90, 10),
        reason="accepted"
    )
    obs = RecognitionObservation.from_recognition_result(res, frame_index=1, timestamp=100.0)
    assert obs.identity == "Alice"
    assert obs.similarity == 0.80
    assert obs.recognized is True
    assert obs.frame_index == 1
    d = obs.to_dict()
    assert d["identity"] == "Alice"
    assert d["frame_index"] == 1


def test_temporal_stabilizer_stable_after_sufficient_evidence():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.BALANCED)
    # Balanced mode requires min_observations=4, window=7
    for i in range(3):
        res = stabilizer.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))
        assert res.is_stable is False
        assert res.state == TemporalState.UNSTABLE

    # 4th observation triggers STABLE
    res4 = stabilizer.update(_make_obs("Alice", frame_idx=3, ts=100.3))
    assert res4.is_stable is True
    assert res4.state == TemporalState.STABLE
    assert res4.stable_identity == "Alice"
    assert res4.consecutive_stable_count == 4


def test_temporal_stabilizer_insufficient_evidence_unstable():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.STABLE)
    # Stable mode requires min_observations=6
    for i in range(5):
        res = stabilizer.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))
        assert res.is_stable is False
        assert res.state == TemporalState.UNSTABLE


def test_temporal_stabilizer_transient_unknown_recovery():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.BALANCED)
    # Establish stable Alice
    for i in range(4):
        stabilizer.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))

    # Single transient Unknown observation
    res_unk = stabilizer.update(_make_obs("Alice", recognized=False, frame_idx=4, ts=100.4))
    # Should absorb Unknown and remain STABLE Alice
    assert res_unk.is_stable is True
    assert res_unk.stable_identity == "Alice"
    assert res_unk.state == TemporalState.STABLE

    # Next frame Alice confirms
    res_next = stabilizer.update(_make_obs("Alice", recognized=True, frame_idx=5, ts=100.5))
    assert res_next.is_stable is True
    assert res_next.stable_identity == "Alice"


def test_temporal_stabilizer_persistent_unknown_expiration():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.BALANCED)
    # Balanced max_unknown_observations=3
    for i in range(4):
        stabilizer.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))

    # 3 consecutive Unknowns absorbed
    for i in range(3):
        res = stabilizer.update(_make_obs("Alice", recognized=False, frame_idx=4 + i, ts=100.4 + i * 0.1))
        assert res.is_stable is True

    # 4th Unknown exceeds tolerance -> Expires to UNKNOWN
    res_exp = stabilizer.update(_make_obs("Alice", recognized=False, frame_idx=7, ts=100.7))
    assert res_exp.is_stable is False
    assert res_exp.stable_identity is None
    assert res_exp.state == TemporalState.UNKNOWN


def test_temporal_stabilizer_anomalous_blip_suppression():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.BALANCED)
    # Establish stable Alice
    for i in range(4):
        stabilizer.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))

    # Single rogue Bob frame (e.g. impostor glitch)
    res_bob = stabilizer.update(_make_obs("Bob", frame_idx=4, ts=100.4))
    # Should NOT switch to Bob; state=SWITCHING, stable_identity remains Alice
    assert res_bob.state == TemporalState.SWITCHING
    assert res_bob.stable_identity == "Alice"
    assert res_bob.challenger_identity == "Bob"
    assert res_bob.challenger_evidence == 1

    # Following frames return to Alice
    res_alice = stabilizer.update(_make_obs("Alice", frame_idx=5, ts=100.5))
    assert res_alice.state == TemporalState.STABLE
    assert res_alice.stable_identity == "Alice"
    assert res_alice.challenger_identity is None


def test_temporal_stabilizer_challenger_identity_switch():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.BALANCED)
    # Balanced challenger_switch_threshold=3
    # Establish stable Alice
    for i in range(4):
        stabilizer.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))

    # Frame 1 of Bob: challenger evidence=1, stable=Alice
    r1 = stabilizer.update(_make_obs("Bob", frame_idx=4, ts=100.4))
    assert r1.stable_identity == "Alice"
    assert r1.state == TemporalState.SWITCHING

    # Frame 2 of Bob: challenger evidence=2, stable=Alice
    r2 = stabilizer.update(_make_obs("Bob", frame_idx=5, ts=100.5))
    assert r2.stable_identity == "Alice"
    assert r2.state == TemporalState.SWITCHING

    # Frame 3 of Bob: challenger threshold met! Switch to Bob committed!
    r3 = stabilizer.update(_make_obs("Bob", frame_idx=6, ts=100.6))
    assert r3.stable_identity == "Bob"
    assert r3.state == TemporalState.STABLE
    assert r3.is_stable is True


def test_temporal_stabilizer_quality_weighting():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.BALANCED)
    # 4 poor quality frames should NOT trigger stable identity
    for i in range(4):
        res = stabilizer.update(_make_obs("Alice", quality_status="poor", frame_idx=i, ts=100.0 + i * 0.1))
        assert res.is_stable is False

    # 4 good quality frames trigger stable identity
    for i in range(4):
        res = stabilizer.update(_make_obs("Alice", quality_status="good", frame_idx=4 + i, ts=100.4 + i * 0.1))
    assert res.is_stable is True
    assert res.stable_identity == "Alice"


def test_temporal_stabilizer_time_gap_expiration():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.BALANCED)
    # Establish stable Alice at t=100.0
    for i in range(4):
        stabilizer.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))

    # Gap of 10.0 seconds (exceeds max_gap_seconds=2.0)
    res_gap = stabilizer.update(_make_obs("Alice", frame_idx=10, ts=110.3))
    # Should have reset and treated as frame 1 -> UNSTABLE
    assert res_gap.is_stable is False
    assert res_gap.state == TemporalState.UNSTABLE


def test_temporal_stabilizer_reset():
    stabilizer = TemporalIdentityStabilizer(mode=TemporalMode.BALANCED)
    for i in range(4):
        stabilizer.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))
    assert stabilizer.current_stable_identity == "Alice"

    stabilizer.reset()
    state = stabilizer.get_state()
    assert state["current_stable_identity"] is None
    assert state["consecutive_stable_count"] == 0
    assert state["window_size"] == 0


def test_temporal_modes_fast_balanced_stable():
    fast_st = TemporalIdentityStabilizer(mode=TemporalMode.FAST)
    stable_st = TemporalIdentityStabilizer(mode=TemporalMode.STABLE)

    # 3 frames of Alice
    for i in range(3):
        r_fast = fast_st.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))
        r_stable = stable_st.update(_make_obs("Alice", frame_idx=i, ts=100.0 + i * 0.1))

    # FAST mode achieves stable in 3 frames
    assert r_fast.is_stable is True
    # STABLE mode requires 6 frames, remains UNSTABLE at 3 frames
    assert r_stable.is_stable is False
