import pytest
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import numpy as np

from ml.runtime import (
    FaceIntelligenceRuntime,
    RuntimeStatus,
    RuntimeConfig,
    StageLatencyMetrics,
    RuntimeFrameResult,
    BaseFrameSource,
    StaticFrameSource,
    SyntheticFrameSource
)
from ml.pipeline import ModernRecognitionPipeline, ModernRecognitionResult
from ml.temporal.schemas import TemporalPolicyConfig, TemporalMode
from ml.temporal.stabilizer import TemporalIdentityStabilizer
from ml.presence.schemas import PresenceMode, PresenceState, PresenceEventType
from ml.presence.state_machine import PRESENCE_PRESETS
from ml.presence.manager import PresenceManager


class MockModernRecognitionPipeline:
    """Mock recognition pipeline for deterministic runtime testing without ONNX weights."""

    def __init__(self, fixed_identity: Optional[str] = "Alice", fixed_sim: float = 0.85):
        self.fixed_identity = fixed_identity
        self.fixed_sim = fixed_sim
        self.threshold = 0.24
        self.calls = 0

    def recognize(self, rgb_image: np.ndarray) -> ModernRecognitionResult:
        self.calls += 1
        if rgb_image is None or rgb_image.size == 0:
            return ModernRecognitionResult(
                identity=None,
                best_candidate="Unknown",
                similarity=-1.0,
                threshold=self.threshold,
                recognized=False,
                reason="invalid_image"
            )

        is_rec = self.fixed_identity is not None
        return ModernRecognitionResult(
            identity=self.fixed_identity if is_rec else None,
            best_candidate=self.fixed_identity or "Unknown",
            similarity=self.fixed_sim if is_rec else 0.10,
            threshold=self.threshold,
            recognized=is_rec,
            bbox=(10, 100, 100, 10),
            latency_ms=5.0,
            reason="accepted" if is_rec else "below_threshold"
        )


def _create_test_runtime(mock_id="Alice"):
    rec_pipe = MockModernRecognitionPipeline(fixed_identity=mock_id)
    temp_stab = TemporalIdentityStabilizer(policy=TemporalPolicyConfig(
        window_size=4,
        min_observations=2,
        min_stable_ratio=0.65,
        mode="fast"
    ))
    pres_mgr = PresenceManager(config=PRESENCE_PRESETS[PresenceMode.FAST])  # min_entry=2, grace=5.0s
    runtime = FaceIntelligenceRuntime(
        recognition_pipeline=rec_pipe,
        temporal_stabilizer=temp_stab,
        presence_manager=pres_mgr,
        config=RuntimeConfig(auto_tick=True, heartbeat_interval_seconds=1.0)
    )
    return runtime


def test_runtime_initial_state():
    runtime = _create_test_runtime()
    assert runtime.status == RuntimeStatus.STOPPED
    assert runtime.frame_counter == 0
    assert len(runtime.buffered_results) == 0


def test_runtime_start_and_process_frame():
    runtime = _create_test_runtime(mock_id="Alice")
    runtime.start()
    assert runtime.status == RuntimeStatus.RUNNING

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    t0 = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
    res = runtime.process_frame(frame, timestamp=t0)

    assert isinstance(res, RuntimeFrameResult)
    assert res.frame_index == 1
    assert res.recognition.identity == "Alice"
    assert res.latencies.total_ms >= 0.0
    assert len(runtime.buffered_results) == 1


def test_runtime_end_to_end_known_identity_entry_and_presence():
    runtime = _create_test_runtime(mock_id="Alice")
    runtime.start()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    base_t = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)

    # Frame 1: Frame recognized -> Temporal unstable -> Candidate
    res1 = runtime.process_frame(frame, timestamp=base_t)
    assert res1.temporal.stable_identity is None
    assert runtime.presence_manager.get_presence_state("Alice") == PresenceState.NOT_PRESENT

    # Frame 2: Temporal reaches STABLE -> Presence enters CANDIDATE
    res2 = runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=0.5))
    assert res2.temporal.stable_identity == "Alice"
    assert runtime.presence_manager.get_presence_state("Alice") == PresenceState.CANDIDATE

    # Frame 3: Presence accumulates 2nd stable observation -> ENTRY_CONFIRMED -> PRESENT
    res3 = runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=1.0))
    assert runtime.presence_manager.get_presence_state("Alice") == PresenceState.PRESENT
    assert len(res3.active_sessions) == 1
    assert res3.active_sessions[0].identity == "Alice"
    assert any(e.event_type == PresenceEventType.ENTRY_CONFIRMED for e in res3.presence_events)


def test_runtime_unknown_frame_absorption():
    runtime = _create_test_runtime(mock_id=None)  # Unknown identity
    runtime.start()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    base_t = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(5):
        res = runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=i * 0.5))
        assert res.recognition.recognized is False
        assert len(res.active_sessions) == 0

    assert len(runtime.presence_manager.get_session_history()) == 0


def test_runtime_grace_recovery_stream():
    runtime = _create_test_runtime(mock_id="Alice")
    runtime.start()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    base_t = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)

    # Establish PRESENT
    for i in range(3):
        runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=i * 0.5))
    assert runtime.presence_manager.get_presence_state("Alice") == PresenceState.PRESENT
    orig_session_id = runtime.presence_manager.get_active_sessions()[0].session_id

    # Simulate temporary camera dropout (Unknown frames)
    runtime.recognition_pipeline.fixed_identity = None
    for i in range(4):
        runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=2.0 + i * 0.5))
    assert runtime.presence_manager.get_presence_state("Alice") == PresenceState.GRACE

    # Identity returns within grace period (feeding 3 frames to re-establish temporal consensus)
    runtime.recognition_pipeline.fixed_identity = "Alice"
    runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=4.5))
    runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=5.0))
    runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=5.5))

    assert runtime.presence_manager.get_presence_state("Alice") == PresenceState.PRESENT
    assert len(runtime.presence_manager.get_active_sessions()) == 1
    assert runtime.presence_manager.get_active_sessions()[0].session_id == orig_session_id
    assert runtime.presence_manager.get_active_sessions()[0].interruption_count >= 1


def test_runtime_grace_timeout_and_session_closure():
    runtime = _create_test_runtime(mock_id="Alice")
    runtime.start()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    base_t = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)

    # Establish PRESENT
    for i in range(3):
        runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=i * 0.5))
    assert runtime.presence_manager.get_presence_state("Alice") == PresenceState.PRESENT

    # Extended absence (12 seconds of Unknowns, exceeding 5s grace period)
    runtime.recognition_pipeline.fixed_identity = None
    for i in range(12):
        runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=2.0 + i * 1.0))

    # Alice is now marked ABSENT and session is archived
    assert runtime.presence_manager.get_presence_state("Alice") == PresenceState.ABSENT
    assert len(runtime.presence_manager.get_active_sessions()) == 0
    assert len(runtime.presence_manager.get_session_history()) == 1


def test_runtime_stop_graceful_shutdown():
    runtime = _create_test_runtime(mock_id="Alice")
    runtime.start()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    base_t = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(3):
        runtime.process_frame(frame, timestamp=base_t + timedelta(seconds=i * 0.5))
    assert len(runtime.presence_manager.get_active_sessions()) == 1

    # Graceful stop
    shutdown_evs = runtime.stop(reason="runtime_shutdown")
    assert runtime.status == RuntimeStatus.STOPPED
    assert len(runtime.presence_manager.get_active_sessions()) == 0
    assert len(runtime.presence_manager.get_session_history()) == 1
    assert any(e.event_type == PresenceEventType.SESSION_ENDED for e in shutdown_evs)
    assert "runtime_shutdown" in shutdown_evs[0].reason


def test_runtime_fault_tolerance_on_invalid_image():
    runtime = _create_test_runtime(mock_id="Alice")
    runtime.start()

    # Pass empty array
    empty_frame = np.array([])
    res = runtime.process_frame(empty_frame)

    assert isinstance(res, RuntimeFrameResult)
    assert res.recognition.recognized is False
    assert res.recognition.reason == "invalid_image"


def test_runtime_frame_source_iteration():
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    src = StaticFrameSource(frames=frames, fps=10.0)

    runtime = _create_test_runtime(mock_id="Alice")
    results = runtime.run_stream(src, max_frames=5)

    assert len(results) == 5
    assert runtime.status == RuntimeStatus.STOPPED


def test_runtime_synthetic_frame_source():
    src = SyntheticFrameSource(max_frames=4, width=80, height=80)
    runtime = _create_test_runtime(mock_id="Alice")
    results = runtime.run_stream(src)

    assert len(results) == 4
    assert runtime.status == RuntimeStatus.STOPPED


def test_runtime_reset():
    runtime = _create_test_runtime(mock_id="Alice")
    runtime.start()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    runtime.process_frame(frame)
    assert runtime.frame_counter == 1

    runtime.reset()
    assert runtime.status == RuntimeStatus.STOPPED
    assert runtime.frame_counter == 0
    assert len(runtime.buffered_results) == 0
    assert len(runtime.presence_manager.get_active_sessions()) == 0
