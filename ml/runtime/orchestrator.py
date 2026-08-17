import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
import numpy as np

from ml.pipeline import ModernRecognitionPipeline, ModernRecognitionResult
from ml.temporal.schemas import RecognitionObservation, TemporalRecognitionResult
from ml.temporal.stabilizer import TemporalIdentityStabilizer
from ml.presence.schemas import PresenceEvent, PresenceSession, PresenceEventType, PresenceState
from ml.presence.manager import PresenceManager
from ml.runtime.schemas import (
    RuntimeStatus,
    RuntimeConfig,
    StageLatencyMetrics,
    RuntimeFrameResult
)
from ml.runtime.frame_source import BaseFrameSource


class FaceIntelligenceRuntime:
    """
    Unified end-to-end runtime orchestrator.

    Composes ModernRecognitionPipeline, TemporalIdentityStabilizer, and PresenceManager
    into an isolated, testable, and fault-tolerant processing engine.
    """

    def __init__(
        self,
        recognition_pipeline: ModernRecognitionPipeline,
        temporal_stabilizer: TemporalIdentityStabilizer,
        presence_manager: PresenceManager,
        config: Optional[RuntimeConfig] = None,
        clock: Optional[Callable[[], datetime]] = None
    ):
        self.recognition_pipeline = recognition_pipeline
        self.temporal_stabilizer = temporal_stabilizer
        self.presence_manager = presence_manager
        self.config = config or RuntimeConfig()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

        self.status = RuntimeStatus.STOPPED
        self.frame_counter = 0
        self.last_tick_time: Optional[datetime] = None
        self.buffered_results: List[RuntimeFrameResult] = []

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        gallery_path: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None
    ) -> "FaceIntelligenceRuntime":
        """Factory method instantiating the entire runtime stack from system config."""
        # 1. Build Modern Recognition Pipeline (YuNet + FQA + ArcFace + Gallery)
        rec_pipeline = ModernRecognitionPipeline.from_config(
            config=config,
            gallery_path=gallery_path
        )

        # 2. Build Temporal Identity Stabilizer
        temp_cfg = config.get("temporal", {})
        temp_stabilizer = TemporalIdentityStabilizer.from_config(temp_cfg)

        # 3. Build Presence Manager
        pres_cfg = config.get("presence", {})
        pres_manager = PresenceManager.from_config(pres_cfg, clock=clock)

        # 4. Build Runtime Config
        runtime_dict = config.get("runtime", {})
        runtime_cfg = RuntimeConfig(
            enabled=runtime_dict.get("enabled", True),
            heartbeat_interval_seconds=float(runtime_dict.get("heartbeat_interval_seconds", 1.0)),
            auto_tick=runtime_dict.get("auto_tick", True),
            max_buffered_history=int(runtime_dict.get("max_buffered_history", 100))
        )

        return cls(
            recognition_pipeline=rec_pipeline,
            temporal_stabilizer=temp_stabilizer,
            presence_manager=pres_manager,
            config=runtime_cfg,
            clock=clock
        )

    def start(self):
        """Starts the runtime orchestrator."""
        self.status = RuntimeStatus.RUNNING
        self.last_tick_time = self._now()

    def _now(self, timestamp: Optional[datetime] = None) -> datetime:
        if timestamp is not None:
            return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        return self.clock()

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        timestamp: Optional[datetime] = None
    ) -> RuntimeFrameResult:
        """
        Executes end-to-end processing on a single frame.

        Args:
            rgb_frame (np.ndarray): RGB image array (H, W, 3).
            timestamp (Optional[datetime]): Timezone-aware timestamp (or injected clock).

        Returns:
            RuntimeFrameResult: Structured output of recognition, temporal, and presence states.
        """
        t0_total = time.perf_counter()
        now = self._now(timestamp)
        self.frame_counter += 1
        current_frame_idx = self.frame_counter

        events: List[PresenceEvent] = []
        error_msg: Optional[str] = None

        # 1. Stage 1: Recognition Pipeline Execution
        t0_rec = time.perf_counter()
        try:
            rec_result: ModernRecognitionResult = self.recognition_pipeline.recognize(rgb_frame)
        except Exception as e:
            rec_result = ModernRecognitionResult(
                identity=None,
                best_candidate="Unknown",
                similarity=-1.0,
                threshold=getattr(self.recognition_pipeline, "threshold", 0.24),
                recognized=False,
                reason=f"pipeline_error: {str(e)}"
            )
            error_msg = f"Recognition error: {str(e)}"
        t1_rec = time.perf_counter()
        rec_latency_ms = (t1_rec - t0_rec) * 1000.0

        # 2. Stage 2: Temporal Identity Stabilization
        t0_temp = time.perf_counter()
        try:
            obs = RecognitionObservation.from_recognition_result(
                result=rec_result,
                frame_index=current_frame_idx,
                timestamp=now.timestamp()
            )
            temp_result: TemporalRecognitionResult = self.temporal_stabilizer.update(obs)
        except Exception as e:
            temp_result = TemporalRecognitionResult(
                stable_identity=None,
                state="unstable",
                confidence_score=0.0,
                observations_count=0,
                consecutive_stable_count=0,
                active_candidate="Unknown",
                is_stable=False
            )
            error_msg = (error_msg + f" | Temporal error: {str(e)}") if error_msg else f"Temporal error: {str(e)}"
        t1_temp = time.perf_counter()
        temp_latency_ms = (t1_temp - t0_temp) * 1000.0

        # 3. Stage 3: Presence & Session State Machine
        t0_pres = time.perf_counter()
        try:
            pres_events = self.presence_manager.update(temp_result, timestamp=now)
            events.extend(pres_events)

            # Auto-tick check for expiring grace timers
            if self.config.auto_tick:
                if self.last_tick_time is None or (now - self.last_tick_time).total_seconds() >= self.config.heartbeat_interval_seconds:
                    tick_events = self.presence_manager.tick(timestamp=now)
                    events.extend(tick_events)
                    self.last_tick_time = now

            active_sessions = self.presence_manager.get_active_sessions()
        except Exception as e:
            active_sessions = []
            error_msg = (error_msg + f" | Presence error: {str(e)}") if error_msg else f"Presence error: {str(e)}"
        t1_pres = time.perf_counter()
        pres_latency_ms = (t1_pres - t0_pres) * 1000.0

        t1_total = time.perf_counter()
        total_latency_ms = (t1_total - t0_total) * 1000.0

        latencies = StageLatencyMetrics(
            recognition_ms=rec_latency_ms,
            temporal_ms=temp_latency_ms,
            presence_ms=pres_latency_ms,
            total_ms=total_latency_ms
        )

        frame_result = RuntimeFrameResult(
            frame_index=current_frame_idx,
            timestamp=now,
            recognition=rec_result,
            temporal=temp_result,
            presence_events=events,
            active_sessions=active_sessions,
            latencies=latencies,
            error=error_msg
        )

        if self.config.max_buffered_history > 0:
            self.buffered_results.append(frame_result)
            if len(self.buffered_results) > self.config.max_buffered_history:
                self.buffered_results.pop(0)

        return frame_result

    def tick(self, timestamp: Optional[datetime] = None) -> List[PresenceEvent]:
        """Manually triggers background grace timeout sweep across all tracked identities."""
        now = self._now(timestamp)
        self.last_tick_time = now
        return self.presence_manager.tick(timestamp=now)

    def stop(self, reason: str = "runtime_shutdown") -> List[PresenceEvent]:
        """
        Gracefully stops runtime and finalizes any currently active sessions.

        Distinguishes explicit runtime shutdown from natural absence/grace timeouts.
        """
        self.status = RuntimeStatus.STOPPED
        now = self._now()
        shutdown_events: List[PresenceEvent] = []

        # Gracefully finalize all active sessions
        for identity, tracker in list(self.presence_manager.trackers.items()):
            if tracker.state in (PresenceState.PRESENT, PresenceState.GRACE) and tracker.active_session is not None:
                evs = tracker._close_session(
                    now=now,
                    reason=f"session closed due to {reason}"
                )
                shutdown_events.extend(evs)
                self.presence_manager._check_archived_sessions(tracker, evs)

        return shutdown_events

    def reset(self):
        """Resets all runtime state, temporal stabilizer, presence manager, and counters."""
        self.status = RuntimeStatus.STOPPED
        self.frame_counter = 0
        self.last_tick_time = None
        self.buffered_results.clear()
        self.temporal_stabilizer.reset()
        self.presence_manager.reset()

    def run_stream(
        self,
        frame_source: BaseFrameSource,
        max_frames: Optional[int] = None,
        on_frame: Optional[Callable[[RuntimeFrameResult], None]] = None
    ) -> List[RuntimeFrameResult]:
        """
        Convenience execution loop processing a frame source stream to completion.
        """
        self.start()
        results: List[RuntimeFrameResult] = []
        try:
            for idx, frame, ts in frame_source:
                res = self.process_frame(frame, timestamp=ts)
                results.append(res)
                if on_frame is not None:
                    on_frame(res)
                if max_frames and len(results) >= max_frames:
                    break
        finally:
            frame_source.release()
            self.stop()

        return results
