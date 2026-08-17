from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np

from ml.pipeline import ModernRecognitionResult
from ml.temporal.schemas import TemporalRecognitionResult
from ml.presence.schemas import PresenceEvent, PresenceSession


class RuntimeStatus(str, Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


@dataclass
class RuntimeConfig:
    """
    Minimal configuration parameters for runtime orchestration.

    Attributes:
        enabled (bool): Whether runtime orchestration is active.
        heartbeat_interval_seconds (float): Interval to trigger background presence grace checks.
        auto_tick (bool): Whether process_frame should automatically trigger periodic grace checks.
        max_buffered_history (int): Maximum count of processed frame metadata kept in memory.
    """
    enabled: bool = True
    heartbeat_interval_seconds: float = 1.0
    auto_tick: bool = True
    max_buffered_history: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "auto_tick": self.auto_tick,
            "max_buffered_history": self.max_buffered_history
        }


@dataclass
class StageLatencyMetrics:
    """
    Latency breakdown of runtime processing stages in milliseconds.

    Attributes:
        recognition_ms (float): Recognition pipeline execution duration.
        temporal_ms (float): Temporal stabilization processing duration.
        presence_ms (float): Presence state machine transition duration.
        total_ms (float): End-to-end frame processing duration.
    """
    recognition_ms: float = 0.0
    temporal_ms: float = 0.0
    presence_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recognition_ms": round(float(self.recognition_ms), 2),
            "temporal_ms": round(float(self.temporal_ms), 2),
            "presence_ms": round(float(self.presence_ms), 2),
            "total_ms": round(float(self.total_ms), 2)
        }


@dataclass
class RuntimeFrameResult:
    """
    Structured end-to-end output for a single processed frame.

    Attributes:
        frame_index (int): Monotonically increasing frame index.
        timestamp (datetime): Timezone-aware UTC timestamp of frame.
        recognition (ModernRecognitionResult): Frame-level face detection & recognition result.
        temporal (TemporalRecognitionResult): Multi-frame temporal consensus result.
        presence_events (List[PresenceEvent]): List of presence state change events emitted in this frame.
        active_sessions (List[PresenceSession]): Current active in-memory presence sessions.
        latencies (StageLatencyMetrics): Latency breakdown across pipeline stages.
        error: (Optional[str]): Error message if frame processing failed gracefully.
    """
    frame_index: int
    timestamp: datetime
    recognition: ModernRecognitionResult
    temporal: TemporalRecognitionResult
    presence_events: List[PresenceEvent] = field(default_factory=list)
    active_sessions: List[PresenceSession] = field(default_factory=list)
    latencies: StageLatencyMetrics = field(default_factory=StageLatencyMetrics)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp": self.timestamp.isoformat(),
            "recognition": self.recognition.to_dict() if hasattr(self.recognition, "to_dict") else None,
            "temporal": self.temporal.to_dict() if hasattr(self.temporal, "to_dict") else None,
            "presence_events": [e.to_dict() for e in self.presence_events],
            "active_sessions": [s.to_dict() for s in self.active_sessions],
            "latencies": self.latencies.to_dict(),
            "error": self.error
        }
