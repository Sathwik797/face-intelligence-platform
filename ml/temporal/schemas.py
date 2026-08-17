import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple


class TemporalMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    STABLE = "stable"


class TemporalState(str, Enum):
    UNSTABLE = "unstable"
    STABLE = "stable"
    UNKNOWN = "unknown"
    SWITCHING = "switching"


@dataclass
class RecognitionObservation:
    """
    Structured representation of a single frame-level recognition observation.

    Attributes:
        timestamp (float): UNIX timestamp of observation in seconds.
        frame_index (int): Monotonically increasing video frame counter or index.
        identity (Optional[str]): Recognized identity name, or None if rejected / Unknown.
        best_candidate (str): Top gallery candidate identity name.
        similarity (float): Cosine similarity score [-1.0, 1.0].
        threshold (float): Recognition decision threshold.
        recognized (bool): Whether similarity >= threshold.
        quality_status (str): "good", "poor", or "none".
        quality_score (Optional[float]): Overall composite quality index in [0.0, 1.0].
        bbox (Optional[Tuple[int, int, int, int]]): CSS bounding box (top, right, bottom, left).
    """
    timestamp: float
    frame_index: int
    identity: Optional[str]
    best_candidate: str
    similarity: float
    threshold: float
    recognized: bool
    quality_status: str = "none"
    quality_score: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None

    @classmethod
    def from_recognition_result(
        cls,
        result: Any,
        frame_index: int = 0,
        timestamp: Optional[float] = None
    ) -> "RecognitionObservation":
        """Factory method to construct an observation from a ModernRecognitionResult."""
        ts = timestamp if timestamp is not None else time.time()
        q_status = "none"
        q_score = None
        if hasattr(result, "quality") and result.quality is not None:
            q_status = getattr(result.quality, "quality_status", "none")
            q_score = getattr(result.quality, "overall_quality", None)

        return cls(
            timestamp=ts,
            frame_index=frame_index,
            identity=result.identity if result.recognized else None,
            best_candidate=result.best_candidate,
            similarity=float(result.similarity),
            threshold=float(result.threshold),
            recognized=bool(result.recognized),
            quality_status=q_status,
            quality_score=q_score,
            bbox=result.bbox
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(float(self.timestamp), 3),
            "frame_index": int(self.frame_index),
            "identity": self.identity,
            "best_candidate": self.best_candidate,
            "similarity": round(float(self.similarity), 4),
            "threshold": round(float(self.threshold), 4),
            "recognized": self.recognized,
            "quality_status": self.quality_status,
            "quality_score": round(float(self.quality_score), 4) if self.quality_score is not None else None,
            "bbox": list(self.bbox) if self.bbox is not None else None
        }


@dataclass
class TemporalPolicyConfig:
    """
    Configurable parameters governing temporal identity stabilization.

    Attributes:
        window_size (int): Size of sliding observation buffer (number of frames).
        min_observations (int): Minimum valid observations required before declaring STABLE.
        min_stable_ratio (float): Ratio of window observations supporting active identity in [0.0, 1.0].
        max_unknown_observations (int): Consecutive Unknown frames tolerated before expiring stable identity.
        challenger_switch_threshold (int): Consecutive observations required for a new identity to replace stable identity.
        max_gap_seconds (float): Maximum elapsed seconds between frames before expiring temporal context.
        mode (str): Operating mode ("fast", "balanced", "stable").
    """
    window_size: int = 7
    min_observations: int = 4
    min_stable_ratio: float = 0.70
    max_unknown_observations: int = 3
    challenger_switch_threshold: int = 3
    max_gap_seconds: float = 2.0
    mode: str = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "min_observations": self.min_observations,
            "min_stable_ratio": self.min_stable_ratio,
            "max_unknown_observations": self.max_unknown_observations,
            "challenger_switch_threshold": self.challenger_switch_threshold,
            "max_gap_seconds": self.max_gap_seconds,
            "mode": self.mode
        }


@dataclass
class TemporalRecognitionResult:
    """
    Output structure representing a temporally-stabilized identity decision.

    Attributes:
        stable_identity (Optional[str]): Stabilized identity name (None if Unstable/Unknown).
        state (TemporalState): Current stability state (STABLE, UNSTABLE, UNKNOWN, SWITCHING).
        confidence_score (float): Temporal evidence strength in [0.0, 1.0].
        observations_count (int): Number of valid observations in current window.
        consecutive_stable_count (int): Monotonic count of consecutive frames supporting stable identity.
        active_candidate (str): Current leading identity candidate.
        challenger_identity (Optional[str]): Competing identity candidate challenging the current stable identity.
        challenger_evidence (int): Consecutive count of challenger observations.
        is_stable (bool): True if state == TemporalState.STABLE.
        latest_observation (Optional[RecognitionObservation]): The most recent frame observation.
    """
    stable_identity: Optional[str]
    state: TemporalState
    confidence_score: float
    observations_count: int
    consecutive_stable_count: int
    active_candidate: str
    challenger_identity: Optional[str] = None
    challenger_evidence: int = 0
    is_stable: bool = False
    latest_observation: Optional[RecognitionObservation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stable_identity": self.stable_identity,
            "state": self.state.value if isinstance(self.state, TemporalState) else str(self.state),
            "is_stable": self.is_stable,
            "confidence_score": round(float(self.confidence_score), 4),
            "observations_count": int(self.observations_count),
            "consecutive_stable_count": int(self.consecutive_stable_count),
            "active_candidate": self.active_candidate,
            "challenger_identity": self.challenger_identity,
            "challenger_evidence": int(self.challenger_evidence),
            "latest_observation": self.latest_observation.to_dict() if self.latest_observation is not None else None
        }
