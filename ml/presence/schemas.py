import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class PresenceState(str, Enum):
    NOT_PRESENT = "NOT_PRESENT"
    CANDIDATE = "CANDIDATE"
    PRESENT = "PRESENT"
    GRACE = "GRACE"
    ABSENT = "ABSENT"


class PresenceEventType(str, Enum):
    ENTRY_CONFIRMED = "ENTRY_CONFIRMED"
    PRESENCE_UPDATED = "PRESENCE_UPDATED"
    GRACE_STARTED = "GRACE_STARTED"
    PRESENCE_RESUMED = "PRESENCE_RESUMED"
    SESSION_ENDED = "SESSION_ENDED"
    STATE_CHANGED = "STATE_CHANGED"


class PresenceMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    STRICT = "strict"


@dataclass
class PresenceConfig:
    """
    Configuration parameters for Presence and Session Management.

    Attributes:
        enabled (bool): Whether presence management is enabled.
        mode (str): Operating mode ("fast", "balanced", "strict").
        min_entry_observations (int): Minimum stable observations required to confirm presence.
        entry_window_seconds (float): Window duration to accumulate entry observations.
        grace_period_seconds (float): Duration to retain session before marking absent.
        max_session_duration_seconds (float): Maximum continuous session duration safeguard.
    """
    enabled: bool = True
    mode: str = "balanced"
    min_entry_observations: int = 3
    entry_window_seconds: float = 5.0
    grace_period_seconds: float = 10.0
    max_session_duration_seconds: float = 28800.0  # 8 hours default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "min_entry_observations": self.min_entry_observations,
            "entry_window_seconds": self.entry_window_seconds,
            "grace_period_seconds": self.grace_period_seconds,
            "max_session_duration_seconds": self.max_session_duration_seconds
        }


@dataclass
class PresenceEvent:
    """
    Structured presence state transition event.

    Attributes:
        event_type (PresenceEventType): Type of presence event.
        identity (str): Enrolled identity name.
        timestamp (datetime): Timezone-aware UTC timestamp of the event.
        previous_state (PresenceState): State before transition.
        new_state (PresenceState): State after transition.
        session_id: (Optional[str]): Associated active or completed session UUID.
        reason (str): Human-readable explanation of why transition occurred.
    """
    event_type: PresenceEventType
    identity: str
    timestamp: datetime
    previous_state: PresenceState
    new_state: PresenceState
    session_id: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value if isinstance(self.event_type, PresenceEventType) else str(self.event_type),
            "identity": self.identity,
            "timestamp": self.timestamp.isoformat(),
            "previous_state": self.previous_state.value if isinstance(self.previous_state, PresenceState) else str(self.previous_state),
            "new_state": self.new_state.value if isinstance(self.new_state, PresenceState) else str(self.new_state),
            "session_id": self.session_id,
            "reason": self.reason
        }


@dataclass
class PresenceSession:
    """
    In-memory representation of a confirmed presence session.

    Attributes:
        session_id (str): Unique session identifier.
        identity (str): Enrolled identity name.
        started_at (datetime): Timezone-aware UTC datetime when presence was confirmed.
        last_seen_at (datetime): Timezone-aware UTC datetime of the most recent observation.
        ended_at (Optional[datetime]): Timezone-aware UTC datetime when session was closed.
        state (PresenceState): Current state (PRESENT, GRACE, ABSENT).
        duration_seconds (float): Total active duration in seconds.
        observation_count (int): Total number of valid observations accumulated in session.
        interruption_count (int): Number of times session entered GRACE and resumed.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    identity: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    state: PresenceState = PresenceState.PRESENT
    duration_seconds: float = 0.0
    observation_count: int = 1
    interruption_count: int = 0

    @property
    def is_active(self) -> bool:
        """True if session is currently active (PRESENT or GRACE)."""
        return self.ended_at is None and self.state in (PresenceState.PRESENT, PresenceState.GRACE)

    def compute_duration(self, current_time: Optional[datetime] = None) -> float:
        """Computes current or finalized duration in seconds."""
        if self.ended_at is not None:
            return max(0.0, (self.ended_at - self.started_at).total_seconds())
        now = current_time or datetime.now(timezone.utc)
        return max(0.0, (now - self.started_at).total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "identity": self.identity,
            "started_at": self.started_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at is not None else None,
            "state": self.state.value if isinstance(self.state, PresenceState) else str(self.state),
            "is_active": self.is_active,
            "duration_seconds": round(float(self.duration_seconds), 2),
            "observation_count": int(self.observation_count),
            "interruption_count": int(self.interruption_count)
        }
