import uuid
from datetime import datetime, timezone, date
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class AttendanceStatus(str, Enum):
    """
    Standardized daily attendance status.
    - PRESENT: Confirmed entry and met configured dwell duration requirements.
    - IN_PROGRESS: Currently active session in progress on the given date.
    - PARTIAL: Completed sessions did not meet optional min_present_seconds threshold.
    """
    PRESENT = "PRESENT"
    IN_PROGRESS = "IN_PROGRESS"
    PARTIAL = "PARTIAL"


@dataclass
class AttendanceConfig:
    """
    Configuration parameters for the Attendance Business Engine.

    Attributes:
        min_present_seconds (float): Minimum cumulative dwell time required to achieve PRESENT status.
                                     Default 0.0 (any confirmed entry qualifies as PRESENT).
        timezone_name (str): Timezone identifier for daily date slicing (default 'UTC').
    """
    min_present_seconds: float = 0.0
    timezone_name: str = "UTC"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_present_seconds": self.min_present_seconds,
            "timezone_name": self.timezone_name
        }


@dataclass
class AttendanceRecord:
    """
    Idempotent daily aggregated attendance record for an enrolled individual.
    Uniquely identified by (identity, date).
    """
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    identity: str = ""
    date: str = ""  # YYYY-MM-DD
    first_check_in: str = ""  # ISO-8601
    last_check_out: Optional[str] = None  # ISO-8601
    total_dwell_seconds: float = 0.0
    session_count: int = 0
    status: AttendanceStatus = AttendanceStatus.PRESENT
    last_confidence_score: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "identity": self.identity,
            "date": self.date,
            "first_check_in": self.first_check_in,
            "last_check_out": self.last_check_out,
            "total_dwell_seconds": round(float(self.total_dwell_seconds), 2),
            "session_count": int(self.session_count),
            "status": self.status.value if isinstance(self.status, AttendanceStatus) else str(self.status),
            "last_confidence_score": round(float(self.last_confidence_score), 4),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class SessionAuditEntry:
    """
    Persistent audit ledger entry representing a completed or finalized continuous presence session.
    """
    session_id: str
    identity: str
    date: str  # YYYY-MM-DD
    started_at: str  # ISO-8601
    ended_at: str  # ISO-8601
    duration_seconds: float
    observation_count: int
    interruption_count: int
    closure_reason: str = "normal_exit"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "identity": self.identity,
            "date": self.date,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(float(self.duration_seconds), 2),
            "observation_count": int(self.observation_count),
            "interruption_count": int(self.interruption_count),
            "closure_reason": self.closure_reason
        }


@dataclass
class AttendanceDailySummary:
    """
    Aggregated statistical summary of attendance records for a target calendar date.
    """
    date: str
    total_records: int = 0
    total_present: int = 0
    total_in_progress: int = 0
    total_partial: int = 0
    total_dwell_seconds: float = 0.0
    average_dwell_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "total_records": int(self.total_records),
            "total_present": int(self.total_present),
            "total_in_progress": int(self.total_in_progress),
            "total_partial": int(self.total_partial),
            "total_dwell_seconds": round(float(self.total_dwell_seconds), 2),
            "average_dwell_seconds": round(float(self.average_dwell_seconds), 2)
        }
