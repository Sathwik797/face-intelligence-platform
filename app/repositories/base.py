from abc import ABC, abstractmethod
from typing import Optional, List
from app.schemas.attendance import AttendanceRecord, SessionAuditEntry, AttendanceDailySummary


class BaseAttendanceRepository(ABC):
    """
    Abstract interface for attendance and session audit persistence.
    Enforces atomic operations and deterministic record retrieval.
    """

    @abstractmethod
    def upsert_attendance_record(self, record: AttendanceRecord) -> AttendanceRecord:
        """Atomically inserts or updates a daily AttendanceRecord."""
        pass

    @abstractmethod
    def get_attendance_record(self, identity: str, date_str: str) -> Optional[AttendanceRecord]:
        """Retrieves the AttendanceRecord for a given identity and calendar date."""
        pass

    @abstractmethod
    def list_attendance_records(
        self,
        date_str: Optional[str] = None,
        identity: Optional[str] = None
    ) -> List[AttendanceRecord]:
        """Lists attendance records filtered by optional date and/or identity."""
        pass

    @abstractmethod
    def record_session_audit(self, entry: SessionAuditEntry) -> None:
        """Records a finalized session into the immutable audit log."""
        pass

    @abstractmethod
    def list_session_audits(
        self,
        date_str: Optional[str] = None,
        identity: Optional[str] = None
    ) -> List[SessionAuditEntry]:
        """Lists session audit entries filtered by optional date and/or identity."""
        pass

    @abstractmethod
    def get_daily_summary(self, date_str: str) -> AttendanceDailySummary:
        """Computes statistical attendance summary for a target date."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all records (primarily for testing fixtures)."""
        pass
