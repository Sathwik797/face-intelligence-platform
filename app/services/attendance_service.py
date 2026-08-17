import io
import csv
import json
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Set, Tuple

from ml.runtime.schemas import RuntimeFrameResult
from ml.presence.schemas import PresenceEvent, PresenceEventType, PresenceSession, PresenceState
from app.schemas.attendance import (
    AttendanceRecord,
    SessionAuditEntry,
    AttendanceStatus,
    AttendanceConfig,
    AttendanceDailySummary
)
from app.repositories.base import BaseAttendanceRepository


class AttendanceService:
    """
    Attendance Business Engine.
    Consumes runtime PresenceEvents and PresenceSessions, enforces idempotency,
    aggregates cumulative dwell time, and persists daily records into BaseAttendanceRepository.
    """

    def __init__(
        self,
        repository: BaseAttendanceRepository,
        config: Optional[AttendanceConfig] = None
    ):
        self.repository = repository
        self.config = config or AttendanceConfig()
        self._lock = threading.Lock()

        # In-memory deduplication structures
        self._processed_session_ids: Set[str] = set()
        self._finalized_session_ids: Set[str] = set()

    def process_frame_result(self, result: RuntimeFrameResult) -> List[AttendanceRecord]:
        """
        Processes events and session transitions emitted by FaceIntelligenceRuntime in a frame tick.
        """
        updated_records = []
        with self._lock:
            # 1. Process explicit presence events
            for event in result.presence_events:
                rec = self._handle_presence_event(event)
                if rec is not None:
                    updated_records.append(rec)

            # 2. Process any newly closed sessions present in presence manager history
            # (Handled deterministically via SESSION_ENDED events or explicit audit)
        return updated_records

    def _handle_presence_event(self, event: PresenceEvent) -> Optional[AttendanceRecord]:
        if not event.identity or event.identity == "Unknown":
            return None

        event_dt = event.timestamp if isinstance(event.timestamp, datetime) else datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        date_str = event_dt.strftime("%Y-%m-%d")
        identity = event.identity

        existing_record = self.repository.get_attendance_record(identity, date_str)

        if event.event_type == PresenceEventType.ENTRY_CONFIRMED:
            return self._handle_entry_confirmed(identity, date_str, event_dt, event.session_id, existing_record)

        elif event.event_type == PresenceEventType.SESSION_ENDED:
            return self._handle_session_ended(identity, date_str, event_dt, event.session_id, event.reason, existing_record)

        elif event.event_type == PresenceEventType.PRESENCE_RESUMED:
            if existing_record and existing_record.status != AttendanceStatus.PRESENT:
                existing_record.status = AttendanceStatus.IN_PROGRESS
                existing_record.updated_at = event_dt.isoformat()
                return self.repository.upsert_attendance_record(existing_record)

        return None

    def _handle_entry_confirmed(
        self,
        identity: str,
        date_str: str,
        timestamp: datetime,
        session_id: Optional[str],
        existing: Optional[AttendanceRecord]
    ) -> AttendanceRecord:
        ts_iso = timestamp.isoformat()

        if existing is None:
            # First check-in of the day
            new_record = AttendanceRecord(
                identity=identity,
                date=date_str,
                first_check_in=ts_iso,
                last_check_out=None,
                total_dwell_seconds=0.0,
                session_count=1,
                status=AttendanceStatus.IN_PROGRESS if self.config.min_present_seconds > 0.0 else AttendanceStatus.PRESENT,
                created_at=ts_iso,
                updated_at=ts_iso
            )
            if session_id:
                self._processed_session_ids.add(session_id)
            return self.repository.upsert_attendance_record(new_record)
        else:
            # Same-day repeated entry / new session
            is_new_session = False
            if session_id and session_id not in self._processed_session_ids:
                self._processed_session_ids.add(session_id)
                existing.session_count += 1
                is_new_session = True

            existing.status = AttendanceStatus.IN_PROGRESS if self.config.min_present_seconds > 0.0 else AttendanceStatus.PRESENT
            existing.updated_at = ts_iso
            return self.repository.upsert_attendance_record(existing)

    def _handle_session_ended(
        self,
        identity: str,
        date_str: str,
        timestamp: datetime,
        session_id: Optional[str],
        reason: str,
        existing: Optional[AttendanceRecord]
    ) -> Optional[AttendanceRecord]:
        ts_iso = timestamp.isoformat()

        # Deduplicate session closure
        if session_id and session_id in self._finalized_session_ids:
            return existing
        if session_id:
            self._finalized_session_ids.add(session_id)

        # Estimate duration if session_id is available
        session_duration = 0.0
        if existing is not None:
            first_in_dt = datetime.fromisoformat(existing.first_check_in.replace("Z", "+00:00"))
            session_duration = max(0.0, (timestamp - first_in_dt).total_seconds())

        # Audit entry
        audit = SessionAuditEntry(
            session_id=session_id or f"session_{ts_iso}",
            identity=identity,
            date=date_str,
            started_at=existing.first_check_in if existing else ts_iso,
            ended_at=ts_iso,
            duration_seconds=session_duration,
            observation_count=1,
            interruption_count=0,
            closure_reason=reason or "session_ended"
        )
        self.repository.record_session_audit(audit)

        if existing is None:
            new_record = AttendanceRecord(
                identity=identity,
                date=date_str,
                first_check_in=ts_iso,
                last_check_out=ts_iso,
                total_dwell_seconds=session_duration,
                session_count=1,
                status=AttendanceStatus.PRESENT if session_duration >= self.config.min_present_seconds else AttendanceStatus.PARTIAL,
                created_at=ts_iso,
                updated_at=ts_iso
            )
            return self.repository.upsert_attendance_record(new_record)
        else:
            existing.last_check_out = ts_iso
            existing.total_dwell_seconds = max(existing.total_dwell_seconds, session_duration)
            if existing.total_dwell_seconds >= self.config.min_present_seconds:
                existing.status = AttendanceStatus.PRESENT
            else:
                existing.status = AttendanceStatus.PARTIAL
            existing.updated_at = ts_iso
            return self.repository.upsert_attendance_record(existing)

    def record_manual_session(self, session: PresenceSession, closure_reason: str = "normal_exit") -> AttendanceRecord:
        """Processes a finalized PresenceSession directly into attendance records."""
        with self._lock:
            dt = session.started_at
            date_str = dt.strftime("%Y-%m-%d")
            existing = self.repository.get_attendance_record(session.identity, date_str)

            audit = SessionAuditEntry(
                session_id=session.session_id,
                identity=session.identity,
                date=date_str,
                started_at=session.started_at.isoformat(),
                ended_at=session.ended_at.isoformat() if session.ended_at else session.last_seen_at.isoformat(),
                duration_seconds=session.duration_seconds,
                observation_count=session.observation_count,
                interruption_count=session.interruption_count,
                closure_reason=closure_reason
            )
            self.repository.record_session_audit(audit)

            if existing is None:
                new_record = AttendanceRecord(
                    identity=session.identity,
                    date=date_str,
                    first_check_in=session.started_at.isoformat(),
                    last_check_out=session.ended_at.isoformat() if session.ended_at else session.last_seen_at.isoformat(),
                    total_dwell_seconds=session.duration_seconds,
                    session_count=1,
                    status=AttendanceStatus.PRESENT if session.duration_seconds >= self.config.min_present_seconds else AttendanceStatus.PARTIAL,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    updated_at=datetime.now(timezone.utc).isoformat()
                )
                return self.repository.upsert_attendance_record(new_record)
            else:
                existing.last_check_out = session.ended_at.isoformat() if session.ended_at else session.last_seen_at.isoformat()
                existing.total_dwell_seconds += session.duration_seconds
                existing.session_count += 1
                if existing.total_dwell_seconds >= self.config.min_present_seconds:
                    existing.status = AttendanceStatus.PRESENT
                else:
                    existing.status = AttendanceStatus.PARTIAL
                existing.updated_at = datetime.now(timezone.utc).isoformat()
                return self.repository.upsert_attendance_record(existing)

    def list_records(
        self,
        date_str: Optional[str] = None,
        identity: Optional[str] = None
    ) -> List[AttendanceRecord]:
        return self.repository.list_attendance_records(date_str=date_str, identity=identity)

    def get_daily_summary(self, date_str: str) -> AttendanceDailySummary:
        return self.repository.get_daily_summary(date_str=date_str)

    def export_csv(self, date_str: Optional[str] = None) -> str:
        records = self.list_records(date_str=date_str)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "record_id",
            "identity",
            "date",
            "first_check_in",
            "last_check_out",
            "total_dwell_seconds",
            "session_count",
            "status",
            "updated_at"
        ])
        for r in records:
            writer.writerow([
                r.record_id,
                r.identity,
                r.date,
                r.first_check_in,
                r.last_check_out or "",
                f"{r.total_dwell_seconds:.2f}",
                r.session_count,
                r.status.value if isinstance(r.status, AttendanceStatus) else str(r.status),
                r.updated_at
            ])
        return output.getvalue()

    def export_json(self, date_str: Optional[str] = None) -> str:
        records = self.list_records(date_str=date_str)
        return json.dumps([r.to_dict() for r in records], indent=2)
