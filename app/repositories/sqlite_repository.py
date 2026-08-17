import os
import sqlite3
import threading
from typing import Optional, List
from datetime import datetime, timezone

from app.schemas.attendance import (
    AttendanceRecord,
    SessionAuditEntry,
    AttendanceStatus,
    AttendanceDailySummary
)
from app.schemas.identities import EnrolledIdentityInfo
from app.repositories.base import BaseAttendanceRepository


class SQLiteAttendanceRepository(BaseAttendanceRepository):
    """
    Thread-safe SQLite implementation of BaseAttendanceRepository.
    Enforces atomic operations, schema migrations, and unique constraints.
    """

    def __init__(self, db_path: str = "data/attendance.db"):
        self.db_path = db_path
        self._lock = threading.Lock()

        # Ensure directory exists if not an in-memory database
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self._conn = None
        if self.db_path == ":memory:":
            # Retain open connection for in-memory database
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initializes tables and indexes deterministically."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        record_id TEXT PRIMARY KEY,
                        identity TEXT NOT NULL,
                        date TEXT NOT NULL,
                        first_check_in TEXT NOT NULL,
                        last_check_out TEXT,
                        total_dwell_seconds REAL NOT NULL DEFAULT 0.0,
                        session_count INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL,
                        last_confidence_score REAL NOT NULL DEFAULT 1.0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(identity, date)
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_records(date);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_identity ON attendance_records(identity);")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_audit_log (
                        session_id TEXT PRIMARY KEY,
                        identity TEXT NOT NULL,
                        date TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        observation_count INTEGER NOT NULL,
                        interruption_count INTEGER NOT NULL,
                        closure_reason TEXT NOT NULL
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_date ON session_audit_log(date);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_identity ON session_audit_log(identity);")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS enrolled_identities (
                        identity TEXT PRIMARY KEY,
                        template_count INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        notes TEXT
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrolled_identity ON enrolled_identities(identity);")
                conn.commit()
            finally:
                if self._conn is None:
                    conn.close()

    def upsert_attendance_record(self, record: AttendanceRecord) -> AttendanceRecord:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                status_val = record.status.value if isinstance(record.status, AttendanceStatus) else str(record.status)
                cursor.execute("""
                    INSERT INTO attendance_records (
                        record_id, identity, date, first_check_in, last_check_out,
                        total_dwell_seconds, session_count, status, last_confidence_score,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(identity, date) DO UPDATE SET
                        first_check_in = excluded.first_check_in,
                        last_check_out = excluded.last_check_out,
                        total_dwell_seconds = excluded.total_dwell_seconds,
                        session_count = excluded.session_count,
                        status = excluded.status,
                        last_confidence_score = excluded.last_confidence_score,
                        updated_at = excluded.updated_at
                """, (
                    record.record_id,
                    record.identity,
                    record.date,
                    record.first_check_in,
                    record.last_check_out,
                    float(record.total_dwell_seconds),
                    int(record.session_count),
                    status_val,
                    float(record.last_confidence_score),
                    record.created_at,
                    record.updated_at
                ))
                conn.commit()
                return record
            finally:
                if self._conn is None:
                    conn.close()

    def get_attendance_record(self, identity: str, date_str: str) -> Optional[AttendanceRecord]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM attendance_records WHERE identity = ? AND date = ?",
                    (identity, date_str)
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_record(row)
            finally:
                if self._conn is None:
                    conn.close()

    def list_attendance_records(
        self,
        date_str: Optional[str] = None,
        identity: Optional[str] = None
    ) -> List[AttendanceRecord]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM attendance_records WHERE 1=1"
                params = []
                if date_str:
                    query += " AND date = ?"
                    params.append(date_str)
                if identity:
                    query += " AND identity = ?"
                    params.append(identity)
                query += " ORDER BY first_check_in ASC"

                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [self._row_to_record(r) for r in rows]
            finally:
                if self._conn is None:
                    conn.close()

    def record_session_audit(self, entry: SessionAuditEntry) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO session_audit_log (
                        session_id, identity, date, started_at, ended_at,
                        duration_seconds, observation_count, interruption_count, closure_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.session_id,
                    entry.identity,
                    entry.date,
                    entry.started_at,
                    entry.ended_at,
                    float(entry.duration_seconds),
                    int(entry.observation_count),
                    int(entry.interruption_count),
                    entry.closure_reason
                ))
                conn.commit()
            finally:
                if self._conn is None:
                    conn.close()

    def list_session_audits(
        self,
        date_str: Optional[str] = None,
        identity: Optional[str] = None
    ) -> List[SessionAuditEntry]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM session_audit_log WHERE 1=1"
                params = []
                if date_str:
                    query += " AND date = ?"
                    params.append(date_str)
                if identity:
                    query += " AND identity = ?"
                    params.append(identity)
                query += " ORDER BY started_at ASC"

                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [self._row_to_audit(r) for r in rows]
            finally:
                if self._conn is None:
                    conn.close()

    def get_daily_summary(self, date_str: str) -> AttendanceDailySummary:
        records = self.list_attendance_records(date_str=date_str)
        total_records = len(records)
        total_present = sum(1 for r in records if r.status == AttendanceStatus.PRESENT)
        total_in_progress = sum(1 for r in records if r.status == AttendanceStatus.IN_PROGRESS)
        total_partial = sum(1 for r in records if r.status == AttendanceStatus.PARTIAL)
        total_dwell = sum(r.total_dwell_seconds for r in records)
        avg_dwell = (total_dwell / total_records) if total_records > 0 else 0.0

        return AttendanceDailySummary(
            date=date_str,
            total_records=total_records,
            total_present=total_present,
            total_in_progress=total_in_progress,
            total_partial=total_partial,
            total_dwell_seconds=total_dwell,
            average_dwell_seconds=avg_dwell
        )

    def upsert_enrolled_identity(self, info: EnrolledIdentityInfo) -> EnrolledIdentityInfo:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO enrolled_identities (
                        identity, template_count, created_at, updated_at, notes
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(identity) DO UPDATE SET
                        template_count = excluded.template_count,
                        updated_at = excluded.updated_at,
                        notes = excluded.notes
                """, (
                    info.identity,
                    int(info.template_count),
                    info.created_at,
                    info.updated_at,
                    info.notes
                ))
                conn.commit()
                return info
            finally:
                if self._conn is None:
                    conn.close()

    def get_enrolled_identity(self, identity: str) -> Optional[EnrolledIdentityInfo]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM enrolled_identities WHERE identity = ?", (identity,))
                row = cursor.fetchone()
                if row is None:
                    return None
                return EnrolledIdentityInfo(
                    identity=row["identity"],
                    template_count=int(row["template_count"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    notes=row["notes"]
                )
            finally:
                if self._conn is None:
                    conn.close()

    def list_enrolled_identities(self) -> List[EnrolledIdentityInfo]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM enrolled_identities ORDER BY identity ASC")
                rows = cursor.fetchall()
                return [
                    EnrolledIdentityInfo(
                        identity=r["identity"],
                        template_count=int(r["template_count"]),
                        created_at=r["created_at"],
                        updated_at=r["updated_at"],
                        notes=r["notes"]
                    )
                    for r in rows
                ]
            finally:
                if self._conn is None:
                    conn.close()

    def delete_enrolled_identity(self, identity: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM enrolled_identities WHERE identity = ?", (identity,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                if self._conn is None:
                    conn.close()

    def clear(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM attendance_records;")
                cursor.execute("DELETE FROM session_audit_log;")
                cursor.execute("DELETE FROM enrolled_identities;")
                conn.commit()
            finally:
                if self._conn is None:
                    conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AttendanceRecord:
        return AttendanceRecord(
            record_id=row["record_id"],
            identity=row["identity"],
            date=row["date"],
            first_check_in=row["first_check_in"],
            last_check_out=row["last_check_out"],
            total_dwell_seconds=float(row["total_dwell_seconds"]),
            session_count=int(row["session_count"]),
            status=AttendanceStatus(row["status"]),
            last_confidence_score=float(row["last_confidence_score"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    @staticmethod
    def _row_to_audit(row: sqlite3.Row) -> SessionAuditEntry:
        return SessionAuditEntry(
            session_id=row["session_id"],
            identity=row["identity"],
            date=row["date"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_seconds=float(row["duration_seconds"]),
            observation_count=int(row["observation_count"]),
            interruption_count=int(row["interruption_count"]),
            closure_reason=row["closure_reason"]
        )
