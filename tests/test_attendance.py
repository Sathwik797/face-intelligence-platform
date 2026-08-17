import os
import io
import csv
import json
import pytest
import threading
from datetime import datetime, timezone, timedelta
import numpy as np

from app import create_app
from app.schemas.attendance import (
    AttendanceRecord,
    SessionAuditEntry,
    AttendanceStatus,
    AttendanceConfig,
    AttendanceDailySummary
)
from app.repositories.sqlite_repository import SQLiteAttendanceRepository
from app.services.attendance_service import AttendanceService
from app.services.runtime_service import RuntimeService
from ml.runtime import (
    FaceIntelligenceRuntime,
    RuntimeStatus,
    RuntimeConfig,
    StageLatencyMetrics,
    RuntimeFrameResult
)
from ml.pipeline import ModernRecognitionResult
from ml.temporal.schemas import TemporalPolicyConfig, TemporalRecognitionResult, TemporalState
from ml.temporal.stabilizer import TemporalIdentityStabilizer
from ml.presence.schemas import PresenceMode, PresenceState, PresenceEventType, PresenceEvent, PresenceSession
from ml.presence.state_machine import PRESENCE_PRESETS
from ml.presence.manager import PresenceManager


@pytest.fixture
def in_memory_repo():
    repo = SQLiteAttendanceRepository(db_path=":memory:")
    yield repo
    repo.clear()


@pytest.fixture
def attendance_service(in_memory_repo):
    return AttendanceService(repository=in_memory_repo, config=AttendanceConfig(min_present_seconds=0.0))


def test_first_entry_creates_daily_record(attendance_service):
    t0 = datetime(2026, 8, 17, 9, 0, 0, tzinfo=timezone.utc)
    ev = PresenceEvent(
        event_type=PresenceEventType.ENTRY_CONFIRMED,
        identity="Alice",
        timestamp=t0,
        previous_state=PresenceState.CANDIDATE,
        new_state=PresenceState.PRESENT,
        session_id="session_1"
    )

    rec = attendance_service._handle_presence_event(ev)
    assert rec is not None
    assert rec.identity == "Alice"
    assert rec.date == "2026-08-17"
    assert rec.first_check_in == t0.isoformat()
    assert rec.session_count == 1
    assert rec.status == AttendanceStatus.PRESENT

    # Verify in repository
    persisted = attendance_service.repository.get_attendance_record("Alice", "2026-08-17")
    assert persisted is not None
    assert persisted.first_check_in == t0.isoformat()


def test_repeated_same_day_entry_idempotency(attendance_service):
    t0 = datetime(2026, 8, 17, 9, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 8, 17, 14, 0, 0, tzinfo=timezone.utc)

    # First session entry
    ev1 = PresenceEvent(
        event_type=PresenceEventType.ENTRY_CONFIRMED,
        identity="Alice",
        timestamp=t0,
        previous_state=PresenceState.CANDIDATE,
        new_state=PresenceState.PRESENT,
        session_id="session_1"
    )
    attendance_service._handle_presence_event(ev1)

    # Second session entry later same day
    ev2 = PresenceEvent(
        event_type=PresenceEventType.ENTRY_CONFIRMED,
        identity="Alice",
        timestamp=t1,
        previous_state=PresenceState.CANDIDATE,
        new_state=PresenceState.PRESENT,
        session_id="session_2"
    )
    rec2 = attendance_service._handle_presence_event(ev2)

    assert rec2.session_count == 2
    assert rec2.first_check_in == t0.isoformat()

    # Verify no duplicate daily rows in repo
    all_alice_records = attendance_service.list_records(date_str="2026-08-17", identity="Alice")
    assert len(all_alice_records) == 1


def test_session_ended_updates_checkout_and_duration(attendance_service):
    t0 = datetime(2026, 8, 17, 9, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 8, 17, 9, 30, 0, tzinfo=timezone.utc)

    ev_in = PresenceEvent(
        event_type=PresenceEventType.ENTRY_CONFIRMED,
        identity="Alice",
        timestamp=t0,
        previous_state=PresenceState.CANDIDATE,
        new_state=PresenceState.PRESENT,
        session_id="session_1"
    )
    attendance_service._handle_presence_event(ev_in)

    ev_out = PresenceEvent(
        event_type=PresenceEventType.SESSION_ENDED,
        identity="Alice",
        timestamp=t1,
        previous_state=PresenceState.GRACE,
        new_state=PresenceState.ABSENT,
        session_id="session_1",
        reason="absence_timeout"
    )
    rec_out = attendance_service._handle_presence_event(ev_out)

    assert rec_out.last_check_out == t1.isoformat()
    assert rec_out.total_dwell_seconds >= 1800.0

    # Verify session audit log entry was written
    audits = attendance_service.repository.list_session_audits(date_str="2026-08-17", identity="Alice")
    assert len(audits) == 1
    assert audits[0].session_id == "session_1"
    assert audits[0].duration_seconds >= 1800.0


def test_multiple_sessions_dwell_accumulation(attendance_service):
    t0 = datetime(2026, 8, 17, 9, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 8, 17, 9, 15, 0, tzinfo=timezone.utc)

    # Session 1 direct recording
    sess1 = PresenceSession(
        session_id="s1",
        identity="Bob",
        started_at=t0,
        last_seen_at=t1,
        ended_at=t1,
        state=PresenceState.ABSENT,
        duration_seconds=900.0,
        observation_count=30,
        interruption_count=0
    )
    attendance_service.record_manual_session(sess1)

    # Session 2 later
    t2 = datetime(2026, 8, 17, 13, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 8, 17, 13, 30, 0, tzinfo=timezone.utc)
    sess2 = PresenceSession(
        session_id="s2",
        identity="Bob",
        started_at=t2,
        last_seen_at=t3,
        ended_at=t3,
        state=PresenceState.ABSENT,
        duration_seconds=1800.0,
        observation_count=60,
        interruption_count=1
    )
    attendance_service.record_manual_session(sess2)

    rec = attendance_service.repository.get_attendance_record("Bob", "2026-08-17")
    assert rec is not None
    assert rec.session_count == 2
    assert rec.total_dwell_seconds == 2700.0
    assert rec.first_check_in == t0.isoformat()
    assert rec.last_check_out == t3.isoformat()


def test_idempotent_duplicate_events_ignored(attendance_service):
    t0 = datetime(2026, 8, 17, 9, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 8, 17, 9, 10, 0, tzinfo=timezone.utc)

    ev_in = PresenceEvent(
        event_type=PresenceEventType.ENTRY_CONFIRMED,
        identity="Charlie",
        timestamp=t0,
        previous_state=PresenceState.CANDIDATE,
        new_state=PresenceState.PRESENT,
        session_id="sess_c1"
    )
    attendance_service._handle_presence_event(ev_in)

    ev_out = PresenceEvent(
        event_type=PresenceEventType.SESSION_ENDED,
        identity="Charlie",
        timestamp=t1,
        previous_state=PresenceState.GRACE,
        new_state=PresenceState.ABSENT,
        session_id="sess_c1"
    )

    # Trigger session ended twice
    attendance_service._handle_presence_event(ev_out)
    attendance_service._handle_presence_event(ev_out)

    audits = attendance_service.repository.list_session_audits(date_str="2026-08-17", identity="Charlie")
    assert len(audits) == 1


def test_sqlite_repository_persistence_across_instances(tmp_path):
    db_file = str(tmp_path / "test_persist.db")
    repo1 = SQLiteAttendanceRepository(db_path=db_file)

    rec = AttendanceRecord(
        identity="Dave",
        date="2026-08-17",
        first_check_in="2026-08-17T09:00:00+00:00",
        total_dwell_seconds=600.0,
        session_count=1,
        status=AttendanceStatus.PRESENT
    )
    repo1.upsert_attendance_record(rec)

    # Second repo instance pointing to same file
    repo2 = SQLiteAttendanceRepository(db_path=db_file)
    retrieved = repo2.get_attendance_record("Dave", "2026-08-17")
    assert retrieved is not None
    assert retrieved.identity == "Dave"
    assert retrieved.total_dwell_seconds == 600.0


def test_sqlite_filtering_by_date_and_identity(in_memory_repo):
    in_memory_repo.upsert_attendance_record(AttendanceRecord(
        identity="Eve", date="2026-08-17", first_check_in="2026-08-17T09:00:00+00:00"
    ))
    in_memory_repo.upsert_attendance_record(AttendanceRecord(
        identity="Eve", date="2026-08-18", first_check_in="2026-08-18T09:00:00+00:00"
    ))
    in_memory_repo.upsert_attendance_record(AttendanceRecord(
        identity="Frank", date="2026-08-17", first_check_in="2026-08-17T10:00:00+00:00"
    ))

    assert len(in_memory_repo.list_attendance_records(date_str="2026-08-17")) == 2
    assert len(in_memory_repo.list_attendance_records(identity="Eve")) == 2
    assert len(in_memory_repo.list_attendance_records(date_str="2026-08-17", identity="Eve")) == 1


def test_sqlite_thread_safety(in_memory_repo):
    errors = []

    def worker(worker_id):
        try:
            for i in range(10):
                rec = AttendanceRecord(
                    identity=f"User_{worker_id}",
                    date=f"2026-08-1{i}",
                    first_check_in="2026-08-17T09:00:00+00:00",
                    total_dwell_seconds=float(i * 10)
                )
                in_memory_repo.upsert_attendance_record(rec)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    records = in_memory_repo.list_attendance_records()
    assert len(records) == 50


def test_daily_summary_calculation(attendance_service):
    # Record Alice present
    attendance_service.repository.upsert_attendance_record(AttendanceRecord(
        identity="Alice", date="2026-08-17", first_check_in="2026-08-17T09:00:00+00:00",
        total_dwell_seconds=1800.0, status=AttendanceStatus.PRESENT
    ))
    # Record Bob partial
    attendance_service.repository.upsert_attendance_record(AttendanceRecord(
        identity="Bob", date="2026-08-17", first_check_in="2026-08-17T09:00:00+00:00",
        total_dwell_seconds=600.0, status=AttendanceStatus.PARTIAL
    ))

    summary = attendance_service.get_daily_summary(date_str="2026-08-17")
    assert summary.total_records == 2
    assert summary.total_present == 1
    assert summary.total_partial == 1
    assert summary.total_dwell_seconds == 2400.0
    assert summary.average_dwell_seconds == 1200.0


def test_export_csv_and_json(attendance_service):
    attendance_service.repository.upsert_attendance_record(AttendanceRecord(
        identity="Alice", date="2026-08-17", first_check_in="2026-08-17T09:00:00+00:00",
        total_dwell_seconds=1800.0, session_count=1, status=AttendanceStatus.PRESENT
    ))

    csv_data = attendance_service.export_csv(date_str="2026-08-17")
    assert "Alice" in csv_data
    assert "1800.00" in csv_data
    assert "PRESENT" in csv_data

    json_data = attendance_service.export_json(date_str="2026-08-17")
    parsed = json.loads(json_data)
    assert len(parsed) == 1
    assert parsed[0]["identity"] == "Alice"


@pytest.fixture
def app_client(attendance_service):
    app = create_app(config_path="config/config.yaml", attendance_service=attendance_service)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_get_attendance_records_endpoint(app_client, attendance_service):
    attendance_service.repository.upsert_attendance_record(AttendanceRecord(
        identity="Alice", date="2026-08-17", first_check_in="2026-08-17T09:00:00+00:00",
        total_dwell_seconds=1800.0, status=AttendanceStatus.PRESENT
    ))

    res = app_client.get("/api/v1/attendance/records?date=2026-08-17")
    assert res.status_code == 200
    data = res.get_json()
    assert data["count"] == 1
    assert data["records"][0]["identity"] == "Alice"


def test_api_get_attendance_summary_endpoint(app_client, attendance_service):
    attendance_service.repository.upsert_attendance_record(AttendanceRecord(
        identity="Alice", date="2026-08-17", first_check_in="2026-08-17T09:00:00+00:00",
        total_dwell_seconds=1800.0, status=AttendanceStatus.PRESENT
    ))

    res = app_client.get("/api/v1/attendance/summary?date=2026-08-17")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total_records"] == 1
    assert data["total_present"] == 1


def test_api_export_csv_and_json_endpoints(app_client, attendance_service):
    attendance_service.repository.upsert_attendance_record(AttendanceRecord(
        identity="Alice", date="2026-08-17", first_check_in="2026-08-17T09:00:00+00:00",
        total_dwell_seconds=1800.0, status=AttendanceStatus.PRESENT
    ))

    res_csv = app_client.get("/api/v1/attendance/export?date=2026-08-17&format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.content_type
    assert b"Alice" in res_csv.data

    res_json = app_client.get("/api/v1/attendance/export?date=2026-08-17&format=json")
    assert res_json.status_code == 200
    assert "application/json" in res_json.content_type
    assert b"Alice" in res_json.data


def test_api_malformed_date_returns_400(app_client):
    res = app_client.get("/api/v1/attendance/records?date=invalid-date")
    assert res.status_code == 400
    assert res.get_json()["error"] == "invalid_date_format"


def test_attendance_service_uninitialized_returns_503():
    app = create_app(config_path="config/config.yaml")
    app.attendance_service = None
    with app.test_client() as client:
        res = client.get("/api/v1/attendance/records")
        assert res.status_code == 503
        assert res.get_json()["error"] == "attendance_uninitialized"
