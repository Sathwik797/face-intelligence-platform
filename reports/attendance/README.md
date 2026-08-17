# Phase 14 Attendance Business Engine & Persistence Repository

## Overview
This directory documents the **Attendance Business Engine & Persistence Repository** established in Phase 14. It connects continuous in-memory presence tracking (`PresenceEvent`, `PresenceSession`) with persistent, idempotent daily attendance records, session audit logging, and reporting APIs.

---

## 1. Architecture & Data Flow

$$\text{FaceIntelligenceRuntime} \xrightarrow{\text{PresenceEvent / PresenceSession}} \text{AttendanceService} \xrightarrow{\text{AttendanceRulePolicy}} \text{SQLiteAttendanceRepository} \xrightarrow{\text{REST APIs}} \text{Attendance Journal \& CSV Export}$$

```
FaceIntelligenceRuntime (ml/runtime/)
                 │
                 ▼ (ENTRY_CONFIRMED, PRESENCE_UPDATED, SESSION_ENDED)
┌─────────────────────────────────────────────────────────────┐
│ 1. AttendanceService (app/services/attendance_service.py)   │
│    - Consumes PresenceEvents and PresenceSessions           │
│    - Enforces Idempotency (preventing duplicate check-ins)  │
│    - Aggregates Cumulative Dwell Duration per Person/Day    │
│    - Evaluates Attendance Status: PRESENT, PARTIAL          │
└─────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SQLiteAttendanceRepository (app/repositories/)           │
│    - Thread-safe SQLite persistence with atomic transactions│
│    - Tables: attendance_records, session_audit_log          │
│    - Zero external database requirements (Standard Library) │
└─────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. REST API Endpoints (/api/v1/attendance/)                 │
│    - GET /api/v1/attendance/records                         │
│    - GET /api/v1/attendance/summary                         │
│    - GET /api/v1/attendance/export?format=csv|json          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema (SQLite)

### Table: `attendance_records`
| Column | Type | Constraints | Description |
|---|---|---|---|
| `record_id` | `TEXT` | `PRIMARY KEY` | Unique record identifier (UUID) |
| `identity` | `TEXT` | `NOT NULL` | Enrolled person's name |
| `date` | `TEXT` | `NOT NULL` | Calendar date (`YYYY-MM-DD`) |
| `first_check_in` | `TEXT` | `NOT NULL` | ISO-8601 timestamp of first confirmed entry |
| `last_check_out` | `TEXT` | `NULLABLE` | ISO-8601 timestamp of latest checkout / grace expiry |
| `total_dwell_seconds`| `REAL` | `NOT NULL DEFAULT 0.0` | Cumulative presence dwell duration in seconds |
| `session_count` | `INTEGER` | `NOT NULL DEFAULT 0` | Total completed presence sessions on date |
| `status` | `TEXT` | `NOT NULL` | `PRESENT`, `IN_PROGRESS`, or `PARTIAL` |
| `last_confidence_score` | `REAL` | `NOT NULL DEFAULT 1.0`| Temporal consensus confidence score |
| `created_at` | `TEXT` | `NOT NULL` | UTC creation timestamp |
| `updated_at` | `TEXT` | `NOT NULL` | UTC update timestamp |
| **Constraint** | | `UNIQUE(identity, date)` | Ensures exactly 1 record per person per calendar day |

### Table: `session_audit_log`
| Column | Type | Constraints | Description |
|---|---|---|---|
| `session_id` | `TEXT` | `PRIMARY KEY` | Presence session UUID |
| `identity` | `TEXT` | `NOT NULL` | Enrolled person's name |
| `date` | `TEXT` | `NOT NULL` | Calendar date (`YYYY-MM-DD`) |
| `started_at` | `TEXT` | `NOT NULL` | Session start ISO-8601 timestamp |
| `ended_at` | `TEXT` | `NOT NULL` | Session end ISO-8601 timestamp |
| `duration_seconds` | `REAL` | `NOT NULL` | Duration of individual continuous session |
| `observation_count`| `INTEGER` | `NOT NULL` | Total observations accumulated in session |
| `interruption_count`| `INTEGER`| `NOT NULL` | Count of grace recovery interruptions |
| `closure_reason` | `TEXT` | `NOT NULL` | Reason for session termination (`absence_timeout`, `runtime_shutdown`) |

---

## 3. Attendance Business Rules & Idempotency
1. **Daily Idempotency**: A person entering multiple times on the same date updates their existing daily record (`session_count += 1`, `total_dwell_seconds += session_duration`, `last_check_out = session.ended_at`) without duplicating rows.
2. **First Check-In Preservation**: `first_check_in` is immutably anchored to the first confirmed entry of the day.
3. **Session Audit Tracking**: Every completed continuous session is recorded in `session_audit_log` for full compliance auditing.
4. **Configurable Status Policy**:
   - Status defaults to `PRESENT` upon confirmed entry.
   - If `min_present_seconds > 0.0` is configured in `AttendanceConfig`, sessions under that threshold are designated `PARTIAL` until cumulative dwell requirement is satisfied.
