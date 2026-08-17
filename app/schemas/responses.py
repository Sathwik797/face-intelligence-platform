from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from ml.runtime.schemas import RuntimeFrameResult, RuntimeStatus
from ml.presence.schemas import PresenceEvent, PresenceSession, PresenceState


def serialize_frame_result(result: RuntimeFrameResult) -> Dict[str, Any]:
    """
    Serializes a RuntimeFrameResult into a safe, sanitized JSON-serializable dictionary.
    Excludes raw deep embeddings and internal filesystem paths.
    """
    rec = result.recognition
    temp = result.temporal

    # 1. Sanitized Recognition Payload
    rec_dict = None
    if rec is not None:
        rec_dict = {
            "identity": rec.identity,
            "best_candidate": rec.best_candidate,
            "similarity": round(float(rec.similarity), 4) if rec.similarity is not None else -1.0,
            "threshold": round(float(rec.threshold), 4),
            "recognized": bool(rec.recognized),
            "bbox": list(rec.bbox) if rec.bbox is not None else None,
            "reason": rec.reason,
            "quality_status": getattr(rec.quality, "quality_status", "none") if rec.quality is not None else "none",
            "quality_score": round(float(rec.quality.overall_quality), 4) if (rec.quality is not None and getattr(rec.quality, "overall_quality", None) is not None) else None
        }

    # 2. Sanitized Temporal Payload
    temp_dict = None
    if temp is not None:
        temp_dict = {
            "stable_identity": temp.stable_identity,
            "state": temp.state.value if hasattr(temp.state, "value") else str(temp.state),
            "is_stable": bool(temp.is_stable),
            "confidence_score": round(float(temp.confidence_score), 4),
            "observations_count": int(temp.observations_count),
            "consecutive_stable_count": int(temp.consecutive_stable_count),
            "active_candidate": temp.active_candidate
        }

    # 3. Sanitized Presence Events
    presence_events = [serialize_presence_event(e) for e in result.presence_events]

    # 4. Sanitized Active Sessions
    active_sessions = [serialize_presence_session(s) for s in result.active_sessions]

    return {
        "frame_index": int(result.frame_index),
        "timestamp": result.timestamp.isoformat(),
        "recognition": rec_dict,
        "temporal": temp_dict,
        "presence_events": presence_events,
        "active_sessions": active_sessions,
        "latencies": result.latencies.to_dict() if result.latencies is not None else {},
        "error": result.error
    }


def serialize_presence_event(event: PresenceEvent) -> Dict[str, Any]:
    """Serializes a PresenceEvent into a sanitized dictionary."""
    return {
        "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
        "identity": event.identity,
        "timestamp": event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else str(event.timestamp),
        "previous_state": event.previous_state.value if hasattr(event.previous_state, "value") else str(event.previous_state),
        "new_state": event.new_state.value if hasattr(event.new_state, "value") else str(event.new_state),
        "session_id": event.session_id,
        "reason": event.reason
    }


def serialize_presence_session(session: PresenceSession) -> Dict[str, Any]:
    """Serializes a PresenceSession into a sanitized dictionary."""
    return {
        "session_id": session.session_id,
        "identity": session.identity,
        "started_at": session.started_at.isoformat() if isinstance(session.started_at, datetime) else str(session.started_at),
        "last_seen_at": session.last_seen_at.isoformat() if isinstance(session.last_seen_at, datetime) else str(session.last_seen_at),
        "ended_at": session.ended_at.isoformat() if session.ended_at is not None else None,
        "state": session.state.value if hasattr(session.state, "value") else str(session.state),
        "is_active": bool(session.is_active),
        "duration_seconds": round(float(session.duration_seconds), 2),
        "observation_count": int(session.observation_count),
        "interruption_count": int(session.interruption_count)
    }


def error_response(code: str, message: str, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
    """Generates standardized error response dictionary and HTTP status code."""
    return {
        "status": "error",
        "error": code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }, status_code
