from flask import Blueprint, jsonify, current_app

from app.schemas.responses import serialize_presence_session, error_response

presence_bp = Blueprint("presence", __name__, url_prefix="/api/v1/presence")


@presence_bp.route("/active", methods=["GET"])
def get_active_sessions():
    """Returns list of all currently active (PRESENT or GRACE) presence sessions."""
    service = getattr(current_app, "runtime_service", None)
    if service is None:
        err, code = error_response("runtime_uninitialized", "Runtime service is not initialized", 503)
        return jsonify(err), code

    sessions = service.get_active_sessions()
    return jsonify({
        "active_sessions_count": len(sessions),
        "sessions": [serialize_presence_session(s) for s in sessions]
    }), 200


@presence_bp.route("/history", methods=["GET"])
def get_session_history():
    """Returns list of all closed / archived presence sessions."""
    service = getattr(current_app, "runtime_service", None)
    if service is None:
        err, code = error_response("runtime_uninitialized", "Runtime service is not initialized", 503)
        return jsonify(err), code

    history = service.get_session_history()
    return jsonify({
        "archived_sessions_count": len(history),
        "sessions": [serialize_presence_session(s) for s in history]
    }), 200


@presence_bp.route("/identity/<name>", methods=["GET"])
def get_identity_state(name: str):
    """Returns current presence state and any active session for a specific identity."""
    service = getattr(current_app, "runtime_service", None)
    if service is None:
        err, code = error_response("runtime_uninitialized", "Runtime service is not initialized", 503)
        return jsonify(err), code

    state = service.get_identity_state(name)
    active_session = None
    for s in service.get_active_sessions():
        if s.identity == name:
            active_session = serialize_presence_session(s)
            break

    return jsonify({
        "identity": name,
        "state": state.value if hasattr(state, "value") else str(state),
        "active_session": active_session
    }), 200
