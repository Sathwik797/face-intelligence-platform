import io
import base64
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Blueprint, request, jsonify, current_app

from app.schemas.responses import serialize_frame_result, error_response

runtime_bp = Blueprint("runtime", __name__, url_prefix="/api/v1/runtime")

# Max request payload limit (10MB)
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@runtime_bp.route("/status", methods=["GET"])
def status():
    """Returns current status and performance metrics of FaceIntelligenceRuntime."""
    service = getattr(current_app, "runtime_service", None)
    if service is None:
        err, code = error_response("runtime_uninitialized", "Runtime service is not initialized", 503)
        return jsonify(err), code

    return jsonify(service.get_status()), 200


@runtime_bp.route("/start", methods=["POST"])
def start():
    """Starts the FaceIntelligenceRuntime orchestrator."""
    service = getattr(current_app, "runtime_service", None)
    if service is None:
        err, code = error_response("runtime_uninitialized", "Runtime service is not initialized", 503)
        return jsonify(err), code

    service.start()
    return jsonify({"status": "RUNNING", "message": "Runtime started successfully"}), 200


@runtime_bp.route("/stop", methods=["POST"])
def stop():
    """Gracefully stops runtime and finalizes active sessions with shutdown reason."""
    service = getattr(current_app, "runtime_service", None)
    if service is None:
        err, code = error_response("runtime_uninitialized", "Runtime service is not initialized", 503)
        return jsonify(err), code

    data = request.get_json(silent=True) or {}
    reason = data.get("reason", "runtime_shutdown")
    shutdown_events = service.stop(reason=reason)
    return jsonify({
        "status": "STOPPED",
        "message": f"Runtime stopped ({reason})",
        "finalized_sessions_count": len(shutdown_events)
    }), 200


@runtime_bp.route("/reset", methods=["POST"])
def reset():
    """Resets temporal history, presence state machines, and frame counters."""
    service = getattr(current_app, "runtime_service", None)
    if service is None:
        err, code = error_response("runtime_uninitialized", "Runtime service is not initialized", 503)
        return jsonify(err), code

    service.reset()
    return jsonify({"status": "STOPPED", "message": "Runtime state reset successfully"}), 200


@runtime_bp.route("/process-frame", methods=["POST"])
def process_frame():
    """
    Ingests base64 image frame, runs end-to-end FaceIntelligenceRuntime,
    and returns serialized RuntimeFrameResult.
    """
    service = getattr(current_app, "runtime_service", None)
    if service is None:
        err, code = error_response("runtime_uninitialized", "Runtime service is not initialized", 503)
        return jsonify(err), code

    # 1. Enforce payload size limit
    if request.content_length and request.content_length > MAX_IMAGE_BYTES:
        err, code = error_response("payload_too_large", "Request payload exceeds 10MB limit", 413)
        return jsonify(err), code

    data = request.get_json(silent=True)
    if data is None:
        err, code = error_response("invalid_json", "Malformed or missing JSON body", 400)
        return jsonify(err), code

    if "image" not in data:
        err, code = error_response("missing_image", "Missing 'image' field in request body", 400)
        return jsonify(err), code

    image_data = data.get("image")
    if not isinstance(image_data, str) or not image_data.strip():
        err, code = error_response("empty_image", "Image field cannot be empty", 400)
        return jsonify(err), code

    # 2. Decode Base64 safely in memory
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data, validate=True)
        if len(image_bytes) == 0:
            err, code = error_response("empty_image", "Decoded image bytes are empty", 422)
            return jsonify(err), code

        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        rgb_frame = np.array(pil_image)
    except Exception as e:
        err, code = error_response("unprocessable_image", f"Failed to decode base64 image: {str(e)}", 422)
        return jsonify(err), code

    # 3. Optional timestamp parsing
    ts = None
    if "timestamp" in data and data["timestamp"]:
        try:
            ts = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        except Exception:
            ts = None

    # 4. Execute runtime processing
    try:
        result = service.process_frame(rgb_frame, timestamp=ts)
        return jsonify(serialize_frame_result(result)), 200
    except Exception as e:
        err, code = error_response("internal_error", "An internal error occurred during frame processing", 500)
        return jsonify(err), code
