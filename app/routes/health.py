from datetime import datetime, timezone
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__, url_prefix="/api/v1")


@health_bp.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint returning application status, runtime status, and readiness.
    """
    runtime_service = getattr(current_app, "runtime_service", None)
    runtime_status = runtime_service.status.value if runtime_service is not None else "UNINITIALIZED"

    # Check if models are loaded in runtime
    models_ready = False
    if runtime_service is not None and hasattr(runtime_service.runtime, "recognition_pipeline"):
        pipe = runtime_service.runtime.recognition_pipeline
        models_ready = (hasattr(pipe, "detector") and pipe.detector is not None)

    return jsonify({
        "status": "healthy",
        "runtime_status": runtime_status,
        "models_ready": models_ready,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200
