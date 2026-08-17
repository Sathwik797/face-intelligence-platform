import io
import base64
from flask import Blueprint, request, jsonify, current_app
from PIL import Image
import numpy as np

from app.schemas.responses import error_response

identities_bp = Blueprint("identities", __name__, url_prefix="/api/v1/identities")


def _decode_image_payload(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_img)


@identities_bp.route("", methods=["GET"])
def list_identities():
    """
    Returns list of all enrolled identities with template counts.
    """
    service = getattr(current_app, "enrollment_service", None)
    if service is None:
        err, code = error_response("enrollment_uninitialized", "Enrollment service is not initialized", 503)
        return jsonify(err), code

    identities = service.list_identities()
    return jsonify({
        "count": len(identities),
        "identities": [i.to_dict() for i in identities]
    }), 200


@identities_bp.route("/<name>", methods=["GET"])
def get_identity_details(name: str):
    """
    Returns metadata and attendance history for a specific identity.
    """
    service = getattr(current_app, "enrollment_service", None)
    if service is None:
        err, code = error_response("enrollment_uninitialized", "Enrollment service is not initialized", 503)
        return jsonify(err), code

    identity_info = service.get_identity(name)
    if identity_info is None:
        err, code = error_response("identity_not_found", f"Identity '{name}' not found", 404)
        return jsonify(err), code

    # Optional attendance info
    attendance_service = getattr(current_app, "attendance_service", None)
    attendance_records = []
    if attendance_service is not None:
        attendance_records = [r.to_dict() for r in attendance_service.list_records(identity=name)]

    return jsonify({
        "identity_info": identity_info.to_dict(),
        "attendance_records_count": len(attendance_records),
        "attendance_records": attendance_records
    }), 200


@identities_bp.route("/enroll", methods=["POST"])
def enroll_identity():
    """
    Enrolls a new identity template after executing Face Quality Assessment (FQA).
    Request JSON:
      - identity (str, required)
      - image (str, base64 data URL, required)
      - quality_mode (str, optional, default "balanced")
      - notes (str, optional)
    """
    service = getattr(current_app, "enrollment_service", None)
    if service is None:
        err, code = error_response("enrollment_uninitialized", "Enrollment service is not initialized", 503)
        return jsonify(err), code

    data = request.get_json(silent=True)
    if not data or "identity" not in data or "image" not in data:
        err, code = error_response("invalid_payload", "Request body must contain 'identity' and 'image'", 400)
        return jsonify(err), code

    identity = data.get("identity", "").strip()
    image_b64 = data.get("image", "")
    quality_mode = data.get("quality_mode", "balanced")
    notes = data.get("notes")

    if not identity:
        err, code = error_response("invalid_identity", "Identity name cannot be empty", 400)
        return jsonify(err), code

    try:
        rgb_image = _decode_image_payload(image_b64)
    except Exception as e:
        err, code = error_response("invalid_image_encoding", f"Failed to decode base64 image: {str(e)}", 400)
        return jsonify(err), code

    result = service.enroll_identity(
        identity=identity,
        rgb_image=rgb_image,
        quality_mode=quality_mode,
        notes=notes
    )

    if not result.success:
        status_code = 422 if result.error_code == "quality_rejected" else 400
        return jsonify(result.to_dict()), status_code

    return jsonify(result.to_dict()), 200


@identities_bp.route("/<name>", methods=["DELETE"])
def delete_identity(name: str):
    """
    Deletes an identity from the live gallery and SQLite metadata.
    """
    service = getattr(current_app, "enrollment_service", None)
    if service is None:
        err, code = error_response("enrollment_uninitialized", "Enrollment service is not initialized", 503)
        return jsonify(err), code

    success = service.delete_identity(name)
    if not success:
        err, code = error_response("identity_not_found", f"Identity '{name}' could not be deleted or was not found", 404)
        return jsonify(err), code

    return jsonify({
        "status": "success",
        "message": f"Identity '{name}' successfully deleted"
    }), 200
