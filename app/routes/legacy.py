import os
import base64
import datetime
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, current_app

legacy_bp = Blueprint("legacy", __name__)


def log_attendance(name: str, attendance_file: str):
    """Logs attendance for recognized identity into CSV (legacy behavior)."""
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(attendance_file, index=False)

    df = pd.read_csv(attendance_file)
    new_entry = pd.DataFrame([{"Name": name, "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)


@legacy_bp.route("/")
def index():
    """Serves the main webcam recognition interface."""
    return render_template("index.html")


@legacy_bp.route("/recognize", methods=["POST"])
def recognize():
    """
    Legacy frame-level recognition route (Phase 1 Baseline Compatibility).
    Receives base64 webcam frame and logs to attendance CSV.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data received"}), 400

    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "Empty image payload"}), 400

    try:
        # Decode base64 image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        rgb_image = np.array(pil_image)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    # Execute legacy pipeline if present
    pipeline = getattr(current_app, "pipeline", None)
    if pipeline is not None:
        results = pipeline.process_image(rgb_image)
    else:
        results = []

    config = current_app.config.get("SYSTEM_CONFIG", {})
    attendance_file = config.get("paths", {}).get("attendance_file", "attendance.csv")

    recognized_names = []
    for res in results:
        recognized_names.append(res.identity)
        if res.recognized and res.identity != "Unknown":
            log_attendance(res.identity, attendance_file)

    resp = jsonify({
        "recognized": recognized_names,
        "details": [r.to_dict() for r in results]
    })
    resp.headers["X-API-Deprecated"] = "true"
    return resp, 200
