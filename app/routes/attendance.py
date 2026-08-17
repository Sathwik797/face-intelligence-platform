from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, Response, current_app

from app.schemas.responses import error_response

attendance_bp = Blueprint("attendance", __name__, url_prefix="/api/v1/attendance")


@attendance_bp.route("/records", methods=["GET"])
def list_records():
    """
    Returns list of daily attendance records filtered by optional date and identity.
    Query parameters:
      - date: Optional YYYY-MM-DD date string (defaults to all or specific date).
      - identity: Optional person name.
    """
    service = getattr(current_app, "attendance_service", None)
    if service is None:
        err, code = error_response("attendance_uninitialized", "Attendance service is not initialized", 503)
        return jsonify(err), code

    date_str = request.args.get("date")
    identity = request.args.get("identity")

    # Validate date format if provided
    if date_str:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            err, code = error_response("invalid_date_format", "Date parameter must be YYYY-MM-DD", 400)
            return jsonify(err), code

    records = service.list_records(date_str=date_str, identity=identity)
    return jsonify({
        "count": len(records),
        "records": [r.to_dict() for r in records]
    }), 200


@attendance_bp.route("/summary", methods=["GET"])
def get_summary():
    """
    Returns aggregated statistical summary for a target calendar date.
    Query parameter:
      - date: Optional YYYY-MM-DD (defaults to current date UTC).
    """
    service = getattr(current_app, "attendance_service", None)
    if service is None:
        err, code = error_response("attendance_uninitialized", "Attendance service is not initialized", 503)
        return jsonify(err), code

    date_str = request.args.get("date")
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            err, code = error_response("invalid_date_format", "Date parameter must be YYYY-MM-DD", 400)
            return jsonify(err), code

    summary = service.get_daily_summary(date_str=date_str)
    return jsonify(summary.to_dict()), 200


@attendance_bp.route("/export", methods=["GET"])
def export():
    """
    Exports attendance records as downloadable CSV or JSON.
    Query parameters:
      - date: Optional YYYY-MM-DD.
      - format: 'csv' (default) or 'json'.
    """
    service = getattr(current_app, "attendance_service", None)
    if service is None:
        err, code = error_response("attendance_uninitialized", "Attendance service is not initialized", 503)
        return jsonify(err), code

    date_str = request.args.get("date")
    if date_str:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            err, code = error_response("invalid_date_format", "Date parameter must be YYYY-MM-DD", 400)
            return jsonify(err), code

    fmt = request.args.get("format", "csv").lower()
    filename_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if fmt == "json":
        data = service.export_json(date_str=date_str)
        return Response(
            data,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=attendance_{filename_date}.json"}
        )
    elif fmt == "csv":
        data = service.export_csv(date_str=date_str)
        return Response(
            data,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=attendance_{filename_date}.csv"}
        )
    else:
        err, code = error_response("invalid_format", "Export format must be 'csv' or 'json'", 400)
        return jsonify(err), code
