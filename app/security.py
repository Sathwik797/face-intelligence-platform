import hmac
import os
from functools import wraps
from typing import Callable

from flask import current_app, request, jsonify


def require_api_key(view: Callable):
    """
    Protects state-changing and privacy-sensitive API endpoints.

    Local test clients remain usable with Flask TESTING enabled. In production,
    FACE_INTELLIGENCE_API_KEY must be configured and supplied as:
        X-API-Key: <key>
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_app.config.get("TESTING"):
            return view(*args, **kwargs)

        expected = os.getenv("FACE_INTELLIGENCE_API_KEY", "").strip()
        if not expected:
            return jsonify({
                "status": "error",
                "error": "api_auth_not_configured",
                "message": "API authentication is not configured."
            }), 503

        supplied = request.headers.get("X-API-Key", "")
        if not supplied or not hmac.compare_digest(supplied, expected):
            return jsonify({
                "status": "error",
                "error": "unauthorized",
                "message": "A valid API key is required."
            }), 401

        return view(*args, **kwargs)

    return wrapped
