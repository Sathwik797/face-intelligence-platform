from app.routes.health import health_bp
from app.routes.runtime import runtime_bp
from app.routes.presence import presence_bp
from app.routes.attendance import attendance_bp
from app.routes.identities import identities_bp
from app.routes.legacy import legacy_bp

__all__ = [
    "health_bp",
    "runtime_bp",
    "presence_bp",
    "attendance_bp",
    "identities_bp",
    "legacy_bp"
]
