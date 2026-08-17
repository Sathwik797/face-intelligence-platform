import os
from typing import Optional, Dict, Any
from flask import Flask
from config import load_config
from ml.pipeline import FaceRecognitionPipeline
from app.services.runtime_service import RuntimeService
from app.services.attendance_service import AttendanceService
from app.repositories.sqlite_repository import SQLiteAttendanceRepository
from app.routes import health_bp, runtime_bp, presence_bp, attendance_bp, legacy_bp


def create_app(
    config_path: str = "config/config.yaml",
    runtime_service: Optional[RuntimeService] = None,
    attendance_service: Optional[AttendanceService] = None
) -> Flask:
    """
    Application factory for Face Recognition Attendance System.
    Initializes Flask application, registers modern API blueprints,
    and mounts thread-safe RuntimeService and AttendanceService.
    """
    if isinstance(config_path, str):
        config = load_config(config_path)
    else:
        config = config_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, "templates"),
        static_folder=os.path.join(project_root, "static")
    )
    app.config["SYSTEM_CONFIG"] = config
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB payload limit

    # 1. Initialize or inject AttendanceService
    if attendance_service is not None:
        app.attendance_service = attendance_service
    else:
        try:
            db_path = os.path.join(project_root, "data", "attendance.db")
            repo = SQLiteAttendanceRepository(db_path=db_path)
            app.attendance_service = AttendanceService(repository=repo)
        except Exception:
            app.attendance_service = None

    # 2. Initialize or inject RuntimeService
    if runtime_service is not None:
        app.runtime_service = runtime_service
        # Connect attendance service if not already connected
        if app.runtime_service.attendance_service is None and app.attendance_service is not None:
            app.runtime_service.attendance_service = app.attendance_service
    else:
        try:
            gallery_path = os.path.join(
                project_root,
                config.get("paths", {}).get("gallery_path", "data/embeddings/arcface_gallery.npz")
            )
            app.runtime_service = RuntimeService.from_config(
                config=config,
                gallery_path=gallery_path,
                attendance_service=app.attendance_service
            )
        except Exception:
            app.runtime_service = None

    # 3. Legacy baseline pipeline initialization (for backward compatibility)
    embeddings_path = os.path.join(
        project_root,
        config.get("paths", {}).get("embeddings_path", "trained_model/face_encodings.pkl")
    )
    try:
        app.pipeline = FaceRecognitionPipeline.from_config(config, embeddings_path=embeddings_path)
    except Exception:
        app.pipeline = None

    # 4. Register Blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(runtime_bp)
    app.register_blueprint(presence_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(legacy_bp)

    return app
