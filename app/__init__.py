import os
from typing import Optional, Dict, Any
from flask import Flask
from config import load_config
from ml.pipeline import FaceRecognitionPipeline
from app.services.runtime_service import RuntimeService
from app.routes import health_bp, runtime_bp, presence_bp, legacy_bp


def create_app(
    config_path: str = "config/config.yaml",
    runtime_service: Optional[RuntimeService] = None
) -> Flask:
    """
    Application factory for Face Recognition Attendance System.
    Initializes Flask application, registers modern API blueprints,
    and mounts thread-safe RuntimeService.
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

    # 1. Initialize or inject RuntimeService
    if runtime_service is not None:
        app.runtime_service = runtime_service
    else:
        try:
            gallery_path = os.path.join(
                project_root,
                config.get("paths", {}).get("gallery_path", "data/embeddings/arcface_gallery.npz")
            )
            app.runtime_service = RuntimeService.from_config(
                config=config,
                gallery_path=gallery_path
            )
        except Exception:
            app.runtime_service = None

    # 2. Legacy baseline pipeline initialization (for backward compatibility)
    embeddings_path = os.path.join(
        project_root,
        config.get("paths", {}).get("embeddings_path", "trained_model/face_encodings.pkl")
    )
    try:
        app.pipeline = FaceRecognitionPipeline.from_config(config, embeddings_path=embeddings_path)
    except Exception:
        app.pipeline = None

    # 3. Register Blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(runtime_bp)
    app.register_blueprint(presence_bp)
    app.register_blueprint(legacy_bp)

    return app
