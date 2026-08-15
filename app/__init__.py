import os
from flask import Flask
from config import load_config
from ml.pipeline import FaceRecognitionPipeline

def create_app(config_path: str = "config/config.yaml") -> Flask:
    """Application factory for Face Recognition Attendance System."""
    config = load_config(config_path)

    # Initialize Flask app with template and static folder relative to root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, "templates"),
        static_folder=os.path.join(project_root, "static")
    )
    app.config["SYSTEM_CONFIG"] = config

    # Initialize ML pipeline
    embeddings_path = os.path.join(project_root, config.get("paths", {}).get("embeddings_path", "trained_model/face_encodings.pkl"))
    app.pipeline = FaceRecognitionPipeline.from_config(config, embeddings_path=embeddings_path)

    # Register blueprints/routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
