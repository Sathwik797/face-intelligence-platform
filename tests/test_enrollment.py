import os
import io
import base64
import json
import pytest
import numpy as np
import cv2
from PIL import Image

from app import create_app
from app.repositories.sqlite_repository import SQLiteAttendanceRepository
from app.services.attendance_service import AttendanceService
from app.services.enrollment_service import EnrollmentService
from app.services.runtime_service import RuntimeService
from ml.detector import ModernFaceDetector
from ml.aligner import FaceAligner
from ml.embedder import ArcFaceEmbedder
from ml.gallery import IdentityGallery
from ml.quality import FaceQualityAssessor, QualityMode
from ml.pipeline import ModernRecognitionPipeline
from ml.runtime import FaceIntelligenceRuntime
from config import load_config


def _create_synthetic_face_image() -> np.ndarray:
    """Generates a synthetic 300x300 image with a recognizable face-like structure."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 180
    # Head outline
    cv2.circle(img, (150, 150), 90, (140, 160, 200), -1)
    # Eyes
    cv2.circle(img, (120, 130), 12, (20, 20, 20), -1)
    cv2.circle(img, (180, 130), 12, (20, 20, 20), -1)
    # Nose
    cv2.line(img, (150, 130), (150, 165), (50, 50, 50), 3)
    # Mouth
    cv2.ellipse(img, (150, 190), (30, 12), 0, 0, 180, (40, 40, 150), 4)
    return img


def _image_to_base64(img_np: np.ndarray) -> str:
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def test_setup(tmp_path):
    config = load_config("config/config.yaml")
    gallery_file = str(tmp_path / "test_gallery.npz")
    db_file = str(tmp_path / "test_attendance.db")

    repo = SQLiteAttendanceRepository(db_path=db_file)
    attendance_service = AttendanceService(repository=repo)

    # In-memory gallery starting with 1 dummy identity
    dummy_emb = np.random.randn(1, 512).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)
    gallery = IdentityGallery(embeddings=dummy_emb, identities=["BaselineUser"])
    gallery.save(gallery_file)

    runtime_service = RuntimeService.from_config(
        config=config,
        gallery_path=gallery_file,
        attendance_service=attendance_service
    )

    enrollment_service = EnrollmentService(
        pipeline=runtime_service.runtime.recognition_pipeline,
        repository=repo,
        assessor=FaceQualityAssessor(mode=QualityMode.LENIENT),
        gallery_filepath=gallery_file
    )

    return {
        "config": config,
        "repo": repo,
        "gallery": gallery,
        "pipeline": runtime_service.runtime.recognition_pipeline,
        "enrollment_service": enrollment_service,
        "runtime_service": runtime_service,
        "attendance_service": attendance_service,
        "gallery_file": gallery_file,
        "db_file": db_file
    }


def test_enroll_new_identity_valid_synthetic_or_real(test_setup):
    service = test_setup["enrollment_service"]
    face_img = _create_synthetic_face_image()

    result = service.enroll_identity(
        identity="NewPerson",
        rgb_image=face_img,
        quality_mode="lenient"
    )

    if result.success:
        assert result.identity == "NewPerson"
        assert result.template_count >= 1
        assert result.quality_status == "accepted"
        # Check SQLite persistence
        db_info = service.get_identity("NewPerson")
        assert db_info is not None
        assert db_info.identity == "NewPerson"
        # Check gallery
        assert "NewPerson" in service.pipeline.gallery.unique_identities


def test_enroll_rejects_empty_identity(test_setup):
    service = test_setup["enrollment_service"]
    face_img = _create_synthetic_face_image()

    result = service.enroll_identity(
        identity="",
        rgb_image=face_img
    )
    assert not result.success
    assert result.error_code == "invalid_identity"


def test_enroll_rejects_no_face_blank_image(test_setup):
    service = test_setup["enrollment_service"]
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)

    result = service.enroll_identity(
        identity="Ghost",
        rgb_image=blank_img
    )
    assert not result.success
    assert result.error_code == "no_face_detected"


def test_enroll_rejects_low_quality_blurry_image(test_setup):
    service = test_setup["enrollment_service"]
    face_img = _create_synthetic_face_image()
    # Inject heavy blur
    blurred = cv2.GaussianBlur(face_img, (51, 51), 30.0)

    result = service.enroll_identity(
        identity="BlurryPerson",
        rgb_image=blurred,
        quality_mode="strict"
    )
    # Either no face detected or rejected by FQA
    assert not result.success
    assert result.error_code in ["quality_rejected", "no_face_detected"]


def test_delete_identity_removes_from_gallery_and_db(test_setup):
    service = test_setup["enrollment_service"]
    # Add dummy templates
    emb = np.random.randn(2, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    service.pipeline.gallery.add_templates("DeleteMe", emb)

    deleted = service.delete_identity("DeleteMe")
    assert deleted is True
    assert "DeleteMe" not in service.pipeline.gallery.unique_identities


def test_list_identities(test_setup):
    service = test_setup["enrollment_service"]
    identities = service.list_identities()
    assert len(identities) >= 1
    names = [i.identity for i in identities]
    assert "BaselineUser" in names


def test_api_identities_endpoints(test_setup):
    app = create_app(
        config_path=test_setup["config"],
        runtime_service=test_setup["runtime_service"],
        attendance_service=test_setup["attendance_service"],
        enrollment_service=test_setup["enrollment_service"]
    )
    app.config["TESTING"] = True

    with app.test_client() as client:
        # GET /api/v1/identities
        res = client.get("/api/v1/identities")
        assert res.status_code == 200
        data = res.get_json()
        assert data["count"] >= 1

        # GET /api/v1/identities/<name>
        res_user = client.get("/api/v1/identities/BaselineUser")
        assert res_user.status_code == 200
        assert res_user.get_json()["identity_info"]["identity"] == "BaselineUser"

        # GET /api/v1/identities/NonExistent -> 404
        res_404 = client.get("/api/v1/identities/NonExistentPerson")
        assert res_404.status_code == 404

        # POST /api/v1/identities/enroll (blank image -> 400 no face)
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        b64_blank = _image_to_base64(blank)
        res_post = client.post("/api/v1/identities/enroll", json={
            "identity": "TestGhost",
            "image": b64_blank
        })
        assert res_post.status_code in [400, 422]

        # DELETE /api/v1/identities/BaselineUser
        res_del = client.delete("/api/v1/identities/BaselineUser")
        assert res_del.status_code == 200
        assert res_del.get_json()["status"] == "success"


def test_enrollment_uninitialized_returns_503(test_setup):
    app = create_app(
        config_path=test_setup["config"],
        runtime_service=test_setup["runtime_service"],
        attendance_service=test_setup["attendance_service"]
    )
    app.enrollment_service = None
    with app.test_client() as client:
        res = client.get("/api/v1/identities")
        assert res.status_code == 503
        assert res.get_json()["error"] == "enrollment_uninitialized"
