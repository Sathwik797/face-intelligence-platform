import os
import pytest
from app import create_app
from config import load_config
from app.repositories.sqlite_repository import SQLiteAttendanceRepository
from app.services.attendance_service import AttendanceService
from app.services.enrollment_service import EnrollmentService
from app.services.runtime_service import RuntimeService
from ml.gallery import IdentityGallery
import numpy as np


@pytest.fixture
def client(tmp_path):
    config = load_config("config/config.yaml")
    gallery_file = str(tmp_path / "test_gallery.npz")
    db_file = str(tmp_path / "test_attendance.db")

    repo = SQLiteAttendanceRepository(db_path=db_file)
    attendance_service = AttendanceService(repository=repo)

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
        gallery_filepath=gallery_file
    )

    app = create_app(
        config_path=config,
        runtime_service=runtime_service,
        attendance_service=attendance_service,
        enrollment_service=enrollment_service
    )
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_dashboard_v2_index_route_contains_all_tabs_and_views(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Brand and backward compatibility
    assert "Face Intelligence Platform" in html
    assert "Face Recognition" in html
    assert "E2 ArcFace + Presence" in html

    # Navigation Tabs
    assert "tab-btn-live" in html
    assert "tab-btn-attendance" in html
    assert "tab-btn-identities" in html

    # Tab Views
    assert "view-live" in html
    assert "view-attendance" in html
    assert "view-identities" in html

    # Enrollment Modal
    assert "modal-enroll" in html
    assert "enroll-name" in html
    assert "enroll-btn-snap" in html


def test_dashboard_v2_serves_all_javascript_modules(client):
    js_modules = [
        "/static/js/api.js",
        "/static/js/camera.js",
        "/static/js/overlay.js",
        "/static/js/state.js",
        "/static/js/attendance_view.js",
        "/static/js/identities_view.js",
        "/static/js/app.js"
    ]
    for path in js_modules:
        res = client.get(path)
        assert res.status_code == 200, f"Failed to serve {path}"


def test_dashboard_v2_serves_all_css_assets(client):
    css_files = [
        "/static/css/dashboard.css",
        "/static/css/variables.css",
        "/static/css/layout.css",
        "/static/css/components.css"
    ]
    for path in css_files:
        res = client.get(path)
        assert res.status_code == 200, f"Failed to serve {path}"


def test_dashboard_v2_attendance_controls_present(client):
    res = client.get("/")
    html = res.data.decode("utf-8")
    assert "att-date-input" in html
    assert "att-btn-export-csv" in html
    assert "att-btn-export-json" in html
    assert "att-table-body" in html


def test_dashboard_v2_identities_controls_present(client):
    res = client.get("/")
    html = res.data.decode("utf-8")
    assert "id-stat-total" in html
    assert "id-btn-open-enroll" in html
    assert "id-table-body" in html
