import pytest
from app import create_app
from app.services.runtime_service import RuntimeService
from ml.runtime import (
    FaceIntelligenceRuntime,
    RuntimeConfig,
    StageLatencyMetrics,
    RuntimeFrameResult
)
from ml.pipeline import ModernRecognitionResult
from ml.temporal.schemas import TemporalPolicyConfig
from ml.temporal.stabilizer import TemporalIdentityStabilizer
from ml.presence.schemas import PresenceMode
from ml.presence.state_machine import PRESENCE_PRESETS
from ml.presence.manager import PresenceManager


class MockRecognitionPipeline:
    def __init__(self, identity="Alice"):
        self.identity = identity
        self.threshold = 0.24

    def recognize(self, rgb_image):
        return ModernRecognitionResult(
            identity=self.identity,
            best_candidate=self.identity,
            similarity=0.85,
            threshold=self.threshold,
            recognized=True,
            bbox=(10, 100, 100, 10),
            latency_ms=5.0,
            reason="accepted"
        )


@pytest.fixture
def client():
    rec_pipe = MockRecognitionPipeline(identity="Alice")
    temp_stab = TemporalIdentityStabilizer(policy=TemporalPolicyConfig(
        window_size=4,
        min_observations=2,
        min_stable_ratio=0.65,
        mode="fast"
    ))
    pres_mgr = PresenceManager(config=PRESENCE_PRESETS[PresenceMode.FAST])
    runtime = FaceIntelligenceRuntime(
        recognition_pipeline=rec_pipe,
        temporal_stabilizer=temp_stab,
        presence_manager=pres_mgr,
        config=RuntimeConfig(auto_tick=True, heartbeat_interval_seconds=1.0)
    )
    service = RuntimeService(runtime=runtime)

    app = create_app(
        config_path="config/config.yaml",
        runtime_service=service
    )
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_dashboard_index_route(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Face Intelligence Platform" in html
    assert "E2 ArcFace + Presence" in html
    assert "camera-video" in html
    assert "camera-overlay" in html


def test_dashboard_static_css_assets_served(client):
    css_files = [
        "/static/css/dashboard.css",
        "/static/css/variables.css",
        "/static/css/layout.css",
        "/static/css/components.css"
    ]
    for path in css_files:
        res = client.get(path)
        assert res.status_code == 200, f"Failed to serve {path}"


def test_dashboard_static_js_assets_served(client):
    js_files = [
        "/static/js/api.js",
        "/static/js/camera.js",
        "/static/js/overlay.js",
        "/static/js/state.js",
        "/static/js/app.js"
    ]
    for path in js_files:
        res = client.get(path)
        assert res.status_code == 200, f"Failed to serve {path}"


def test_dashboard_template_contains_all_script_tags(client):
    res = client.get("/")
    html = res.data.decode("utf-8")
    assert "js/api.js" in html
    assert "js/camera.js" in html
    assert "js/overlay.js" in html
    assert "js/state.js" in html
    assert "js/app.js" in html


def test_dashboard_api_contracts_compatibility(client):
    # Verify health response
    health_res = client.get("/api/v1/health")
    assert health_res.status_code == 200
    h_data = health_res.get_json()
    assert "status" in h_data
    assert "runtime_status" in h_data

    # Verify status response
    status_res = client.get("/api/v1/runtime/status")
    assert status_res.status_code == 200
    s_data = status_res.get_json()
    assert "frame_counter" in s_data
    assert "active_sessions_count" in s_data


def test_dashboard_legacy_endpoint_compatibility(client):
    res = client.post("/recognize", json={"image": ""})
    assert res.status_code == 400
    assert res.headers.get("X-API-Deprecated") is None or res.headers.get("X-API-Deprecated") == "true"
