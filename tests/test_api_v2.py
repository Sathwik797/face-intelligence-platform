import os
import io
import base64
import pytest
from datetime import datetime, timezone
from PIL import Image
import numpy as np

from app import create_app
from app.services.runtime_service import RuntimeService
from ml.runtime import (
    FaceIntelligenceRuntime,
    RuntimeStatus,
    RuntimeConfig,
    StageLatencyMetrics,
    RuntimeFrameResult
)
from ml.pipeline import ModernRecognitionResult
from ml.temporal.schemas import TemporalPolicyConfig, TemporalRecognitionResult, TemporalState
from ml.temporal.stabilizer import TemporalIdentityStabilizer
from ml.presence.schemas import PresenceMode, PresenceState, PresenceEventType, PresenceSession
from ml.presence.state_machine import PRESENCE_PRESETS
from ml.presence.manager import PresenceManager


class MockModernRecognitionPipeline:
    """Mock recognition pipeline for deterministic API testing without heavy weights."""

    def __init__(self, fixed_identity="Alice", fixed_sim=0.85):
        self.fixed_identity = fixed_identity
        self.fixed_sim = fixed_sim
        self.threshold = 0.24

    def recognize(self, rgb_image: np.ndarray) -> ModernRecognitionResult:
        if rgb_image is None or rgb_image.size == 0:
            return ModernRecognitionResult(
                identity=None,
                best_candidate="Unknown",
                similarity=-1.0,
                threshold=self.threshold,
                recognized=False,
                reason="invalid_image"
            )

        is_rec = self.fixed_identity is not None
        return ModernRecognitionResult(
            identity=self.fixed_identity if is_rec else None,
            best_candidate=self.fixed_identity or "Unknown",
            similarity=self.fixed_sim if is_rec else 0.10,
            threshold=self.threshold,
            recognized=is_rec,
            bbox=(10, 100, 100, 10),
            latency_ms=5.0,
            reason="accepted" if is_rec else "below_threshold"
        )


def _create_mock_app(mock_id="Alice"):
    rec_pipe = MockModernRecognitionPipeline(fixed_identity=mock_id)
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
    return app


@pytest.fixture
def client():
    app = _create_mock_app(mock_id="Alice")
    with app.test_client() as c:
        yield c


def _encode_test_image(color="blue", size=(100, 100)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "runtime_status" in data
    assert "timestamp" in data


def test_runtime_status_endpoint(client):
    res = client.get("/api/v1/runtime/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "STOPPED"
    assert data["frame_counter"] == 0
    assert data["active_sessions_count"] == 0


def test_runtime_start_endpoint(client):
    res = client.post("/api/v1/runtime/start")
    assert res.status_code == 200
    assert res.get_json()["status"] == "RUNNING"


def test_runtime_stop_endpoint(client):
    client.post("/api/v1/runtime/start")
    res = client.post("/api/v1/runtime/stop", json={"reason": "test_shutdown"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "STOPPED"
    assert "test_shutdown" in data["message"]


def test_runtime_reset_endpoint(client):
    client.post("/api/v1/runtime/start")
    b64_img = _encode_test_image()
    client.post("/api/v1/runtime/process-frame", json={"image": b64_img})

    res = client.post("/api/v1/runtime/reset")
    assert res.status_code == 200
    assert res.get_json()["status"] == "STOPPED"

    status_res = client.get("/api/v1/runtime/status")
    assert status_res.get_json()["frame_counter"] == 0


def test_process_frame_valid_recognized_image(client):
    b64_img = _encode_test_image()
    res = client.post("/api/v1/runtime/process-frame", json={"image": b64_img})
    assert res.status_code == 200
    data = res.get_json()
    assert data["frame_index"] == 1
    assert data["recognition"]["identity"] == "Alice"
    assert data["recognition"]["recognized"] is True
    assert "latencies" in data
    assert "presence_events" in data


def test_process_frame_missing_image(client):
    res = client.post("/api/v1/runtime/process-frame", json={})
    assert res.status_code == 400
    data = res.get_json()
    assert data["error"] == "missing_image"


def test_process_frame_empty_image(client):
    res = client.post("/api/v1/runtime/process-frame", json={"image": ""})
    assert res.status_code == 400
    data = res.get_json()
    assert data["error"] == "empty_image"


def test_process_frame_invalid_base64(client):
    res = client.post("/api/v1/runtime/process-frame", json={"image": "not_valid_base64!!!"})
    assert res.status_code == 422
    data = res.get_json()
    assert data["error"] == "unprocessable_image"


def test_process_frame_unreadable_image_bytes(client):
    # Valid base64 of plain text bytes that cannot be parsed as an image
    bad_bytes = base64.b64encode(b"not an image").decode("utf-8")
    res = client.post("/api/v1/runtime/process-frame", json={"image": bad_bytes})
    assert res.status_code == 422
    data = res.get_json()
    assert data["error"] == "unprocessable_image"


def test_runtime_unavailable_when_none():
    app = create_app(config_path="config/config.yaml")
    app.runtime_service = None
    with app.test_client() as c:
        res = c.get("/api/v1/runtime/status")
        assert res.status_code == 503
        assert res.get_json()["error"] == "runtime_uninitialized"


def test_runtime_frame_result_serialization_no_embeddings(client):
    b64_img = _encode_test_image()
    res = client.post("/api/v1/runtime/process-frame", json={"image": b64_img})
    data = res.get_json()
    # Confirm no raw embedding vector or model filesystem path leaked in payload
    assert "embedding" not in str(data).lower()
    assert "onnx" not in str(data).lower()


def test_presence_active_endpoint(client):
    b64_img = _encode_test_image()
    # Feed 3 frames to establish PRESENT session for Alice
    for _ in range(3):
        client.post("/api/v1/runtime/process-frame", json={"image": b64_img})

    res = client.get("/api/v1/presence/active")
    assert res.status_code == 200
    data = res.get_json()
    assert data["active_sessions_count"] == 1
    assert data["sessions"][0]["identity"] == "Alice"
    assert data["sessions"][0]["is_active"] is True


def test_presence_history_endpoint(client):
    b64_img = _encode_test_image()
    # Establish PRESENT
    for _ in range(3):
        client.post("/api/v1/runtime/process-frame", json={"image": b64_img})

    # Graceful stop archives session to history
    client.post("/api/v1/runtime/stop")

    res = client.get("/api/v1/presence/history")
    assert res.status_code == 200
    data = res.get_json()
    assert data["archived_sessions_count"] == 1
    assert data["sessions"][0]["identity"] == "Alice"


def test_identity_presence_endpoint(client):
    b64_img = _encode_test_image()
    for _ in range(3):
        client.post("/api/v1/runtime/process-frame", json={"image": b64_img})

    res = client.get("/api/v1/presence/identity/Alice")
    assert res.status_code == 200
    data = res.get_json()
    assert data["identity"] == "Alice"
    assert data["state"] == "PRESENT"
    assert data["active_session"] is not None


def test_legacy_index_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Face Recognition" in res.data


def test_legacy_recognize_deprecated_header(client):
    img = Image.new("RGB", (100, 100), color="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    res = client.post("/recognize", json={"image": b64_str})
    assert res.status_code == 200
    assert res.headers.get("X-API-Deprecated") == "true"
    assert "recognized" in res.get_json()
