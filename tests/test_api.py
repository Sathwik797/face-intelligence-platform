import os
import io
import base64
import pytest
import numpy as np
from PIL import Image

from app import create_app

@pytest.fixture
def client():
    app = create_app("config/config.yaml")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Face Recognition" in response.data


def test_recognize_missing_data(client):
    response = client.post("/recognize", json={})
    assert response.status_code == 400


def test_recognize_empty_image(client):
    response = client.post("/recognize", json={"image": ""})
    assert response.status_code == 400


def test_recognize_blank_image(client):
    # Create blank black image encoded to base64
    img = Image.new("RGB", (100, 100), color="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    response = client.post("/recognize", json={"image": b64_str})
    assert response.status_code == 200
    data = response.get_json()
    assert "recognized" in data
    assert len(data["recognized"]) == 0
