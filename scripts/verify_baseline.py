import os
import sys
import time
import base64
import json
from io import BytesIO
from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import create_app

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")

def run_manual_verification():
    app = create_app("config/config.yaml")
    client = app.test_client()

    print("\n" + "="*50)
    print("PHASE 1 BASELINE MANUAL & API VERIFICATION")
    print("="*50)

    # 1. Test Known Person (Pavan)
    pavan_path = "dataset/pavan/pavan_photo.jpeg"
    if os.path.exists(pavan_path):
        b64_img = image_to_base64(pavan_path)
        start = time.perf_counter()
        resp = client.post("/recognize", json={"image": b64_img})
        latency = (time.perf_counter() - start) * 1000
        data = resp.get_json()
        print(f"[TEST 1: Known Person (Pavan)]")
        print(f"  - Status Code: {resp.status_code}")
        print(f"  - Recognized: {data.get('recognized')}")
        print(f"  - Details: {data.get('details')}")
        print(f"  - Latency: {latency:.2f} ms")
        assert "pavan" in data.get("recognized", [])

    # 2. Test Known Person (Sathwik)
    sathwik_path = "dataset/sathwik/sathwik_photo.jpg"
    if os.path.exists(sathwik_path):
        b64_img = image_to_base64(sathwik_path)
        start = time.perf_counter()
        resp = client.post("/recognize", json={"image": b64_img})
        latency = (time.perf_counter() - start) * 1000
        data = resp.get_json()
        print(f"\n[TEST 2: Known Person (Sathwik)]")
        print(f"  - Status Code: {resp.status_code}")
        print(f"  - Recognized: {data.get('recognized')}")
        print(f"  - Details: {data.get('details')}")
        print(f"  - Latency: {latency:.2f} ms")
        assert "sathwik" in data.get("recognized", [])

    # 3. Test Blank / No Face Image
    blank_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    buf = BytesIO()
    blank_img.save(buf, format="JPEG")
    blank_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    start = time.perf_counter()
    resp = client.post("/recognize", json={"image": blank_b64})
    latency = (time.perf_counter() - start) * 1000
    data = resp.get_json()
    print(f"\n[TEST 3: No Face Image]")
    print(f"  - Status Code: {resp.status_code}")
    print(f"  - Recognized: {data.get('recognized')}")
    print(f"  - Latency: {latency:.2f} ms")
    assert len(data.get("recognized", [])) == 0

    print("\n" + "="*50)
    print("ALL BASELINE VERIFICATIONS SUCCESSFUL!")
    print("="*50)

if __name__ == "__main__":
    run_manual_verification()
