import os
import pytest
import numpy as np
from PIL import Image

from ml.detector import DlibHOGDetector
from ml.embedder import DlibEmbedder
from ml.matcher import EuclideanMatcher
from ml.pipeline import FaceRecognitionPipeline, RecognitionResult

def test_recognition_result_to_dict():
    result = RecognitionResult(
        identity="Alice",
        distance=0.34567,
        location=(10, 50, 50, 10),
        recognized=True
    )
    d = result.to_dict()
    assert d["identity"] == "Alice"
    assert d["distance"] == 0.3457
    assert d["location"] == [10, 50, 50, 10]
    assert d["recognized"] is True


def test_pipeline_on_blank_image():
    pipeline = FaceRecognitionPipeline(
        detector=DlibHOGDetector(),
        embedder=DlibEmbedder(),
        matcher=EuclideanMatcher()
    )
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    results = pipeline.process_image(blank)
    assert isinstance(results, list)
    assert len(results) == 0


def test_pipeline_on_real_image():
    test_path = "dataset/pavan/pavan_photo.jpeg"
    if not os.path.exists(test_path):
        pytest.skip("Test image not found.")

    img = np.array(Image.open(test_path).convert("RGB"))

    # Extract embedding to populate gallery
    detector = DlibHOGDetector()
    embedder = DlibEmbedder()
    locs = detector.detect(img)
    if not locs:
        pytest.skip("No face detected in test image.")

    embs = embedder.embed(img, locs)
    matcher = EuclideanMatcher(
        known_encodings=[embs[0]],
        known_names=["Pavan"],
        threshold=0.6
    )

    pipeline = FaceRecognitionPipeline(
        detector=detector,
        embedder=embedder,
        matcher=matcher
    )

    results = pipeline.process_image(img)
    assert len(results) == 1
    assert results[0].identity == "Pavan"
    assert results[0].recognized is True
    assert results[0].distance < 0.1  # Identical image should have near zero distance
