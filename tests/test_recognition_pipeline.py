import os
import pytest
import numpy as np
from PIL import Image

from config import load_config
from ml.detector import ModernFaceDetector
from ml.aligner import FaceAligner
from ml.embedder import ArcFaceEmbedder
from ml.gallery import IdentityGallery
from ml.pipeline import ModernRecognitionPipeline, ModernRecognitionResult

@pytest.fixture
def mock_pipeline():
    """Constructs a lightweight ModernRecognitionPipeline with a synthetic enrolled gallery."""
    detector = ModernFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    gallery = IdentityGallery()

    # Enroll a synthetic reference template
    emb_dummy = np.zeros((1, 512), dtype=np.float32)
    emb_dummy[0, 0] = 1.0
    gallery.add_templates("EnrolledPerson", emb_dummy)

    return ModernRecognitionPipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        gallery=gallery,
        threshold=0.45,
        multi_face_policy="highest_confidence"
    )


def test_modern_result_to_dict():
    res = ModernRecognitionResult(
        identity="Alice",
        best_candidate="Alice",
        similarity=0.88,
        threshold=0.45,
        recognized=True,
        bbox=(10, 100, 110, 20),
        landmarks=np.zeros((5, 2)),
        model="arcface_resnet50_512d",
        embedding_dim=512,
        latency_ms=95.5,
        reason="accepted"
    )
    d = res.to_dict()
    assert d["identity"] == "Alice"
    assert d["recognized"] is True
    assert d["similarity"] == 0.88
    assert d["model"] == "arcface_resnet50_512d"
    assert d["embedding_dimension"] == 512


def test_pipeline_on_blank_image(mock_pipeline):
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    res = mock_pipeline.recognize(blank)
    assert isinstance(res, ModernRecognitionResult)
    assert res.identity is None
    assert res.recognized is False
    assert res.reason == "no_face_detected"


def test_pipeline_on_invalid_input(mock_pipeline):
    res = mock_pipeline.recognize(None)
    assert res.identity is None
    assert res.recognized is False
    assert res.reason == "invalid_image"


def test_pipeline_from_config_and_real_image():
    config = load_config("config/config.yaml")
    gallery_path = "data/embeddings/arcface_gallery.npz"

    if not os.path.exists(gallery_path):
        pytest.skip("ArcFace gallery artifact not found.")

    pipeline = ModernRecognitionPipeline.from_config(config, gallery_path=gallery_path)
    assert pipeline.gallery.total_templates > 0

    # Query using an enrolled validation image if present
    test_img_path = "data/evaluation/validation/Alejandro_Toledo/Alejandro_Toledo_0002.jpg"
    if os.path.exists(test_img_path):
        with Image.open(test_img_path) as img:
            rgb_img = np.array(img.convert("RGB"))

        res = pipeline.recognize(rgb_img)
        assert isinstance(res, ModernRecognitionResult)
        assert res.best_candidate == "Alejandro_Toledo"
        assert res.recognized is True
        assert res.similarity > 0.45
        assert res.identity == "Alejandro_Toledo"
