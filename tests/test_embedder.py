import os
import pytest
import numpy as np
from PIL import Image

from ml.embedder import DlibEmbedder, ArcFaceEmbedder

@pytest.fixture
def sample_face_crop():
    """Loads a 112x112 aligned face crop fixture."""
    test_path = "dataset/pavan/pavan_photo.jpeg"
    if os.path.exists(test_path):
        from ml.detector import ModernFaceDetector
        from ml.aligner import FaceAligner
        img = np.array(Image.open(test_path).convert("RGB"))
        det = ModernFaceDetector()
        aligner = FaceAligner()
        faces = det.detect_faces(img)
        if faces and faces[0].landmarks is not None:
            return aligner.align(img, faces[0].landmarks)
    # Synthetic 112x112 RGB face crop
    return np.ones((112, 112, 3), dtype=np.uint8) * 128


# --- DlibEmbedder Tests (Phase 1 Baseline) ---

def test_dlib_embedder_dimension_property():
    embedder = DlibEmbedder()
    assert embedder.embedding_dim == 128
    assert embedder.model_name == "dlib_resnet34_128d"


def test_dlib_embedder_empty_locations():
    embedder = DlibEmbedder()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    embeddings = embedder.embed(img, [])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 128)


def test_dlib_embedder_valid_extraction():
    test_path = "dataset/pavan/pavan_photo.jpeg"
    if not os.path.exists(test_path):
        pytest.skip("Test image not found.")
    img = np.array(Image.open(test_path).convert("RGB"))
    embedder = DlibEmbedder()
    embeddings = embedder.embed(img)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 128)
    assert np.linalg.norm(embeddings[0]) > 0


def test_dlib_embedder_invalid_input():
    embedder = DlibEmbedder()
    with pytest.raises(ValueError):
        embedder.embed(None)


# --- ArcFaceEmbedder Tests (Phase 4 Modern Model) ---

def test_arcface_embedder_dimension_property():
    embedder = ArcFaceEmbedder()
    assert embedder.embedding_dim == 512
    assert embedder.model_name == "arcface_resnet50_512d"


def test_arcface_embedder_single_crop(sample_face_crop):
    embedder = ArcFaceEmbedder()
    emb = embedder.embed(sample_face_crop)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 512)
    assert emb.dtype == np.float32

    # Check finite values
    assert np.all(np.isfinite(emb))

    # Check unit L2 norm
    norm = np.linalg.norm(emb[0])
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_arcface_embedder_determinism(sample_face_crop):
    embedder = ArcFaceEmbedder()
    emb1 = embedder.embed(sample_face_crop)
    emb2 = embedder.embed(sample_face_crop)
    assert np.allclose(emb1, emb2, atol=1e-6)


def test_arcface_embedder_batch(sample_face_crop):
    embedder = ArcFaceEmbedder()
    batch = [sample_face_crop, sample_face_crop.copy(), np.zeros((112, 112, 3), dtype=np.uint8)]
    embs = embedder.embed_batch(batch)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (3, 512)

    # Check that batch output equals single output
    single_emb = embedder.embed(sample_face_crop)
    assert np.allclose(embs[0], single_emb[0], atol=1e-5)


def test_arcface_embedder_invalid_inputs():
    embedder = ArcFaceEmbedder()

    with pytest.raises(ValueError):
        embedder.embed(None)

    with pytest.raises(ValueError):
        # 2D array instead of 3D RGB crop
        embedder.embed(np.zeros((112, 112), dtype=np.uint8))


def test_arcface_embedder_empty_batch():
    embedder = ArcFaceEmbedder()
    embs = embedder.embed_batch([])
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (0, 512)
