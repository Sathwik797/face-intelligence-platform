import pytest
import numpy as np
from ml.aligner import FaceAligner, DEFAULT_ARCFACE_CANONICAL_5_POINTS

def test_aligner_initialization():
    aligner = FaceAligner(output_size=(112, 112))
    assert aligner.output_size == (112, 112)
    assert aligner.canonical_points.shape == (5, 2)


def test_aligner_output_dimensions():
    aligner = FaceAligner(output_size=(112, 112))
    rgb_img = np.zeros((300, 300, 3), dtype=np.uint8)

    # Dummy 5 landmarks around center of 300x300 image
    landmarks = np.array([
        [100.0, 100.0],  # left eye
        [200.0, 100.0],  # right eye
        [150.0, 150.0],  # nose
        [120.0, 200.0],  # left mouth
        [180.0, 200.0]   # right mouth
    ], dtype=np.float32)

    aligned = aligner.align(rgb_img, landmarks)
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_aligner_custom_size():
    aligner = FaceAligner(output_size=(160, 160))
    rgb_img = np.zeros((300, 300, 3), dtype=np.uint8)
    landmarks = np.array([
        [100.0, 100.0],
        [200.0, 100.0],
        [150.0, 150.0],
        [120.0, 200.0],
        [180.0, 200.0]
    ], dtype=np.float32)

    aligned = aligner.align(rgb_img, landmarks)
    assert aligned.shape == (160, 160, 3)


def test_aligner_invalid_inputs():
    aligner = FaceAligner()

    with pytest.raises(ValueError):
        aligner.align(None, DEFAULT_ARCFACE_CANONICAL_5_POINTS)

    with pytest.raises(ValueError):
        aligner.align(np.zeros((100, 100, 3), dtype=np.uint8), None)

    with pytest.raises(ValueError):
        # 4 landmarks instead of 5
        aligner.align(np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((4, 2)))


def test_aligner_batch():
    aligner = FaceAligner()
    rgb_img = np.zeros((300, 300, 3), dtype=np.uint8)
    lms1 = np.array([[100, 100], [200, 100], [150, 150], [120, 200], [180, 200]], dtype=np.float32)
    lms2 = np.array([[80, 80], [160, 80], [120, 120], [90, 160], [150, 160]], dtype=np.float32)

    batch_aligned = aligner.align_batch(rgb_img, [lms1, lms2])
    assert len(batch_aligned) == 2
    assert batch_aligned[0].shape == (112, 112, 3)
    assert batch_aligned[1].shape == (112, 112, 3)
