import os
import pytest
import numpy as np
from PIL import Image

from ml.detector import DlibHOGDetector, ModernFaceDetector, FaceDetection

@pytest.fixture
def sample_face_image():
    """Loads a real face image from dataset if available, or creates an RGB fixture."""
    test_path = "dataset/pavan/pavan_photo.jpeg"
    if os.path.exists(test_path):
        return np.array(Image.open(test_path).convert("RGB"))
    return np.zeros((300, 300, 3), dtype=np.uint8)


# --- DlibHOGDetector Tests (Phase 1 Baseline) ---

def test_dlib_detector_initialization():
    detector = DlibHOGDetector(number_of_times_to_upsample=1)
    assert detector.number_of_times_to_upsample == 1


def test_dlib_detector_on_valid_face(sample_face_image):
    detector = DlibHOGDetector()
    locations = detector.detect(sample_face_image)
    assert isinstance(locations, list)
    if os.path.exists("dataset/pavan/pavan_photo.jpeg"):
        assert len(locations) >= 1
        top, right, bottom, left = locations[0]
        assert top < bottom
        assert left < right


def test_dlib_detector_on_blank_image():
    detector = DlibHOGDetector()
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    locations = detector.detect(blank)
    assert isinstance(locations, list)
    assert len(locations) == 0


def test_dlib_detector_invalid_input():
    detector = DlibHOGDetector()
    with pytest.raises(ValueError):
        detector.detect(None)

    with pytest.raises(ValueError):
        detector.detect(np.zeros((100, 100), dtype=np.uint8))


# --- ModernFaceDetector Tests (Phase 3 Modern CNN + Landmarks) ---

def test_modern_detector_initialization():
    detector = ModernFaceDetector(score_threshold=0.7, nms_threshold=0.3)
    assert detector.score_threshold == 0.7
    assert detector.nms_threshold == 0.3


def test_modern_detector_on_valid_face(sample_face_image):
    detector = ModernFaceDetector(score_threshold=0.5)
    detections = detector.detect_faces(sample_face_image)
    assert isinstance(detections, list)

    if os.path.exists("dataset/pavan/pavan_photo.jpeg"):
        assert len(detections) >= 1
        d = detections[0]
        assert isinstance(d, FaceDetection)
        assert 0.0 <= d.confidence <= 1.0
        assert d.landmarks is not None
        assert d.landmarks.shape == (5, 2)
        assert d.bbox[0] < d.bbox[2]  # top < bottom
        assert d.bbox[3] < d.bbox[1]  # left < right

        # Check dictionary conversion
        d_dict = d.to_dict()
        assert "bbox" in d_dict
        assert "landmarks" in d_dict
        assert "confidence" in d_dict


def test_modern_detector_compatibility_interface(sample_face_image):
    """Verify that detect() returns List[Tuple[int, int, int, int]] for interface consistency."""
    detector = ModernFaceDetector(score_threshold=0.5)
    bboxes = detector.detect(sample_face_image)
    assert isinstance(bboxes, list)
    if os.path.exists("dataset/pavan/pavan_photo.jpeg"):
        assert len(bboxes) >= 1
        assert len(bboxes[0]) == 4


def test_modern_detector_on_blank_image():
    detector = ModernFaceDetector()
    blank = np.zeros((250, 250, 3), dtype=np.uint8)
    detections = detector.detect_faces(blank)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_modern_detector_invalid_input():
    detector = ModernFaceDetector()
    with pytest.raises(ValueError):
        detector.detect_faces(None)

    with pytest.raises(ValueError):
        detector.detect_faces(np.zeros((100, 100), dtype=np.uint8))
