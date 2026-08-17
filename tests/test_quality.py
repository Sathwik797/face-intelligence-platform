import pytest
import numpy as np
import cv2

from ml.detector import FaceDetection, ModernFaceDetector
from ml.aligner import FaceAligner
from ml.embedder import ArcFaceEmbedder
from ml.gallery import IdentityGallery
from ml.pipeline import ModernRecognitionPipeline, ModernRecognitionResult
from ml.quality import (
    FaceQualityAssessor,
    FaceQualityMetrics,
    QualityMode,
    QualityThresholds,
    PRESET_THRESHOLDS,
    extract_face_crop_gray,
    compute_face_dimensions,
    compute_blur_score,
    compute_brightness_score,
    compute_contrast_score,
    compute_alignment_quality,
    compute_pose_quality,
    compute_composite_quality_score
)


def test_compute_face_dimensions():
    bbox = (20, 120, 140, 20)  # top=20, right=120, bottom=140, left=20 -> w=100, h=120
    img_shape = (200, 200, 3)
    w, h, area, ratio = compute_face_dimensions(bbox, img_shape)
    assert w == 100
    assert h == 120
    assert area == 12000
    assert ratio == pytest.approx(12000 / (200 * 200), rel=1e-4)


def test_blur_sharpness_calculation():
    # 1. Sharp checkerboard
    sharp_img = np.zeros((100, 100), dtype=np.uint8)
    sharp_img[::2, ::2] = 255
    sharp_img[1::2, 1::2] = 255
    sharp_score = compute_blur_score(sharp_img)

    # 2. Gaussian blurred version
    blurred_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)
    blurred_score = compute_blur_score(blurred_img)

    assert sharp_score > 500.0
    assert blurred_score < sharp_score * 0.1


def test_brightness_and_contrast_calculation():
    # Dark uniform image
    dark_img = np.full((50, 50), 20, dtype=np.uint8)
    assert compute_brightness_score(dark_img) == pytest.approx(20.0, abs=0.1)
    assert compute_contrast_score(dark_img) == pytest.approx(0.0, abs=0.1)

    # High contrast image (half dark, half bright)
    contrast_img = np.zeros((50, 50), dtype=np.uint8)
    contrast_img[:, :25] = 20
    contrast_img[:, 25:] = 220
    assert compute_brightness_score(contrast_img) == pytest.approx(120.0, abs=1.0)
    assert compute_contrast_score(contrast_img) > 90.0


def test_alignment_and_pose_quality():
    bbox = (20, 120, 140, 20)  # [top, right, bottom, left]
    # Perfect frontal landmarks: le=(50, 50), re=(90, 50), nose=(70, 80), lm=(55, 110), rm=(85, 110)
    frontal_landmarks = np.array([
        [50.0, 50.0],
        [90.0, 50.0],
        [70.0, 80.0],
        [55.0, 110.0],
        [85.0, 110.0]
    ], dtype=np.float32)

    align_score = compute_alignment_quality(frontal_landmarks, bbox)
    pose_score = compute_pose_quality(frontal_landmarks)

    assert align_score > 0.85
    assert pose_score > 0.85

    # Tilted/distorted landmarks
    tilted_landmarks = np.array([
        [50.0, 30.0],
        [90.0, 90.0],  # Severe roll tilt
        [70.0, 70.0],
        [55.0, 110.0],
        [85.0, 110.0]
    ], dtype=np.float32)

    tilted_align = compute_alignment_quality(tilted_landmarks, bbox)
    assert tilted_align < align_score


def test_quality_assessor_good_face():
    # Synthetic clean face crop
    rgb_img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Add texture for sharpness and contrast
    rgb_img[50:150, 50:150, :] = np.random.RandomState(42).randint(50, 200, size=(100, 100, 3), dtype=np.uint8)

    bbox = (50, 150, 150, 50)
    landmarks = np.array([
        [75.0, 80.0],
        [125.0, 80.0],
        [100.0, 105.0],
        [80.0, 130.0],
        [120.0, 130.0]
    ], dtype=np.float32)

    detection = FaceDetection(bbox=bbox, confidence=0.95, landmarks=landmarks)
    assessor = FaceQualityAssessor(mode=QualityMode.BALANCED)
    result = assessor.assess(rgb_img, detection)

    assert isinstance(result, FaceQualityMetrics)
    assert result.quality_status == "good"
    assert len(result.rejection_reasons) == 0
    assert result.overall_quality is not None
    assert result.overall_quality > 0.5


def test_quality_assessor_rejection_reasons():
    # 1. Very blurry, dark image
    dark_blur_img = np.full((200, 200, 3), 10, dtype=np.uint8)
    bbox = (50, 150, 150, 50)
    detection = FaceDetection(bbox=bbox, confidence=0.30, landmarks=None)

    assessor = FaceQualityAssessor(mode=QualityMode.BALANCED)
    res = assessor.assess(dark_blur_img, detection)

    assert res.quality_status == "poor"
    assert "blurry" in res.rejection_reasons
    assert "underexposed" in res.rejection_reasons
    assert "low_detection_confidence" in res.rejection_reasons


def test_quality_modes_strict_balanced_lenient():
    # Sample that passes lenient but fails strict
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[70:130, 70:130, :] = np.random.RandomState(42).randint(80, 170, size=(60, 60, 3), dtype=np.uint8)

    bbox = (70, 130, 130, 70)  # 60x60 bbox
    landmarks = np.array([
        [85.0, 90.0],
        [115.0, 90.0],
        [100.0, 105.0],
        [90.0, 120.0],
        [110.0, 120.0]
    ], dtype=np.float32)

    detection = FaceDetection(bbox=bbox, confidence=0.70, landmarks=landmarks)

    assessor_strict = FaceQualityAssessor(mode=QualityMode.STRICT)
    assessor_lenient = FaceQualityAssessor(mode=QualityMode.LENIENT)

    res_strict = assessor_strict.assess(img, detection)
    res_lenient = assessor_lenient.assess(img, detection)

    assert res_strict.quality_status == "poor"  # Fails strict (e.g. confidence < 0.80)
    assert res_lenient.quality_status == "good"  # Passes lenient (confidence >= 0.45)


def test_quality_integration_in_pipeline():
    detector = ModernFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    gallery = IdentityGallery()

    # Enforce strict assessor that will reject low-resolution/blurry synthetic face
    strict_assessor = FaceQualityAssessor(mode=QualityMode.STRICT, enabled=True)

    pipeline = ModernRecognitionPipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        gallery=gallery,
        threshold=0.24,
        quality_assessor=strict_assessor
    )

    # Uniform gray image with a dummy face drawn (blurry, low contrast)
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    cv2.circle(img, (100, 100), 30, (140, 140, 140), -1)

    res = pipeline.recognize(img)
    assert isinstance(res, ModernRecognitionResult)
    # Either no face detected or rejected by quality
    if res.reason.startswith("quality_rejected"):
        assert res.recognized is False
        assert res.quality is not None
        assert res.quality.quality_status == "poor"
