from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from ml.detector import FaceDetection
from ml.quality.schemas import FaceQualityMetrics, QualityThresholds, QualityMode
from ml.quality.metrics import (
    extract_face_crop_gray,
    compute_face_dimensions,
    compute_blur_score,
    compute_brightness_score,
    compute_contrast_score,
    compute_alignment_quality,
    compute_pose_quality,
    compute_composite_quality_score
)


# Presets for operational modes
PRESET_THRESHOLDS: Dict[QualityMode, QualityThresholds] = {
    QualityMode.STRICT: QualityThresholds(
        min_face_width=60,
        min_face_height=60,
        min_face_area_ratio=0.03,
        min_blur_score=75.0,
        min_brightness=50.0,
        max_brightness=205.0,
        min_contrast=25.0,
        min_detection_confidence=0.80,
        min_alignment_quality=0.65,
        min_pose_quality=0.55
    ),
    QualityMode.BALANCED: QualityThresholds(
        min_face_width=40,
        min_face_height=40,
        min_face_area_ratio=0.015,
        min_blur_score=40.0,
        min_brightness=35.0,
        max_brightness=225.0,
        min_contrast=18.0,
        min_detection_confidence=0.60,
        min_alignment_quality=0.50,
        min_pose_quality=0.40
    ),
    QualityMode.LENIENT: QualityThresholds(
        min_face_width=25,
        min_face_height=25,
        min_face_area_ratio=0.008,
        min_blur_score=20.0,
        min_brightness=20.0,
        max_brightness=240.0,
        min_contrast=12.0,
        min_detection_confidence=0.45,
        min_alignment_quality=0.35,
        min_pose_quality=0.25
    )
}


class FaceQualityAssessor:
    """
    Face Quality Assessment (FQA) component.
    Evaluates detected faces across physical and geometric quality signals,
    flagging substandard samples prior to embedding extraction.
    """

    def __init__(
        self,
        thresholds: Optional[QualityThresholds] = None,
        mode: QualityMode = QualityMode.BALANCED,
        enabled: bool = True
    ):
        self.mode = mode
        self.enabled = enabled
        self.thresholds = thresholds or PRESET_THRESHOLDS.get(mode, PRESET_THRESHOLDS[QualityMode.BALANCED])

    @classmethod
    def from_config(cls, quality_config: Dict[str, Any]) -> "FaceQualityAssessor":
        """Constructs a FaceQualityAssessor from system configuration dictionary."""
        enabled = quality_config.get("enabled", True)
        mode_str = quality_config.get("mode", "balanced").lower()

        try:
            mode = QualityMode(mode_str)
        except ValueError:
            mode = QualityMode.BALANCED

        base_thresholds = PRESET_THRESHOLDS.get(mode, PRESET_THRESHOLDS[QualityMode.BALANCED])

        # Allow explicit threshold overrides if defined in config
        custom_thresholds = QualityThresholds(
            min_face_width=quality_config.get("min_face_width", base_thresholds.min_face_width),
            min_face_height=quality_config.get("min_face_height", base_thresholds.min_face_height),
            min_face_area_ratio=quality_config.get("min_face_area_ratio", base_thresholds.min_face_area_ratio),
            min_blur_score=quality_config.get("min_blur_score", base_thresholds.min_blur_score),
            min_brightness=quality_config.get("min_brightness", base_thresholds.min_brightness),
            max_brightness=quality_config.get("max_brightness", base_thresholds.max_brightness),
            min_contrast=quality_config.get("min_contrast", base_thresholds.min_contrast),
            min_detection_confidence=quality_config.get("min_detection_confidence", base_thresholds.min_detection_confidence),
            min_alignment_quality=quality_config.get("min_alignment_quality", base_thresholds.min_alignment_quality),
            min_pose_quality=quality_config.get("min_pose_quality", base_thresholds.min_pose_quality)
        )

        return cls(thresholds=custom_thresholds, mode=mode, enabled=enabled)

    def assess(
        self,
        rgb_image: np.ndarray,
        detection: FaceDetection
    ) -> FaceQualityMetrics:
        """
        Assesses face quality for a detected face.

        Args:
            rgb_image (np.ndarray): Full RGB image array.
            detection (FaceDetection): Detection containing bbox, confidence, and landmarks.

        Returns:
            FaceQualityMetrics: Complete quality feature evaluation and decision status.
        """
        bbox = detection.bbox
        conf = float(detection.confidence)
        landmarks = detection.landmarks

        # 1. Dimensions
        w, h, area, area_ratio = compute_face_dimensions(bbox, rgb_image.shape)

        # 2. Grayscale Crop Metrics
        crop_gray = extract_face_crop_gray(rgb_image, bbox)
        blur = compute_blur_score(crop_gray)
        brightness = compute_brightness_score(crop_gray)
        contrast = compute_contrast_score(crop_gray)

        # 3. Geometric Alignment & Pose Proxies
        align_qual = compute_alignment_quality(landmarks, bbox)
        pose_qual = compute_pose_quality(landmarks)

        # 4. Composite Quality Index
        overall = compute_composite_quality_score(
            blur_score=blur,
            brightness=brightness,
            contrast=contrast,
            detection_conf=conf,
            alignment_qual=align_qual,
            pose_qual=pose_qual,
            min_blur=self.thresholds.min_blur_score
        )

        # 5. Threshold Checks & Rejection Reasons
        rejections: List[str] = []

        if w < self.thresholds.min_face_width or h < self.thresholds.min_face_height:
            rejections.append("low_resolution")
        if area_ratio < self.thresholds.min_face_area_ratio:
            rejections.append("face_too_small")
        if blur < self.thresholds.min_blur_score:
            rejections.append("blurry")
        if brightness < self.thresholds.min_brightness:
            rejections.append("underexposed")
        elif brightness > self.thresholds.max_brightness:
            rejections.append("overexposed")
        if contrast < self.thresholds.min_contrast:
            rejections.append("low_contrast")
        if conf < self.thresholds.min_detection_confidence:
            rejections.append("low_detection_confidence")
        if align_qual < self.thresholds.min_alignment_quality:
            rejections.append("poor_alignment")
        if pose_qual < self.thresholds.min_pose_quality:
            rejections.append("extreme_pose")

        status = "poor" if (rejections and self.enabled) else "good"

        return FaceQualityMetrics(
            face_width=w,
            face_height=h,
            face_area=area,
            face_area_ratio=area_ratio,
            blur_score=blur,
            brightness_score=brightness,
            contrast_score=contrast,
            detection_confidence=conf,
            alignment_quality=align_qual,
            pose_quality=pose_qual,
            overall_quality=overall,
            quality_status=status,
            rejection_reasons=rejections
        )
