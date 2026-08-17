from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple


class QualityMode(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"
    LENIENT = "lenient"


@dataclass
class FaceQualityMetrics:
    """
    Structured representation of all measured face quality metrics for a detected face.

    Attributes:
        face_width (int): Bounding box width in pixels.
        face_height (int): Bounding box height in pixels.
        face_area (int): Bounding box area (width * height) in pixels.
        face_area_ratio (float): Ratio of face area to total image area.
        blur_score (float): Sharpness measure (Variance of Laplacian on grayscale face crop).
        brightness_score (float): Mean grayscale intensity in [0, 255].
        contrast_score (float): Standard deviation of grayscale intensity in [0, 255].
        detection_confidence (float): Detector confidence score in [0.0, 1.0].
        alignment_quality (float): Landmark geometry and dispersion sanity in [0.0, 1.0].
        pose_quality (float): Proxy for frontal symmetry from eye-nose geometry in [0.0, 1.0].
        overall_quality (Optional[float]): Composite normalized quality index in [0.0, 1.0].
        quality_status (str): Decision status ("good" or "poor").
        rejection_reasons (List[str]): List of reasons if rejected (empty if "good").
    """
    face_width: int
    face_height: int
    face_area: int
    face_area_ratio: float
    blur_score: float
    brightness_score: float
    contrast_score: float
    detection_confidence: float
    alignment_quality: float
    pose_quality: float
    overall_quality: Optional[float] = None
    quality_status: str = "good"
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_width": int(self.face_width),
            "face_height": int(self.face_height),
            "face_area": int(self.face_area),
            "face_area_ratio": round(float(self.face_area_ratio), 6),
            "blur_score": round(float(self.blur_score), 4),
            "brightness_score": round(float(self.brightness_score), 2),
            "contrast_score": round(float(self.contrast_score), 2),
            "detection_confidence": round(float(self.detection_confidence), 4),
            "alignment_quality": round(float(self.alignment_quality), 4),
            "pose_quality": round(float(self.pose_quality), 4),
            "overall_quality": round(float(self.overall_quality), 4) if self.overall_quality is not None else None,
            "quality_status": self.quality_status,
            "rejection_reasons": list(self.rejection_reasons)
        }


@dataclass
class QualityThresholds:
    """
    Threshold constraints defining acceptable face quality boundaries.
    """
    min_face_width: int = 40
    min_face_height: int = 40
    min_face_area_ratio: float = 0.015
    min_blur_score: float = 40.0
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_contrast: float = 20.0
    min_detection_confidence: float = 0.60
    min_alignment_quality: float = 0.50
    min_pose_quality: float = 0.40

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_face_width": self.min_face_width,
            "min_face_height": self.min_face_height,
            "min_face_area_ratio": self.min_face_area_ratio,
            "min_blur_score": self.min_blur_score,
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "min_contrast": self.min_contrast,
            "min_detection_confidence": self.min_detection_confidence,
            "min_alignment_quality": self.min_alignment_quality,
            "min_pose_quality": self.min_pose_quality
        }
