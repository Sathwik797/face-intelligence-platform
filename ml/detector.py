import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import face_recognition

@dataclass
class FaceDetection:
    """
    Structured representation of a face detection result.

    Attributes:
        bbox (Tuple[int, int, int, int]): Bounding box in CSS order (top, right, bottom, left).
        confidence (float): Detection confidence score in [0.0, 1.0].
        landmarks (Optional[np.ndarray]): 5 facial landmarks shape (5, 2) in image coordinates:
                                          [left_eye, right_eye, nose, left_mouth, right_mouth].
        box_xywh (Tuple[int, int, int, int]): Bounding box in (x, y, w, h) format.
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray] = None
    box_xywh: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "box_xywh": list(self.box_xywh) if self.box_xywh is not None else None
        }


class BaseDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(self, rgb_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces and return bounding boxes in CSS format (top, right, bottom, left).
        Maintains compatibility with Phase 1 pipeline interface.
        """
        pass

    def detect_faces(self, rgb_image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces and return rich FaceDetection objects with confidence and landmarks.
        Default fallback creates FaceDetection with default confidence for detectors without landmarks.
        """
        bboxes = self.detect(rgb_image)
        return [FaceDetection(bbox=b, confidence=1.0, landmarks=None) for b in bboxes]


class DlibHOGDetector(BaseDetector):
    """
    Phase 1 Baseline Face Detector using dlib's HOG + Linear SVM.

    Characteristics:
        - CPU-bound inference.
        - Returns bounding boxes without 5-point facial landmarks.
        - Preserved as Experiment E1 baseline.
    """

    def __init__(self, number_of_times_to_upsample: int = 1):
        self.number_of_times_to_upsample = number_of_times_to_upsample

    def detect(self, rgb_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if rgb_image is None or not isinstance(rgb_image, np.ndarray):
            raise ValueError("Input image must be a valid numpy array.")

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {rgb_image.shape}.")

        if rgb_image.size == 0:
            return []

        locations = face_recognition.face_locations(
            rgb_image,
            number_of_times_to_upsample=self.number_of_times_to_upsample,
            model="hog"
        )
        return locations


class ModernFaceDetector(BaseDetector):
    """
    Modern Deep CNN Face Detector using OpenCV YuNet (ONNX runtime).

    Features:
        - Robust under scale variations, severe yaw/pitch, and non-frontal poses.
        - Generates 5 precise facial landmarks: left eye, right eye, nose, left mouth, right mouth.
        - Provides continuous detection confidence scores in [0.0, 1.0].
        - Fast CPU inference (<15ms per frame).
    """

    DEFAULT_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    DEFAULT_MODEL_PATH = "ml/models/face_detection_yunet_2023mar.onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000
    ):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)

        self._ensure_model_exists()
        self._detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k
        )

    def _ensure_model_exists(self):
        """Ensures the ONNX model file is available locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            print(f"[INFO] Downloading YuNet face detection ONNX model to {self.model_path}...")
            urllib.request.urlretrieve(self.DEFAULT_MODEL_URL, self.model_path)

    def detect_faces(self, rgb_image: np.ndarray) -> List[FaceDetection]:
        """
        Detects faces in an RGB image and extracts 5-point facial landmarks.

        Args:
            rgb_image (np.ndarray): Image array (H, W, 3) in RGB format.

        Returns:
            List[FaceDetection]: List of detected faces with bounding boxes, confidence, and 5 landmarks.
        """
        if rgb_image is None or not isinstance(rgb_image, np.ndarray):
            raise ValueError("Input image must be a valid numpy array.")

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {rgb_image.shape}.")

        h, w, _ = rgb_image.shape
        if h == 0 or w == 0:
            return []

        # Convert RGB to BGR for OpenCV YuNet input
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Set dynamic input resolution
        self._detector.setInputSize((w, h))
        _, raw_faces = self._detector.detect(bgr_image)

        if raw_faces is None or len(raw_faces) == 0:
            return []

        detections: List[FaceDetection] = []
        for face in raw_faces:
            # Parse bounding box [x, y, w, h]
            x, y, bw, bh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # Clamp coordinates to image boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            # CSS format: (top, right, bottom, left)
            css_bbox = (y1, x2, y2, x1)

            # Parse 5 facial landmarks in order:
            # YuNet output: right_eye(4,5), left_eye(6,7), nose(8,9), right_mouth(10,11), left_mouth(12,13)
            # Standard alignment order: left_eye, right_eye, nose, left_mouth, right_mouth
            # Note: in image coordinates, right_eye is viewer's left, left_eye is viewer's right
            right_eye = [float(face[4]), float(face[5])]
            left_eye = [float(face[6]), float(face[7])]
            nose = [float(face[8]), float(face[9])]
            right_mouth = [float(face[10]), float(face[11])]
            left_mouth = [float(face[12]), float(face[13])]

            # Ensure viewer-left (smaller x) is index 0, viewer-right (larger x) is index 1
            if right_eye[0] > left_eye[0]:
                pt_left_eye, pt_right_eye = left_eye, right_eye
            else:
                pt_left_eye, pt_right_eye = right_eye, left_eye

            if right_mouth[0] > left_mouth[0]:
                pt_left_mouth, pt_right_mouth = left_mouth, right_mouth
            else:
                pt_left_mouth, pt_right_mouth = right_mouth, left_mouth

            landmarks_5 = np.array([
                pt_left_eye,
                pt_right_eye,
                nose,
                pt_left_mouth,
                pt_right_mouth
            ], dtype=np.float32)

            confidence = float(face[14])

            detections.append(
                FaceDetection(
                    bbox=css_bbox,
                    confidence=confidence,
                    landmarks=landmarks_5,
                    box_xywh=(x1, y1, x2 - x1, y2 - y1)
                )
            )

        return detections

    def detect(self, rgb_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extracts only bounding boxes in CSS format for interface compatibility."""
        detections = self.detect_faces(rgb_image)
        return [d.bbox for d in detections]
