"""Machine Learning and Computer Vision Modules for Face Recognition Attendance System."""

from ml.detector import BaseDetector, DlibHOGDetector, ModernFaceDetector, FaceDetection
from ml.aligner import FaceAligner
from ml.embedder import BaseEmbedder, DlibEmbedder, ArcFaceEmbedder
from ml.matcher import BaseMatcher, EuclideanMatcher, CosineMatcher
from ml.pipeline import FaceRecognitionPipeline, RecognitionResult

__all__ = [
    "BaseDetector",
    "DlibHOGDetector",
    "ModernFaceDetector",
    "FaceDetection",
    "FaceAligner",
    "BaseEmbedder",
    "DlibEmbedder",
    "ArcFaceEmbedder",
    "BaseMatcher",
    "EuclideanMatcher",
    "CosineMatcher",
    "FaceRecognitionPipeline",
    "RecognitionResult",
]
