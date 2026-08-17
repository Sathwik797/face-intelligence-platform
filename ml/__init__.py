"""Machine Learning and Computer Vision Modules for Face Recognition Attendance System."""

from ml.detector import BaseDetector, DlibHOGDetector, ModernFaceDetector, FaceDetection
from ml.aligner import FaceAligner
from ml.embedder import BaseEmbedder, DlibEmbedder, ArcFaceEmbedder
from ml.matcher import BaseMatcher, EuclideanMatcher, CosineMatcher
from ml.gallery import IdentityGallery
from ml.quality import FaceQualityAssessor, FaceQualityMetrics, QualityMode, QualityThresholds
from ml.temporal import (
    TemporalIdentityStabilizer,
    RecognitionObservation,
    TemporalRecognitionResult,
    TemporalPolicyConfig,
    TemporalMode,
    TemporalState
)
from ml.presence import (
    PresenceManager,
    PresenceState,
    PresenceEventType,
    PresenceMode,
    PresenceConfig,
    PresenceEvent,
    PresenceSession,
    IdentityPresenceStateMachine,
    PRESENCE_PRESETS
)
from ml.runtime import (
    RuntimeStatus,
    RuntimeConfig,
    StageLatencyMetrics,
    RuntimeFrameResult,
    BaseFrameSource,
    StaticFrameSource,
    SyntheticFrameSource,
    OpenCVFrameSource,
    FaceIntelligenceRuntime
)
from ml.pipeline import (
    FaceRecognitionPipeline,
    RecognitionResult,
    ModernRecognitionPipeline,
    ModernRecognitionResult
)

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
    "IdentityGallery",
    "FaceQualityAssessor",
    "FaceQualityMetrics",
    "QualityMode",
    "QualityThresholds",
    "TemporalIdentityStabilizer",
    "RecognitionObservation",
    "TemporalRecognitionResult",
    "TemporalPolicyConfig",
    "TemporalMode",
    "TemporalState",
    "PresenceManager",
    "PresenceState",
    "PresenceEventType",
    "PresenceMode",
    "PresenceConfig",
    "PresenceEvent",
    "PresenceSession",
    "IdentityPresenceStateMachine",
    "PRESENCE_PRESETS",
    "RuntimeStatus",
    "RuntimeConfig",
    "StageLatencyMetrics",
    "RuntimeFrameResult",
    "BaseFrameSource",
    "StaticFrameSource",
    "SyntheticFrameSource",
    "OpenCVFrameSource",
    "FaceIntelligenceRuntime",
    "FaceRecognitionPipeline",
    "RecognitionResult",
    "ModernRecognitionPipeline",
    "ModernRecognitionResult",
]
