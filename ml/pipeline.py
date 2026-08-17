import os
import time
import pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ml.detector import BaseDetector, DlibHOGDetector, ModernFaceDetector, FaceDetection
from ml.aligner import FaceAligner
from ml.embedder import BaseEmbedder, DlibEmbedder, ArcFaceEmbedder
from ml.matcher import BaseMatcher, EuclideanMatcher, CosineMatcher
from ml.gallery import IdentityGallery
from ml.quality import FaceQualityAssessor, FaceQualityMetrics

# --- Baseline E1 Data Structure ---

@dataclass
class RecognitionResult:
    """Structured result of a baseline face recognition query (Experiment E1)."""
    identity: str
    distance: float
    location: Tuple[int, int, int, int]
    recognized: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity,
            "distance": round(self.distance, 4) if self.distance != float("inf") else None,
            "location": list(self.location),
            "recognized": self.recognized
        }


# --- Modern E2 Data Structure ---

@dataclass
class ModernRecognitionResult:
    """
    Structured result of a modern ArcFace recognition query (Experiment E2).
    Explicitly distinguishes between best_candidate and final open-set recognition decision.
    Includes Phase 8 Face Quality Assessment metrics when enabled.
    """
    identity: Optional[str]              # Enrolled name if recognized, None if rejected as Unknown
    best_candidate: str                  # Closest gallery candidate
    similarity: float                    # Continuous cosine similarity score [-1.0, 1.0]
    threshold: float                     # Decision threshold
    recognized: bool                     # True if similarity >= threshold
    bbox: Optional[Tuple[int, int, int, int]] = None  # CSS bounding box (top, right, bottom, left)
    landmarks: Optional[np.ndarray] = None            # 5-point landmarks
    model: str = "arcface_resnet50_512d"
    embedding_dim: int = 512
    latency_ms: float = 0.0
    reason: str = "accepted"             # "accepted", "below_threshold", "no_face_detected", "multiple_faces_rejected", "quality_rejected: ..."
    quality: Optional[FaceQualityMetrics] = None      # Phase 8 Face Quality Assessment metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity,
            "best_candidate": self.best_candidate,
            "recognized": self.recognized,
            "similarity": round(float(self.similarity), 4),
            "threshold": round(float(self.threshold), 4),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "model": self.model,
            "embedding_dimension": self.embedding_dim,
            "latency_ms": round(float(self.latency_ms), 2),
            "reason": self.reason,
            "quality": self.quality.to_dict() if self.quality is not None else None
        }


# --- Phase 1 Baseline Pipeline (E1) ---

class FaceRecognitionPipeline:
    """
    Phase 1 Baseline Face Recognition Pipeline (Experiment E1).
    Orchestrates: dlib HOG Detection -> dlib 128D Embedding -> Euclidean Matcher.
    """

    def __init__(
        self,
        detector: BaseDetector,
        embedder: BaseEmbedder,
        matcher: BaseMatcher
    ):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        embeddings_path: Optional[str] = None
    ) -> "FaceRecognitionPipeline":
        threshold = config.get("model", {}).get("recognition_threshold", 0.6)
        emb_path = embeddings_path or config.get("paths", {}).get("embeddings_path", "trained_model/face_encodings.pkl")

        detector = DlibHOGDetector()
        embedder = DlibEmbedder()
        matcher = EuclideanMatcher(threshold=threshold)

        if os.path.exists(emb_path):
            with open(emb_path, "rb") as f:
                known_encodings, known_names = pickle.load(f)
            matcher.update_gallery(known_encodings, known_names)

        return cls(detector=detector, embedder=embedder, matcher=matcher)

    def process_image(self, rgb_image: np.ndarray) -> List[RecognitionResult]:
        if rgb_image is None or rgb_image.size == 0:
            return []

        face_locations = self.detector.detect(rgb_image)
        if not face_locations:
            return []

        embeddings = self.embedder.embed(rgb_image, face_locations)

        results: List[RecognitionResult] = []
        for loc, emb in zip(face_locations, embeddings):
            identity, distance, is_recognized = self.matcher.match(emb)
            results.append(
                RecognitionResult(
                    identity=identity,
                    distance=distance,
                    location=loc,
                    recognized=is_recognized
                )
            )

        return results


# --- Phase 5 Modern Pipeline (E2) ---

class ModernRecognitionPipeline:
    """
    Modern Face Recognition & Identity Decision Pipeline (Experiment E2).

    Orchestrates:
        1. RGB Image Input
        2. YuNet Face Detection
        3. Deterministic Primary Face Selection (Highest Confidence Policy)
        4. Face Quality Assessment (Phase 8 FQA) -> Reject poor frames if enabled
        5. 5-Point Landmark Affine Alignment (112x112 Canonical Crop)
        6. ArcFace Deep Embedding Extraction (512D ResNet-50)
        7. L2 Normalization
        8. Multi-template Cosine Similarity Search against IdentityGallery
        9. Open-Set Threshold Decision (Recognized Identity OR Unknown)
    """

    def __init__(
        self,
        detector: ModernFaceDetector,
        aligner: FaceAligner,
        embedder: ArcFaceEmbedder,
        gallery: IdentityGallery,
        threshold: float = 0.24,
        multi_face_policy: str = "highest_confidence",
        quality_assessor: Optional[FaceQualityAssessor] = None
    ):
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.gallery = gallery
        self.threshold = float(threshold)
        self.multi_face_policy = multi_face_policy
        self.quality_assessor = quality_assessor

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        gallery_path: Optional[str] = None
    ) -> "ModernRecognitionPipeline":
        """Factory method constructing ModernRecognitionPipeline from configuration."""
        g_path = gallery_path or config.get("paths", {}).get(
            "gallery_path", "data/embeddings/arcface_gallery.npz"
        )
        threshold = config.get("recognition", {}).get("similarity_threshold", 0.24)
        multi_face_policy = config.get("model", {}).get("multi_face_policy", "highest_confidence")

        detector = ModernFaceDetector(
            model_path=config.get("model", {}).get("yunet_model_path"),
            score_threshold=config.get("model", {}).get("score_threshold", 0.6),
            nms_threshold=config.get("model", {}).get("nms_threshold", 0.3)
        )
        aligned_size = tuple(config.get("model", {}).get("aligned_face_size", [112, 112]))
        aligner = FaceAligner(output_size=aligned_size)
        embedder = ArcFaceEmbedder(
            model_path=config.get("model", {}).get("arcface_model_path")
        )

        if os.path.exists(g_path):
            gallery = IdentityGallery.load(g_path)
        else:
            gallery = IdentityGallery()

        quality_cfg = config.get("quality", {})
        quality_assessor = None
        if quality_cfg.get("enabled", False):
            quality_assessor = FaceQualityAssessor.from_config(quality_cfg)

        return cls(
            detector=detector,
            aligner=aligner,
            embedder=embedder,
            gallery=gallery,
            threshold=threshold,
            multi_face_policy=multi_face_policy,
            quality_assessor=quality_assessor
        )

    def recognize(self, rgb_image: np.ndarray) -> ModernRecognitionResult:
        """
        Executes end-to-end recognition on a single image.

        Args:
            rgb_image (np.ndarray): Image array (H, W, 3) in RGB format.

        Returns:
            ModernRecognitionResult: Structured recognition result.
        """
        t0 = time.perf_counter()

        if rgb_image is None or rgb_image.size == 0 or not isinstance(rgb_image, np.ndarray):
            t_end = time.perf_counter()
            return ModernRecognitionResult(
                identity=None,
                best_candidate="Unknown",
                similarity=-1.0,
                threshold=self.threshold,
                recognized=False,
                latency_ms=(t_end - t0) * 1000.0,
                reason="invalid_image"
            )

        # 1. Detect faces
        faces: List[FaceDetection] = self.detector.detect_faces(rgb_image)
        if not faces:
            t_end = time.perf_counter()
            return ModernRecognitionResult(
                identity=None,
                best_candidate="Unknown",
                similarity=-1.0,
                threshold=self.threshold,
                recognized=False,
                latency_ms=(t_end - t0) * 1000.0,
                reason="no_face_detected"
            )

        # 2. Multi-face policy
        if len(faces) > 1 and self.multi_face_policy == "reject":
            t_end = time.perf_counter()
            return ModernRecognitionResult(
                identity=None,
                best_candidate="Unknown",
                similarity=-1.0,
                threshold=self.threshold,
                recognized=False,
                latency_ms=(t_end - t0) * 1000.0,
                reason="multiple_faces_rejected"
            )

        primary_face = max(faces, key=lambda d: d.confidence)

        # 3. Face Quality Assessment (Phase 8 FQA)
        quality_metrics = None
        if self.quality_assessor is not None and self.quality_assessor.enabled:
            quality_metrics = self.quality_assessor.assess(rgb_image, primary_face)
            if quality_metrics.quality_status == "poor":
                t_end = time.perf_counter()
                return ModernRecognitionResult(
                    identity=None,
                    best_candidate="Unknown",
                    similarity=-1.0,
                    threshold=self.threshold,
                    recognized=False,
                    bbox=primary_face.bbox,
                    landmarks=primary_face.landmarks,
                    model=self.embedder.model_name,
                    embedding_dim=self.embedder.embedding_dim,
                    latency_ms=(t_end - t0) * 1000.0,
                    reason=f"quality_rejected: {', '.join(quality_metrics.rejection_reasons)}",
                    quality=quality_metrics
                )

        # 4. 5-point alignment
        if primary_face.landmarks is None:
            t_end = time.perf_counter()
            return ModernRecognitionResult(
                identity=None,
                best_candidate="Unknown",
                similarity=-1.0,
                threshold=self.threshold,
                recognized=False,
                bbox=primary_face.bbox,
                latency_ms=(t_end - t0) * 1000.0,
                reason="missing_landmarks",
                quality=quality_metrics
            )

        aligned_crop = self.aligner.align(rgb_image, primary_face.landmarks)

        # 5. Extract ArcFace 512D normalized embedding
        emb = self.embedder.embed(aligned_crop)

        # 6. Query IdentityGallery
        rec_id, best_cand, best_sim, is_rec = self.gallery.search(
            query_embedding=emb,
            threshold=self.threshold
        )

        t_end = time.perf_counter()
        latency_ms = (t_end - t0) * 1000.0

        reason = "accepted" if is_rec else "below_threshold"

        return ModernRecognitionResult(
            identity=rec_id,
            best_candidate=best_cand,
            similarity=best_sim,
            threshold=self.threshold,
            recognized=is_rec,
            bbox=primary_face.bbox,
            landmarks=primary_face.landmarks,
            model=self.embedder.model_name,
            embedding_dim=self.embedder.embedding_dim,
            latency_ms=latency_ms,
            reason=reason,
            quality=quality_metrics
        )
