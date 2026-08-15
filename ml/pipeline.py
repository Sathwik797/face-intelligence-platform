import os
import pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ml.detector import BaseDetector, DlibHOGDetector
from ml.embedder import BaseEmbedder, DlibEmbedder
from ml.matcher import BaseMatcher, EuclideanMatcher

@dataclass
class RecognitionResult:
    """Structured result of a face recognition query."""
    identity: str
    distance: float
    location: Tuple[int, int, int, int]
    recognized: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert result dataclass to dictionary."""
        return {
            "identity": self.identity,
            "distance": round(self.distance, 4) if self.distance != float("inf") else None,
            "location": list(self.location),
            "recognized": self.recognized
        }


class FaceRecognitionPipeline:
    """
    Central orchestration pipeline for Face Recognition.

    Pipeline stages:
        1. RGB Image Input
        2. Face Detection (Bounding Boxes)
        3. Feature Embedding Extraction
        4. Gallery Comparison & Decision Logic
        5. Structured RecognitionResult Output
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
        """
        Factory method to initialize pipeline from config dictionary.
        """
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
        """
        Executes end-to-end face recognition on an RGB image.

        Args:
            rgb_image (np.ndarray): Image array in RGB format (H, W, 3).

        Returns:
            List[RecognitionResult]: List of recognition results for all detected faces.
        """
        if rgb_image is None or rgb_image.size == 0:
            return []

        # 1. Detect face locations
        face_locations = self.detector.detect(rgb_image)
        if not face_locations:
            return []

        # 2. Extract embeddings
        embeddings = self.embedder.embed(rgb_image, face_locations)

        # 3. Match each detected face against reference gallery
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
