from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class BaseMatcher(ABC):
    """Abstract base class for face similarity matchers."""

    @abstractmethod
    def match(self, query_embedding: np.ndarray) -> Tuple[str, float, bool]:
        """
        Compare query embedding against enrolled reference gallery.

        Args:
            query_embedding (np.ndarray): 1D array representing face embedding vector.

        Returns:
            Tuple[str, float, bool]: (identity_name, distance_or_similarity, is_recognized)
        """
        pass


class EuclideanMatcher(BaseMatcher):
    """
    Phase 1 Baseline Matcher using Euclidean distance for 128D dlib embeddings.

    Distance formula:
        d(q, k) = ||q - k||_2 = sqrt(sum((q_i - k_i)^2))

    Decision rule:
        If min(d) <= threshold:
            identity = known_names[argmin(d)]
            is_recognized = True
        Else:
            identity = "Unknown"
            is_recognized = False
    """

    def __init__(
        self,
        known_encodings: Optional[List[np.ndarray]] = None,
        known_names: Optional[List[str]] = None,
        threshold: float = 0.6
    ):
        self.threshold = float(threshold)
        self.known_encodings = np.array(known_encodings, dtype=np.float64) if known_encodings is not None and len(known_encodings) > 0 else np.empty((0, 128))
        self.known_names = list(known_names) if known_names is not None else []

    def update_gallery(self, known_encodings: List[np.ndarray], known_names: List[str]):
        """Updates the enrolled gallery templates."""
        if len(known_encodings) != len(known_names):
            raise ValueError(f"Length mismatch: {len(known_encodings)} encodings vs {len(known_names)} names.")
        self.known_encodings = np.array(known_encodings, dtype=np.float64) if len(known_encodings) > 0 else np.empty((0, 128))
        self.known_names = list(known_names)

    def match(self, query_embedding: np.ndarray) -> Tuple[str, float, bool]:
        if query_embedding is None or len(query_embedding) == 0:
            return "Unknown", float("inf"), False

        if len(self.known_encodings) == 0 or len(self.known_names) == 0:
            return "Unknown", float("inf"), False

        query_vec = np.asarray(query_embedding, dtype=np.float64).reshape(1, -1)
        distances = np.linalg.norm(self.known_encodings - query_vec, axis=1)

        best_match_idx = int(np.argmin(distances))
        best_distance = float(distances[best_match_idx])

        if best_distance <= self.threshold:
            return self.known_names[best_match_idx], best_distance, True
        else:
            return "Unknown", best_distance, False


class CosineMatcher(BaseMatcher):
    """
    Modern Cosine Similarity Matcher for L2-normalized 512D ArcFace embeddings.

    Similarity formula (for unit length vectors ||q|| = 1, ||k|| = 1):
        S(q, k) = q · k = sum(q_i * k_i) in [-1.0, 1.0]

    Decision rule:
        If max(S) >= threshold:
            identity = known_names[argmax(S)]
            is_recognized = True
        Else:
            identity = "Unknown"
            is_recognized = False
    """

    def __init__(
        self,
        known_encodings: Optional[List[np.ndarray]] = None,
        known_names: Optional[List[str]] = None,
        threshold: float = 0.4  # Preliminary threshold for cosine similarity
    ):
        self.threshold = float(threshold)
        self.known_encodings = np.array(known_encodings, dtype=np.float32) if known_encodings is not None and len(known_encodings) > 0 else np.empty((0, 512), dtype=np.float32)
        self.known_names = list(known_names) if known_names is not None else []

    def update_gallery(self, known_encodings: List[np.ndarray], known_names: List[str]):
        """Updates enrolled gallery templates and ensures unit L2 normalization."""
        if len(known_encodings) != len(known_names):
            raise ValueError(f"Length mismatch: {len(known_encodings)} encodings vs {len(known_names)} names.")
        if len(known_encodings) > 0:
            enc_arr = np.array(known_encodings, dtype=np.float32)
            # Ensure 2D
            if enc_arr.ndim == 1:
                enc_arr = enc_arr.reshape(1, -1)
            # Normalize gallery
            norms = np.linalg.norm(enc_arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self.known_encodings = enc_arr / norms
        else:
            self.known_encodings = np.empty((0, 512), dtype=np.float32)
        self.known_names = list(known_names)

    def match(self, query_embedding: np.ndarray) -> Tuple[str, float, bool]:
        """
        Matches a single query embedding against the gallery using cosine similarity.

        Returns:
            Tuple[str, float, bool]: (identity_name, similarity_score, is_recognized)
        """
        if query_embedding is None or len(query_embedding) == 0:
            return "Unknown", -1.0, False

        if len(self.known_encodings) == 0 or len(self.known_names) == 0:
            return "Unknown", -1.0, False

        query_vec = np.asarray(query_embedding, dtype=np.float32).flatten()
        # Ensure query is unit normalized
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Compute cosine similarity across all gallery vectors: shape (N,)
        similarities = np.dot(self.known_encodings, query_vec)

        best_match_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_match_idx])

        if best_similarity >= self.threshold:
            return self.known_names[best_match_idx], best_similarity, True
        else:
            return "Unknown", best_similarity, False
