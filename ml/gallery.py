import os
import json
import datetime
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

class IdentityGallery:
    """
    Gallery repository for enrolled identity face embeddings.
    Supports multi-template enrollment per identity and exact cosine similarity search.
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        identities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.embeddings = (
            np.asarray(embeddings, dtype=np.float32)
            if embeddings is not None and len(embeddings) > 0
            else np.empty((0, 512), dtype=np.float32)
        )
        self.identities = list(identities) if identities is not None else []
        self.metadata = dict(metadata) if metadata is not None else {
            "created_at": datetime.datetime.now().isoformat(),
            "model_name": "arcface_resnet50_512d",
            "embedding_dim": 512,
            "source": "LFW_enrollment_split",
            "template_count": len(self.identities),
            "unique_identities": len(set(self.identities))
        }

        if len(self.embeddings) != len(self.identities):
            raise ValueError(f"Mismatch between embeddings count ({len(self.embeddings)}) and identities count ({len(self.identities)}).")

    @property
    def total_templates(self) -> int:
        return len(self.identities)

    @property
    def unique_identities(self) -> List[str]:
        return sorted(list(set(self.identities)))

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1] if self.embeddings.ndim == 2 and self.embeddings.shape[0] > 0 else 512

    def get_identity_template_count(self, identity: str) -> int:
        """Returns the number of enrolled templates for the specified identity."""
        return sum(1 for ident in self.identities if ident == identity)

    def add_templates(
        self,
        identity: str,
        embeddings: np.ndarray,
        source_paths: Optional[List[str]] = None
    ):
        """Adds one or more normalized embedding vectors for an enrolled identity."""
        embs = np.asarray(embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        if embs.shape[1] != self.embedding_dim and self.total_templates > 0:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embs.shape[1]}")

        # Ensure unit L2 normalization
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embs_norm = embs / norms

        if self.embeddings.shape[0] == 0:
            self.embeddings = embs_norm
        else:
            self.embeddings = np.vstack([self.embeddings, embs_norm])

        for _ in range(len(embs_norm)):
            self.identities.append(identity)

        self.metadata["template_count"] = len(self.identities)
        self.metadata["unique_identities"] = len(set(self.identities))
        if source_paths:
            if "sources" not in self.metadata:
                self.metadata["sources"] = []
            self.metadata["sources"].extend(source_paths)

    def remove_identity(self, identity: str) -> int:
        """
        Removes all templates associated with an identity.
        Returns the number of templates removed.
        """
        indices_to_keep = [i for i, ident in enumerate(self.identities) if ident != identity]
        removed_count = len(self.identities) - len(indices_to_keep)
        if removed_count == 0:
            return 0

        if len(indices_to_keep) == 0:
            self.embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
            self.identities = []
        else:
            self.embeddings = self.embeddings[indices_to_keep]
            self.identities = [self.identities[i] for i in indices_to_keep]

        self.metadata["template_count"] = len(self.identities)
        self.metadata["unique_identities"] = len(set(self.identities))
        return removed_count

    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.45
    ) -> Tuple[Optional[str], str, float, bool]:
        """
        Queries the gallery using multi-template cosine similarity.

        Algorithm:
            1. Compute cosine similarity against all gallery templates: S_i = q · k_i
            2. For each unique identity, calculate identity-level score: max_{j in id} S_j
            3. Select best candidate: argmax_{id} (max S_j)
            4. Open-set decision: if best_score >= threshold -> recognized; else -> Unknown (None)

        Returns:
            Tuple[Optional[str], str, float, bool]:
                - recognized_identity: name if recognized, None if rejected as Unknown
                - best_candidate: name of the highest-matching gallery candidate
                - best_similarity: continuous cosine similarity score [-1.0, 1.0]
                - is_recognized: True if best_similarity >= threshold
        """
        if query_embedding is None or len(query_embedding) == 0 or self.total_templates == 0:
            return None, "Unknown", -1.0, False

        query_vec = np.asarray(query_embedding, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Compute dot product across all enrolled template vectors: shape (M,)
        raw_similarities = np.dot(self.embeddings, query_vec)

        # Aggregate max similarity per identity
        identity_scores: Dict[str, float] = {}
        for sim, ident in zip(raw_similarities, self.identities):
            sim_val = float(sim)
            if ident not in identity_scores or sim_val > identity_scores[ident]:
                identity_scores[ident] = sim_val

        # Find best matching candidate identity
        best_candidate = max(identity_scores, key=lambda k: identity_scores[k])
        best_similarity = float(identity_scores[best_candidate])

        if best_similarity >= threshold:
            return best_candidate, best_candidate, best_similarity, True
        else:
            return None, best_candidate, best_similarity, False

    def validate(self) -> Dict[str, Any]:
        """Validates gallery structural and numerical integrity."""
        errors = []
        if len(self.embeddings) != len(self.identities):
            errors.append(f"Length mismatch: {len(self.embeddings)} embeddings vs {len(self.identities)} identities.")

        if self.total_templates > 0:
            if not np.all(np.isfinite(self.embeddings)):
                errors.append("Embeddings contain non-finite values (NaN or Inf).")

            norms = np.linalg.norm(self.embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                errors.append(f"Embeddings not unit normalized: min norm = {np.min(norms):.4f}, max norm = {np.max(norms):.4f}")

            if self.embeddings.shape[1] != 512:
                errors.append(f"Unexpected embedding dimension: {self.embeddings.shape[1]} (expected 512).")

        return {
            "valid": len(errors) == 0,
            "total_templates": self.total_templates,
            "unique_identities": len(self.unique_identities),
            "embedding_dim": self.embedding_dim,
            "errors": errors
        }

    def save(self, filepath: str):
        """Serializes gallery into a compressed NPZ archive."""
        val = self.validate()
        if not val["valid"]:
            raise ValueError(f"Cannot save invalid gallery: {val['errors']}")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            embeddings=self.embeddings,
            identities=np.array(self.identities, dtype=object),
            metadata_json=json.dumps(self.metadata)
        )

    @classmethod
    def load(cls, filepath: str) -> "IdentityGallery":
        """Loads and validates an IdentityGallery from a serialized NPZ archive."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Gallery file not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        embeddings = data["embeddings"]
        identities = list(data["identities"])
        metadata = json.loads(str(data["metadata_json"]))

        gallery = cls(embeddings=embeddings, identities=identities, metadata=metadata)
        val = gallery.validate()
        if not val["valid"]:
            raise ValueError(f"Loaded gallery failed validation: {val['errors']}")

        return gallery
