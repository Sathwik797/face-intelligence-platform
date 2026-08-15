import os
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import numpy as np
import onnxruntime as ort
import face_recognition

class BaseEmbedder(ABC):
    """Abstract base class for face embedding extractors."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the face embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the name/identifier of the embedding model."""
        pass

    @abstractmethod
    def embed(
        self,
        image_or_crop: np.ndarray,
        face_locations: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        """
        Extract normalized face embedding(s).

        Args:
            image_or_crop (np.ndarray): Full RGB image (if detector-based) or aligned face crop (H, W, 3).
            face_locations (Optional[List[Tuple[int, int, int, int]]]): Face bounding boxes (for dlib).

        Returns:
            np.ndarray: Embedding matrix with shape (N, embedding_dim) and unit L2 norm.
        """
        pass

    def embed_batch(self, aligned_crops: List[np.ndarray]) -> np.ndarray:
        """Extracts embeddings for a batch of aligned face crops."""
        if not aligned_crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack([self.embed(crop) for crop in aligned_crops if crop is not None])


class DlibEmbedder(BaseEmbedder):
    """
    Phase 1 Baseline Face Embedder using dlib's pretrained ResNet-34 model (128D).

    Characteristics:
        - Pretrained on ~3 million images (Davis King / dlib model v1).
        - Generates 128-dimensional floating point embeddings.
        - Preserved as Experiment E1 baseline.
    """

    def __init__(self, num_jitters: int = 1, model: str = "large"):
        self.num_jitters = num_jitters
        self.model = model
        self._dim = 128

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "dlib_resnet34_128d"

    def embed(
        self,
        image_or_crop: np.ndarray,
        face_locations: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        if image_or_crop is None or not isinstance(image_or_crop, np.ndarray):
            raise ValueError("Input image must be a valid numpy array.")

        if face_locations is None:
            # If no bounding box supplied, assume the entire input is the face crop
            h, w = image_or_crop.shape[:2]
            face_locations = [(0, w, h, 0)]

        if not face_locations or image_or_crop.size == 0:
            return np.empty((0, self._dim), dtype=np.float64)

        encodings = face_recognition.face_encodings(
            image_or_crop,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters,
            model=self.model
        )

        if not encodings:
            return np.empty((0, self._dim), dtype=np.float64)

        return np.array(encodings, dtype=np.float64)


class ArcFaceEmbedder(BaseEmbedder):
    """
    Modern Pretrained ArcFace Face Embedder (512D).

    Model Specifications:
        - Architecture: ResNet-50 backbone with Additive Angular Margin Loss (ArcFace).
        - Source: InsightFace (buffalo_l / w600k_r50 ONNX model).
        - Input: Aligned (112, 112, 3) RGB face image.
        - Preprocessing: Standard pixel normalization (x - 127.5) / 127.5, transposed to (C, H, W).
        - Output: 512-dimensional vector with unit L2 normalization ||e||_2 = 1.0.
        - Execution: Optimized ONNX Runtime on CPU.
    """

    DEFAULT_MODEL_URL = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
    DEFAULT_MODEL_PATH = "ml/models/arcface_w600k_r50.onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        providers: Optional[List[str]] = None
    ):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self._dim = 512
        self.providers = providers or ["CPUExecutionProvider"]

        self._ensure_model_exists()
        
        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(self.model_path, sess_options, providers=self.providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def _ensure_model_exists(self):
        """Ensures ArcFace ONNX weights are available locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            print(f"[INFO] Downloading pretrained ArcFace ResNet-50 ONNX model to {self.model_path}...")
            urllib.request.urlretrieve(self.DEFAULT_MODEL_URL, self.model_path)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "arcface_resnet50_512d"

    def _preprocess_crop(self, rgb_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses a single 112x112 RGB face crop.
        Transforms: uint8 [0, 255] -> float32 [-1.0, 1.0] and (H, W, C) -> (C, H, W).
        """
        if rgb_crop is None or not isinstance(rgb_crop, np.ndarray):
            raise ValueError("Input aligned crop must be a valid numpy array.")

        if rgb_crop.ndim != 3 or rgb_crop.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB crop (H, W, 3), got shape {rgb_crop.shape}.")

        if rgb_crop.shape[0] != 112 or rgb_crop.shape[1] != 112:
            rgb_crop = cv2.resize(rgb_crop, (112, 112))

        # Normalize pixels to [-1.0, 1.0]
        blob = (rgb_crop.astype(np.float32) - 127.5) / 127.5
        # Transpose from (H, W, C) to (C, H, W)
        blob = np.transpose(blob, (2, 0, 1))
        return blob

    def embed(
        self,
        image_or_crop: np.ndarray,
        face_locations: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        """
        Extracts 512D L2-normalized embedding for an aligned face crop.

        Args:
            image_or_crop (np.ndarray): Aligned (112, 112, 3) RGB face image.
            face_locations (Optional): Ignored for pre-aligned crops.

        Returns:
            np.ndarray: Matrix of shape (1, 512) with unit L2 norm.
        """
        if image_or_crop is None or not isinstance(image_or_crop, np.ndarray):
            raise ValueError("Input aligned crop must be a valid numpy array.")

        if image_or_crop.size == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        blob = self._preprocess_crop(image_or_crop)
        # Add batch dimension: (1, 3, 112, 112)
        blob_batch = np.expand_dims(blob, axis=0)

        raw_embedding = self._session.run([self._output_name], {self._input_name: blob_batch})[0]
        
        # Apply L2 normalization: e_norm = e / max(||e||_2, 1e-10)
        norm = np.linalg.norm(raw_embedding, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        normalized_embedding = raw_embedding / norm
        return normalized_embedding.astype(np.float32)

    def embed_batch(self, aligned_crops: List[np.ndarray]) -> np.ndarray:
        """
        Extracts 512D L2-normalized embeddings for a list of aligned face crops in a single batch.

        Args:
            aligned_crops (List[np.ndarray]): List of (112, 112, 3) RGB face images.

        Returns:
            np.ndarray: Matrix of shape (N, 512) with unit L2 norm.
        """
        if not aligned_crops:
            return np.empty((0, self._dim), dtype=np.float32)

        blobs = [self._preprocess_crop(crop) for crop in aligned_crops if crop is not None and crop.size > 0]
        if not blobs:
            return np.empty((0, self._dim), dtype=np.float32)

        # Stack into tensor (N, 3, 112, 112)
        batch_tensor = np.stack(blobs, axis=0)

        raw_embeddings = self._session.run([self._output_name], {self._input_name: batch_tensor})[0]

        # Apply L2 normalization
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized_embeddings = raw_embeddings / norms
        return normalized_embeddings.astype(np.float32)
