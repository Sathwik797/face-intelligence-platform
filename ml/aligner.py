from typing import Tuple, List, Optional
import numpy as np
import cv2

# Standard ArcFace canonical 5-point coordinates on 112x112 image space:
# 1. Left eye: (38.2946, 51.6963)
# 2. Right eye: (73.5318, 51.5014)
# 3. Nose tip: (56.0252, 71.7366)
# 4. Left mouth corner: (41.5493, 92.3655)
# 5. Right mouth corner: (70.7299, 92.2041)
DEFAULT_ARCFACE_CANONICAL_5_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class FaceAligner:
    """
    Performs 5-point facial landmark affine/similarity alignment
    to normalize scale, rotation, and translation into canonical face crops.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (112, 112),
        canonical_points: Optional[np.ndarray] = None,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: float = 0.0
    ):
        self.output_size = output_size
        self.border_mode = border_mode
        self.border_value = border_value

        if canonical_points is None:
            # Scale reference template if output_size differs from 112x112
            if output_size == (112, 112):
                self.canonical_points = DEFAULT_ARCFACE_CANONICAL_5_POINTS.copy()
            else:
                scale_x = output_size[0] / 112.0
                scale_y = output_size[1] / 112.0
                self.canonical_points = DEFAULT_ARCFACE_CANONICAL_5_POINTS.copy()
                self.canonical_points[:, 0] *= scale_x
                self.canonical_points[:, 1] *= scale_y
        else:
            self.canonical_points = np.asarray(canonical_points, dtype=np.float32)

    def align(self, rgb_image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Aligns a single face using 5 facial landmarks via 2D similarity transform.

        Args:
            rgb_image (np.ndarray): Full input image (H, W, 3) in RGB format.
            landmarks (np.ndarray): 5 landmark points with shape (5, 2) in image coordinates:
                                    [left_eye, right_eye, nose, left_mouth, right_mouth].

        Returns:
            np.ndarray: Aligned face crop with shape (output_size[1], output_size[0], 3).
        """
        if rgb_image is None or not isinstance(rgb_image, np.ndarray):
            raise ValueError("Input image must be a valid numpy array.")

        if landmarks is None or not isinstance(landmarks, np.ndarray):
            raise ValueError("Landmarks must be a valid numpy array.")

        src_pts = np.asarray(landmarks, dtype=np.float32)
        if src_pts.shape != (5, 2):
            raise ValueError(f"Expected landmarks shape (5, 2), got {src_pts.shape}.")

        # Estimate partial 2D affine (similarity) transform: translation, rotation, uniform scale
        transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, self.canonical_points)

        if transform_matrix is None:
            # Fallback to simple crop if landmarks are collinear/degenerate
            min_x = max(0, int(np.min(src_pts[:, 0])))
            max_x = min(rgb_image.shape[1], int(np.max(src_pts[:, 0])))
            min_y = max(0, int(np.min(src_pts[:, 1])))
            max_y = min(rgb_image.shape[0], int(np.max(src_pts[:, 1])))
            crop = rgb_image[min_y:max_y, min_x:max_x]
            return cv2.resize(crop, self.output_size)

        # Warp image into canonical alignment
        aligned_face = cv2.warpAffine(
            rgb_image,
            transform_matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=self.border_value
        )
        return aligned_face

    def align_batch(self, rgb_image: np.ndarray, landmark_list: List[np.ndarray]) -> List[np.ndarray]:
        """Aligns multiple faces from a list of 5-point landmarks."""
        return [self.align(rgb_image, lms) for lms in landmark_list if lms is not None]
