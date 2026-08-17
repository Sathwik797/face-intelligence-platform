import math
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2


def extract_face_crop_gray(
    rgb_image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Extracts the cropped face region and converts to grayscale.

    Args:
        rgb_image (np.ndarray): Input RGB image array (H, W, 3).
        bbox (Tuple[int, int, int, int]): Bounding box in CSS format (top, right, bottom, left).

    Returns:
        np.ndarray: Grayscale face crop array (h, w).
    """
    h_img, w_img = rgb_image.shape[:2]
    top, right, bottom, left = bbox

    # Clamp coordinates to image boundaries
    top = max(0, min(top, h_img - 1))
    bottom = max(top + 1, min(bottom, h_img))
    left = max(0, min(left, w_img - 1))
    right = max(left + 1, min(right, w_img))

    crop_rgb = rgb_image[top:bottom, left:right]
    if crop_rgb.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)


def compute_face_dimensions(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int, ...]
) -> Tuple[int, int, int, float]:
    """
    Calculates face bounding box width, height, area, and area ratio.

    Returns:
        Tuple[int, int, int, float]: (width, height, area, area_ratio)
    """
    h_img, w_img = image_shape[:2]
    top, right, bottom, left = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    area = width * height
    total_img_area = max(1, h_img * w_img)
    area_ratio = float(area / total_img_area)
    return width, height, area, area_ratio


def compute_blur_score(face_crop_gray: np.ndarray) -> float:
    """
    Measures image sharpness using the Variance of Laplacian.
    Higher values indicate a sharper, less blurry image.
    """
    if face_crop_gray is None or face_crop_gray.size < 4:
        return 0.0
    laplacian = cv2.Laplacian(face_crop_gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_brightness_score(face_crop_gray: np.ndarray) -> float:
    """
    Calculates the mean grayscale intensity in [0, 255].
    """
    if face_crop_gray is None or face_crop_gray.size == 0:
        return 0.0
    return float(np.mean(face_crop_gray))


def compute_contrast_score(face_crop_gray: np.ndarray) -> float:
    """
    Calculates the standard deviation of grayscale intensity in [0, 255].
    """
    if face_crop_gray is None or face_crop_gray.size == 0:
        return 0.0
    return float(np.std(face_crop_gray))


def compute_alignment_quality(
    landmarks: Optional[np.ndarray],
    bbox: Tuple[int, int, int, int]
) -> float:
    """
    Assesses landmark geometric plausibility.
    Checks:
    1. Landmark availability (5 points).
    2. Eye tilt angle (roll deviation from horizontal).
    3. Vertical ordering sanity (eyes above nose above mouth).
    4. Landmark bounds containment.

    Returns:
        float: Alignment quality score in [0.0, 1.0].
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0

    top, right, bottom, left = bbox
    w_box = max(1, right - left)
    h_box = max(1, bottom - top)

    pts = np.asarray(landmarks, dtype=np.float32)
    le, re, nose, lm, rm = pts[0], pts[1], pts[2], pts[3], pts[4]

    # 1. Roll angle penalty
    dx = re[0] - le[0]
    dy = re[1] - le[1]
    eye_dist = math.hypot(dx, dy)
    if eye_dist < 2.0:
        return 0.1

    roll_rad = math.atan2(abs(dy), max(abs(dx), 1e-5))
    roll_deg = math.degrees(roll_rad)
    # Roll score: 1.0 at 0 deg, drops to 0.0 at 45 deg
    roll_score = max(0.0, 1.0 - (roll_deg / 45.0))

    # 2. Vertical structure penalty (eyes.y < nose.y < mouth.y)
    eyes_y = (le[1] + re[1]) / 2.0
    mouth_y = (lm[1] + rm[1]) / 2.0
    vert_score = 1.0
    if not (eyes_y < nose[1] < mouth_y):
        vert_score = 0.3
    else:
        # Check inter-feature vertical proportions
        eye_nose_dist = nose[1] - eyes_y
        nose_mouth_dist = mouth_y - nose[1]
        if eye_nose_dist <= 0 or nose_mouth_dist <= 0:
            vert_score = 0.3

    # 3. Containment check (landmarks within slightly padded bbox)
    pad_x = w_box * 0.25
    pad_y = h_box * 0.25
    x_min, x_max = left - pad_x, right + pad_x
    y_min, y_max = top - pad_y, bottom + pad_y
    inside = np.all((pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max))
    containment_score = 1.0 if inside else 0.4

    alignment_score = 0.5 * roll_score + 0.3 * vert_score + 0.2 * containment_score
    return float(np.clip(alignment_score, 0.0, 1.0))


def compute_pose_quality(landmarks: Optional[np.ndarray]) -> float:
    """
    Computes a frontal symmetry proxy from 5-point landmarks.
    Evaluates the horizontal symmetry of the nose relative to the left and right eyes.
    A frontal face yields a symmetry ratio close to 1.0; severe yaw yaw reduces it toward 0.0.

    Returns:
        float: Pose quality / frontal symmetry score in [0.0, 1.0].
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0

    pts = np.asarray(landmarks, dtype=np.float32)
    le, re, nose = pts[0], pts[1], pts[2]

    d_left = math.hypot(nose[0] - le[0], nose[1] - le[1])
    d_right = math.hypot(nose[0] - re[0], nose[1] - re[1])

    if d_left <= 1e-5 or d_right <= 1e-5:
        return 0.1

    ratio = min(d_left, d_right) / max(d_left, d_right)
    return float(np.clip(ratio, 0.0, 1.0))


def compute_composite_quality_score(
    blur_score: float,
    brightness: float,
    contrast: float,
    detection_conf: float,
    alignment_qual: float,
    pose_qual: float,
    min_blur: float = 40.0,
    target_blur: float = 250.0
) -> float:
    """
    Computes a normalized overall composite quality index in [0.0, 1.0].
    Combines individual factors with documented, empirical weights.
    """
    # 1. Blur normalization (log-sigmoid response)
    s_blur = float(np.clip((blur_score - min_blur) / max(1.0, (target_blur - min_blur)), 0.0, 1.0))

    # 2. Illumination score (optimal at mid-range ~128)
    s_bright = float(max(0.0, 1.0 - (abs(brightness - 128.0) / 128.0)))

    # 3. Contrast score (normalized up to 60.0 std)
    s_contrast = float(np.clip(contrast / 60.0, 0.0, 1.0))

    # 4. Detection confidence
    s_conf = float(np.clip(detection_conf, 0.0, 1.0))

    # 5. Alignment and Pose
    s_align = float(np.clip(alignment_qual, 0.0, 1.0))
    s_pose = float(np.clip(pose_qual, 0.0, 1.0))

    # Weighted linear combination
    composite = (
        0.30 * s_blur +
        0.20 * s_conf +
        0.15 * s_align +
        0.15 * s_pose +
        0.10 * s_bright +
        0.10 * s_contrast
    )
    return float(np.clip(composite, 0.0, 1.0))
