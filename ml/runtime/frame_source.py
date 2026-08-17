import abc
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Iterator, List, Union
import numpy as np


class BaseFrameSource(abc.ABC):
    """
    Abstract Base Class for video and image frame sources.
    Provides standard iterator interface and frame capture methods.
    """

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, datetime]]:
        return self

    @abc.abstractmethod
    def __next__(self) -> Tuple[int, np.ndarray, datetime]:
        """Yields next (frame_index, rgb_frame, timestamp) or raises StopIteration."""
        pass

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        """Reads a single frame. Returns (success, rgb_image, timestamp)."""
        pass

    @abc.abstractmethod
    def release(self):
        """Releases underlying hardware or memory resources."""
        pass


class StaticFrameSource(BaseFrameSource):
    """
    Frame source that replays a pre-loaded list or sequence of numpy RGB image arrays.
    Ideal for deterministic unit and integration tests.
    """

    def __init__(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        start_time: Optional[datetime] = None
    ):
        self.frames = frames
        self.fps = float(fps)
        self.frame_interval = 1.0 / max(0.1, self.fps)
        self.current_idx = 0
        self.start_time = start_time or datetime.now(timezone.utc)

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        if self.current_idx >= len(self.frames):
            return False, None, None

        frame = self.frames[self.current_idx]
        timestamp = self.start_time + timedelta(seconds=self.current_idx * self.frame_interval)
        self.current_idx += 1
        return True, frame, timestamp

    def __next__(self) -> Tuple[int, np.ndarray, datetime]:
        idx = self.current_idx
        success, frame, ts = self.read()
        if not success or frame is None or ts is None:
            raise StopIteration
        return idx, frame, ts

    def release(self):
        self.frames = []
        self.current_idx = 0


class SyntheticFrameSource(BaseFrameSource):
    """
    Generates synthetic deterministic RGB image arrays with configurable resolution and counts.
    Allows zero-dependency offline pipeline testing.
    """

    def __init__(
        self,
        max_frames: int = 50,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        start_time: Optional[datetime] = None
    ):
        self.max_frames = int(max_frames)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frame_interval = 1.0 / max(0.1, self.fps)
        self.current_idx = 0
        self.start_time = start_time or datetime.now(timezone.utc)

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        if self.current_idx >= self.max_frames:
            return False, None, None

        # Create deterministic synthetic test image (H, W, 3)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[50:150, 50:150, :] = (self.current_idx * 5) % 255  # Dynamic pixel variance
        timestamp = self.start_time + timedelta(seconds=self.current_idx * self.frame_interval)
        self.current_idx += 1
        return True, img, timestamp

    def __next__(self) -> Tuple[int, np.ndarray, datetime]:
        idx = self.current_idx
        success, frame, ts = self.read()
        if not success or frame is None or ts is None:
            raise StopIteration
        return idx, frame, ts

    def release(self):
        self.current_idx = self.max_frames


class OpenCVFrameSource(BaseFrameSource):
    """
    Optional OpenCV VideoCapture frame source for physical webcams or video files.
    Safely handles OpenCV availability and device connection failures.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None
    ):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.current_idx = 0
        self._init_capture()

    def _init_capture(self):
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.source)
            if self.width and self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            self.cap = None

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[datetime]]:
        if self.cap is None or not self.cap.isOpened():
            return False, None, None

        import cv2
        ret, bgr_frame = self.cap.read()
        if not ret or bgr_frame is None:
            return False, None, None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        timestamp = datetime.now(timezone.utc)
        self.current_idx += 1
        return True, rgb_frame, timestamp

    def __next__(self) -> Tuple[int, np.ndarray, datetime]:
        idx = self.current_idx
        success, frame, ts = self.read()
        if not success or frame is None or ts is None:
            raise StopIteration
        return idx, frame, ts

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
