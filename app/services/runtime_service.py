import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
import numpy as np

from ml.runtime import (
    FaceIntelligenceRuntime,
    RuntimeStatus,
    RuntimeConfig,
    RuntimeFrameResult
)
from ml.presence.schemas import PresenceSession, PresenceState, PresenceEvent


class RuntimeService:
    """
    Application-level singleton adapter providing thread-safe lifecycle control
    and frame processing access to FaceIntelligenceRuntime.
    """

    def __init__(self, runtime: FaceIntelligenceRuntime):
        self.runtime = runtime
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        gallery_path: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None
    ) -> "RuntimeService":
        """Factory creating a RuntimeService with an initialized FaceIntelligenceRuntime."""
        runtime = FaceIntelligenceRuntime.from_config(
            config=config,
            gallery_path=gallery_path,
            clock=clock
        )
        return cls(runtime=runtime)

    @property
    def status(self) -> RuntimeStatus:
        return self.runtime.status

    def start(self):
        with self._lock:
            self.runtime.start()

    def stop(self, reason: str = "runtime_shutdown") -> List[PresenceEvent]:
        with self._lock:
            return self.runtime.stop(reason=reason)

    def reset(self):
        with self._lock:
            self.runtime.reset()

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        timestamp: Optional[datetime] = None
    ) -> RuntimeFrameResult:
        with self._lock:
            if self.runtime.status != RuntimeStatus.RUNNING:
                # Automatically start runtime if not yet started
                self.runtime.start()
            return self.runtime.process_frame(rgb_frame, timestamp=timestamp)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            active_sessions = self.runtime.presence_manager.get_active_sessions()
            history_sessions = self.runtime.presence_manager.get_session_history()
            return {
                "status": self.runtime.status.value,
                "frame_counter": self.runtime.frame_counter,
                "active_sessions_count": len(active_sessions),
                "archived_sessions_count": len(history_sessions),
                "buffered_history_count": len(self.runtime.buffered_results)
            }

    def get_active_sessions(self) -> List[PresenceSession]:
        with self._lock:
            return self.runtime.presence_manager.get_active_sessions()

    def get_session_history(self) -> List[PresenceSession]:
        with self._lock:
            return self.runtime.presence_manager.get_session_history()

    def get_identity_state(self, identity: str) -> PresenceState:
        with self._lock:
            return self.runtime.presence_manager.get_presence_state(identity)
