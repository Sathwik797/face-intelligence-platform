from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable

from ml.presence.schemas import (
    PresenceState,
    PresenceEventType,
    PresenceMode,
    PresenceConfig,
    PresenceEvent,
    PresenceSession
)
from ml.presence.state_machine import IdentityPresenceStateMachine, PRESENCE_PRESETS
from ml.temporal.schemas import TemporalRecognitionResult


class PresenceManager:
    """
    Multi-identity presence and session orchestrator.

    Orchestrates individual IdentityPresenceStateMachine instances for each tracked identity,
    processes Phase 9 TemporalRecognitionResult observations, manages the lifecycle of
    in-memory PresenceSession instances, and emits structured presence events.
    """

    def __init__(
        self,
        config: Optional[PresenceConfig] = None,
        clock: Optional[Callable[[], datetime]] = None
    ):
        self.config = config or PRESENCE_PRESETS[PresenceMode.BALANCED]
        self.clock = clock or (lambda: datetime.now(timezone.utc))

        self.trackers: Dict[str, IdentityPresenceStateMachine] = {}
        self.session_history: List[PresenceSession] = []

    @classmethod
    def from_config(
        cls,
        presence_config: Dict[str, Any],
        clock: Optional[Callable[[], datetime]] = None
    ) -> "PresenceManager":
        """Factory method constructing PresenceManager from configuration dictionary."""
        enabled = presence_config.get("enabled", True)
        mode_str = presence_config.get("mode", "balanced").lower()

        try:
            mode = PresenceMode(mode_str)
        except ValueError:
            mode = PresenceMode.BALANCED

        base = PRESENCE_PRESETS.get(mode, PRESENCE_PRESETS[PresenceMode.BALANCED])

        custom_cfg = PresenceConfig(
            enabled=enabled,
            mode=mode.value,
            min_entry_observations=int(presence_config.get("min_entry_observations", base.min_entry_observations)),
            entry_window_seconds=float(presence_config.get("entry_window_seconds", base.entry_window_seconds)),
            grace_period_seconds=float(presence_config.get("grace_period_seconds", base.grace_period_seconds)),
            max_session_duration_seconds=float(presence_config.get("max_session_duration_seconds", base.max_session_duration_seconds))
        )

        return cls(config=custom_cfg, clock=clock)

    def _get_or_create_tracker(self, identity: str) -> IdentityPresenceStateMachine:
        if identity not in self.trackers:
            self.trackers[identity] = IdentityPresenceStateMachine(
                identity=identity,
                config=self.config,
                clock=self.clock
            )
        return self.trackers[identity]

    def update(
        self,
        temporal_result: TemporalRecognitionResult,
        timestamp: Optional[datetime] = None
    ) -> List[PresenceEvent]:
        """
        Updates multi-identity presence state given a Phase 9 TemporalRecognitionResult observation.

        Args:
            temporal_result (TemporalRecognitionResult): Phase 9 stabilized output.
            timestamp (Optional[datetime]): Timezone-aware timestamp (or injected clock).

        Returns:
            List[PresenceEvent]: List of emitted presence events.
        """
        if not self.config.enabled:
            return []

        active_id = temporal_result.stable_identity if temporal_result.is_stable else None
        events: List[PresenceEvent] = []

        if active_id:
            # 1. Update the recognized active identity with is_stable=True
            tracker = self._get_or_create_tracker(active_id)
            evs = tracker.process_observation(is_stable=True, timestamp=timestamp)
            events.extend(evs)
            self._check_archived_sessions(tracker, evs)

            # 2. Inform other active/candidate trackers that they were not observed in this frame
            for identity, trk in list(self.trackers.items()):
                if identity != active_id and trk.state in (PresenceState.PRESENT, PresenceState.CANDIDATE, PresenceState.GRACE):
                    other_evs = trk.process_observation(is_stable=False, timestamp=timestamp)
                    events.extend(other_evs)
                    self._check_archived_sessions(trk, other_evs)
        else:
            # 3. No stable identity in this frame (Unknown / Unstable) -> notify all active trackers
            for identity, trk in list(self.trackers.items()):
                if trk.state in (PresenceState.PRESENT, PresenceState.CANDIDATE, PresenceState.GRACE):
                    trk_evs = trk.process_observation(is_stable=False, timestamp=timestamp)
                    events.extend(trk_evs)
                    self._check_archived_sessions(trk, trk_evs)

        return events

    def tick(self, timestamp: Optional[datetime] = None) -> List[PresenceEvent]:
        """Periodic heartbeat sweep to detect grace timeouts on inactive identities."""
        if not self.config.enabled:
            return []

        events: List[PresenceEvent] = []
        for identity, trk in list(self.trackers.items()):
            if trk.state == PresenceState.GRACE:
                timeout_evs = trk.check_grace_timeout(timestamp=timestamp)
                events.extend(timeout_evs)
                self._check_archived_sessions(trk, timeout_evs)

        return events

    def _check_archived_sessions(self, tracker: IdentityPresenceStateMachine, events: List[PresenceEvent]):
        for ev in events:
            if ev.event_type == PresenceEventType.SESSION_ENDED and tracker.active_session is not None:
                if tracker.active_session not in self.session_history:
                    self.session_history.append(tracker.active_session)

    def get_active_sessions(self) -> List[PresenceSession]:
        """Returns list of all currently active (PRESENT or GRACE) sessions."""
        return [
            trk.active_session for trk in self.trackers.values()
            if trk.active_session is not None and trk.active_session.is_active
        ]

    def get_session_history(self) -> List[PresenceSession]:
        """Returns all completed closed sessions."""
        return list(self.session_history)

    def get_presence_state(self, identity: str) -> PresenceState:
        """Returns current presence state of a given identity."""
        if identity in self.trackers:
            return self.trackers[identity].state
        return PresenceState.NOT_PRESENT

    def reset(self):
        """Resets all presence state machines and clears session history."""
        self.trackers.clear()
        self.session_history.clear()
