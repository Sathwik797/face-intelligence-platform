import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable

from ml.presence.schemas import (
    PresenceState,
    PresenceEventType,
    PresenceMode,
    PresenceConfig,
    PresenceEvent,
    PresenceSession
)

PRESENCE_PRESETS: Dict[PresenceMode, PresenceConfig] = {
    PresenceMode.FAST: PresenceConfig(
        mode="fast",
        min_entry_observations=2,
        entry_window_seconds=3.0,
        grace_period_seconds=5.0,
        max_session_duration_seconds=28800.0
    ),
    PresenceMode.BALANCED: PresenceConfig(
        mode="balanced",
        min_entry_observations=3,
        entry_window_seconds=5.0,
        grace_period_seconds=10.0,
        max_session_duration_seconds=28800.0
    ),
    PresenceMode.STRICT: PresenceConfig(
        mode="strict",
        min_entry_observations=5,
        entry_window_seconds=8.0,
        grace_period_seconds=20.0,
        max_session_duration_seconds=28800.0
    )
}


class IdentityPresenceStateMachine:
    """
    Explicit finite state machine governing the presence and session lifecycle for a single identity.

    State Transitions:
        NOT_PRESENT -> (stable observation) -> CANDIDATE
        CANDIDATE   -> (N >= min_entry_observations within window) -> PRESENT (creates PresenceSession)
        CANDIDATE   -> (entry window timeout) -> NOT_PRESENT
        PRESENT     -> (stable observation) -> PRESENT (updates last_seen_at)
        PRESENT     -> (observation missing / Unknown) -> GRACE (starts grace timer)
        GRACE       -> (stable identity returns before timeout) -> PRESENT (resumes session)
        GRACE       -> (grace period timeout expired) -> ABSENT (finalizes PresenceSession)
        PRESENT     -> (duration >= max_session_duration) -> ABSENT (duration safeguard)
        ABSENT      -> (stable observation) -> CANDIDATE
    """

    def __init__(
        self,
        identity: str,
        config: Optional[PresenceConfig] = None,
        clock: Optional[Callable[[], datetime]] = None
    ):
        self.identity = identity
        self.config = config or PRESENCE_PRESETS[PresenceMode.BALANCED]
        self.clock = clock or (lambda: datetime.now(timezone.utc))

        self.state: PresenceState = PresenceState.NOT_PRESENT
        self.active_session: Optional[PresenceSession] = None
        self.entry_timestamps: List[datetime] = []
        self.grace_start_time: Optional[datetime] = None

    def _now(self, timestamp: Optional[datetime] = None) -> datetime:
        if timestamp is not None:
            return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        return self.clock()

    def process_observation(
        self,
        is_stable: bool,
        timestamp: Optional[datetime] = None
    ) -> List[PresenceEvent]:
        """
        Processes an identity presence observation at a given timestamp.

        Args:
            is_stable (bool): True if Phase 9 temporal stabilizer confirmed this identity.
            timestamp (Optional[datetime]): Timezone-aware timestamp (or injected clock).

        Returns:
            List[PresenceEvent]: List of emitted state change events.
        """
        now = self._now(timestamp)
        events: List[PresenceEvent] = []

        # 1. Max session duration safeguard check
        if self.state in (PresenceState.PRESENT, PresenceState.GRACE) and self.active_session is not None:
            if self.active_session.compute_duration(now) >= self.config.max_session_duration_seconds:
                events.extend(self._close_session(
                    now=now,
                    reason=f"maximum session duration safeguard reached ({self.config.max_session_duration_seconds}s)"
                ))
                return events

        # 2. State Machine Transition Dispatch
        if self.state in (PresenceState.NOT_PRESENT, PresenceState.ABSENT):
            if is_stable:
                prev = self.state
                self.state = PresenceState.CANDIDATE
                self.entry_timestamps = [now]
                events.append(PresenceEvent(
                    event_type=PresenceEventType.STATE_CHANGED,
                    identity=self.identity,
                    timestamp=now,
                    previous_state=prev,
                    new_state=PresenceState.CANDIDATE,
                    reason="first stable identity observation recorded; entering candidate confirmation"
                ))

        elif self.state == PresenceState.CANDIDATE:
            if is_stable:
                self.entry_timestamps.append(now)
                # Purge timestamps older than entry window
                cutoff_time = (now.timestamp() - self.config.entry_window_seconds)
                self.entry_timestamps = [t for t in self.entry_timestamps if t.timestamp() >= cutoff_time]

                if len(self.entry_timestamps) >= self.config.min_entry_observations:
                    # Entry confirmed -> Transition to PRESENT and create new session
                    prev = self.state
                    self.state = PresenceState.PRESENT
                    self.active_session = PresenceSession(
                        session_id=str(uuid.uuid4()),
                        identity=self.identity,
                        started_at=now,
                        last_seen_at=now,
                        ended_at=None,
                        state=PresenceState.PRESENT,
                        duration_seconds=0.0,
                        observation_count=len(self.entry_timestamps),
                        interruption_count=0
                    )
                    self.entry_timestamps.clear()
                    events.append(PresenceEvent(
                        event_type=PresenceEventType.ENTRY_CONFIRMED,
                        identity=self.identity,
                        timestamp=now,
                        previous_state=prev,
                        new_state=PresenceState.PRESENT,
                        session_id=self.active_session.session_id,
                        reason=f"stable identity observed for {self.config.min_entry_observations} frames in {self.config.entry_window_seconds}s window"
                    ))
            else:
                # Missing observation during candidate window
                if self.entry_timestamps:
                    oldest = self.entry_timestamps[0]
                    if (now - oldest).total_seconds() > self.config.entry_window_seconds:
                        # Candidate window expired without sufficient evidence -> Revert
                        prev = self.state
                        self.state = PresenceState.NOT_PRESENT
                        self.entry_timestamps.clear()
                        events.append(PresenceEvent(
                            event_type=PresenceEventType.STATE_CHANGED,
                            identity=self.identity,
                            timestamp=now,
                            previous_state=prev,
                            new_state=PresenceState.NOT_PRESENT,
                            reason="entry window expired without sufficient stable observations"
                        ))

        elif self.state == PresenceState.PRESENT:
            if is_stable and self.active_session is not None:
                # Continue active session
                self.active_session.last_seen_at = now
                self.active_session.observation_count += 1
                self.active_session.duration_seconds = self.active_session.compute_duration(now)
                events.append(PresenceEvent(
                    event_type=PresenceEventType.PRESENCE_UPDATED,
                    identity=self.identity,
                    timestamp=now,
                    previous_state=PresenceState.PRESENT,
                    new_state=PresenceState.PRESENT,
                    session_id=self.active_session.session_id,
                    reason="presence confirmed with ongoing stable observations"
                ))
            else:
                # Lost stable observation -> Enter GRACE
                prev = self.state
                self.state = PresenceState.GRACE
                self.grace_start_time = now
                if self.active_session is not None:
                    self.active_session.state = PresenceState.GRACE
                    self.active_session.duration_seconds = self.active_session.compute_duration(now)
                events.append(PresenceEvent(
                    event_type=PresenceEventType.GRACE_STARTED,
                    identity=self.identity,
                    timestamp=now,
                    previous_state=prev,
                    new_state=PresenceState.GRACE,
                    session_id=self.active_session.session_id if self.active_session else None,
                    reason=f"stable observation lost; grace timer started ({self.config.grace_period_seconds}s allowance)"
                ))

        elif self.state == PresenceState.GRACE:
            if is_stable and self.active_session is not None:
                # Identity returned before grace timeout -> Resume PRESENT
                prev = self.state
                self.state = PresenceState.PRESENT
                self.grace_start_time = None
                self.active_session.state = PresenceState.PRESENT
                self.active_session.last_seen_at = now
                self.active_session.observation_count += 1
                self.active_session.interruption_count += 1
                self.active_session.duration_seconds = self.active_session.compute_duration(now)
                events.append(PresenceEvent(
                    event_type=PresenceEventType.PRESENCE_RESUMED,
                    identity=self.identity,
                    timestamp=now,
                    previous_state=prev,
                    new_state=PresenceState.PRESENT,
                    session_id=self.active_session.session_id,
                    reason="identity reappeared before grace period expiration; session resumed"
                ))
            else:
                # Still absent during grace -> Check if grace period expired
                if self.grace_start_time is not None:
                    elapsed = (now - self.grace_start_time).total_seconds()
                    if elapsed >= self.config.grace_period_seconds:
                        events.extend(self._close_session(
                            now=now,
                            reason=f"identity absent beyond configured grace period ({self.config.grace_period_seconds}s)"
                        ))

        return events

    def check_grace_timeout(self, timestamp: Optional[datetime] = None) -> List[PresenceEvent]:
        """Checks if current state is in GRACE and has exceeded the grace period."""
        if self.state != PresenceState.GRACE or self.grace_start_time is None:
            return []
        now = self._now(timestamp)
        elapsed = (now - self.grace_start_time).total_seconds()
        if elapsed >= self.config.grace_period_seconds:
            return self._close_session(
                now=now,
                reason=f"identity absent beyond configured grace period ({self.config.grace_period_seconds}s)"
            )
        return []

    def _close_session(self, now: datetime, reason: str) -> List[PresenceEvent]:
        """Finalizes active session and transitions state to ABSENT."""
        prev = self.state
        self.state = PresenceState.ABSENT
        self.grace_start_time = None

        events: List[PresenceEvent] = []
        if self.active_session is not None:
            self.active_session.state = PresenceState.ABSENT
            self.active_session.ended_at = now
            self.active_session.duration_seconds = self.active_session.compute_duration(now)

            events.append(PresenceEvent(
                event_type=PresenceEventType.SESSION_ENDED,
                identity=self.identity,
                timestamp=now,
                previous_state=prev,
                new_state=PresenceState.ABSENT,
                session_id=self.active_session.session_id,
                reason=reason
            ))

        return events

    def reset(self):
        """Resets state machine to initial NOT_PRESENT."""
        self.state = PresenceState.NOT_PRESENT
        self.active_session = None
        self.entry_timestamps.clear()
        self.grace_start_time = None
