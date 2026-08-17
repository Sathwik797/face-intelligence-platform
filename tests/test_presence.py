import pytest
from datetime import datetime, timezone, timedelta

from ml.presence import (
    PresenceManager,
    PresenceState,
    PresenceEventType,
    PresenceMode,
    PresenceConfig,
    PresenceEvent,
    PresenceSession,
    IdentityPresenceStateMachine,
    PRESENCE_PRESETS
)
from ml.temporal import (
    TemporalRecognitionResult,
    TemporalState,
    RecognitionObservation
)


def _mock_time(base_timestamp_sec: float = 1000.0):
    return datetime.fromtimestamp(base_timestamp_sec, tz=timezone.utc)


def _make_temp_result(identity="Alice", is_stable=True):
    return TemporalRecognitionResult(
        stable_identity=identity if is_stable else None,
        state=TemporalState.STABLE if is_stable else TemporalState.UNSTABLE,
        confidence_score=0.85 if is_stable else 0.0,
        observations_count=5,
        consecutive_stable_count=5,
        active_candidate=identity,
        is_stable=is_stable
    )


def test_presence_initial_state():
    sm = IdentityPresenceStateMachine(identity="Alice")
    assert sm.state == PresenceState.NOT_PRESENT
    assert sm.active_session is None


def test_presence_candidate_accumulation_and_entry_confirmation():
    t0 = _mock_time(100.0)
    sm = IdentityPresenceStateMachine(
        identity="Alice",
        config=PRESENCE_PRESETS[PresenceMode.BALANCED]  # min_entry=3, window=5.0s
    )

    # Frame 1: NOT_PRESENT -> CANDIDATE
    ev1 = sm.process_observation(is_stable=True, timestamp=t0)
    assert sm.state == PresenceState.CANDIDATE
    assert len(ev1) == 1
    assert ev1[0].new_state == PresenceState.CANDIDATE

    # Frame 2: CANDIDATE
    ev2 = sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=0.5))
    assert sm.state == PresenceState.CANDIDATE
    assert len(ev2) == 0

    # Frame 3: ENTRY_CONFIRMED -> PRESENT
    ev3 = sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=1.0))
    assert sm.state == PresenceState.PRESENT
    assert len(ev3) == 1
    assert ev3[0].event_type == PresenceEventType.ENTRY_CONFIRMED
    assert sm.active_session is not None
    assert sm.active_session.identity == "Alice"
    assert sm.active_session.observation_count == 3


def test_presence_candidate_timeout_reverts_to_not_present():
    t0 = _mock_time(100.0)
    sm = IdentityPresenceStateMachine(
        identity="Alice",
        config=PRESENCE_PRESETS[PresenceMode.BALANCED]  # window=5.0s
    )

    # Frame 1 at t=0
    sm.process_observation(is_stable=True, timestamp=t0)
    assert sm.state == PresenceState.CANDIDATE

    # Next observation at t=10s with is_stable=False (window expired)
    ev_timeout = sm.process_observation(is_stable=False, timestamp=t0 + timedelta(seconds=10.0))
    assert sm.state == PresenceState.NOT_PRESENT
    assert len(ev_timeout) == 1
    assert ev_timeout[0].new_state == PresenceState.NOT_PRESENT


def test_presence_session_update_on_repeated_observations():
    t0 = _mock_time(100.0)
    sm = IdentityPresenceStateMachine(
        identity="Alice",
        config=PRESENCE_PRESETS[PresenceMode.FAST]  # min_entry=2
    )
    # Establish PRESENT
    sm.process_observation(is_stable=True, timestamp=t0)
    sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=0.5))
    assert sm.state == PresenceState.PRESENT
    session_id = sm.active_session.session_id

    # Continuous frame updates
    for i in range(1, 6):
        evs = sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=0.5 + i * 0.5))
        assert sm.state == PresenceState.PRESENT
        assert len(evs) == 1
        assert evs[0].event_type == PresenceEventType.PRESENCE_UPDATED
        assert sm.active_session.session_id == session_id
        assert sm.active_session.observation_count == 2 + i


def test_presence_grace_transition_and_recovery():
    t0 = _mock_time(100.0)
    sm = IdentityPresenceStateMachine(
        identity="Alice",
        config=PRESENCE_PRESETS[PresenceMode.BALANCED]  # grace=10.0s
    )
    # Confirm PRESENT
    for i in range(3):
        sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=i * 0.5))
    assert sm.state == PresenceState.PRESENT
    session_id = sm.active_session.session_id

    # Stable observation lost -> GRACE
    ev_grace = sm.process_observation(is_stable=False, timestamp=t0 + timedelta(seconds=2.0))
    assert sm.state == PresenceState.GRACE
    assert len(ev_grace) == 1
    assert ev_grace[0].event_type == PresenceEventType.GRACE_STARTED
    assert sm.active_session.is_active is True

    # Identity returns at t=4.0s (within 10s grace period) -> Resume PRESENT
    ev_resume = sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=4.0))
    assert sm.state == PresenceState.PRESENT
    assert len(ev_resume) == 1
    assert ev_resume[0].event_type == PresenceEventType.PRESENCE_RESUMED
    assert sm.active_session.session_id == session_id  # Same session preserved
    assert sm.active_session.interruption_count == 1


def test_presence_grace_timeout_session_ended():
    t0 = _mock_time(100.0)
    sm = IdentityPresenceStateMachine(
        identity="Alice",
        config=PRESENCE_PRESETS[PresenceMode.FAST]  # grace=5.0s
    )
    # Confirm PRESENT at t=100.0s to t=101.0s
    sm.process_observation(is_stable=True, timestamp=t0)
    sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=1.0))
    assert sm.state == PresenceState.PRESENT

    # Enter GRACE at t=102.0s
    sm.process_observation(is_stable=False, timestamp=t0 + timedelta(seconds=2.0))
    assert sm.state == PresenceState.GRACE

    # Check timeout at t=108.0s (6.0s elapsed in grace > 5.0s limit) -> ABSENT
    ev_timeout = sm.check_grace_timeout(timestamp=t0 + timedelta(seconds=8.0))
    assert sm.state == PresenceState.ABSENT
    assert len(ev_timeout) == 1
    assert ev_timeout[0].event_type == PresenceEventType.SESSION_ENDED
    assert sm.active_session.is_active is False
    assert sm.active_session.ended_at == t0 + timedelta(seconds=8.0)
    assert sm.active_session.duration_seconds == 7.0  # 108.0 - 101.0


def test_presence_new_session_after_absence():
    t0 = _mock_time(100.0)
    sm = IdentityPresenceStateMachine(
        identity="Alice",
        config=PRESENCE_PRESETS[PresenceMode.FAST]  # min_entry=2, grace=5.0s
    )
    # Session 1: t=100.0 -> t=108.0
    sm.process_observation(is_stable=True, timestamp=t0)
    sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=1.0))
    sm.process_observation(is_stable=False, timestamp=t0 + timedelta(seconds=2.0))
    sm.check_grace_timeout(timestamp=t0 + timedelta(seconds=8.0))
    assert sm.state == PresenceState.ABSENT
    session1_id = sm.active_session.session_id

    # Later at t=200.0s, Alice returns -> CANDIDATE -> Session 2
    sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=100.0))
    assert sm.state == PresenceState.CANDIDATE

    sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=101.0))
    assert sm.state == PresenceState.PRESENT
    session2_id = sm.active_session.session_id
    assert session2_id != session1_id  # Brand new session created!


def test_presence_unknown_only_observations():
    t0 = _mock_time(100.0)
    sm = IdentityPresenceStateMachine(identity="Alice")
    for i in range(10):
        evs = sm.process_observation(is_stable=False, timestamp=t0 + timedelta(seconds=i * 0.5))
        assert sm.state == PresenceState.NOT_PRESENT
        assert len(evs) == 0
    assert sm.active_session is None


def test_presence_multiple_identities_independent():
    mgr = PresenceManager(config=PRESENCE_PRESETS[PresenceMode.FAST])
    t0 = _mock_time(100.0)

    # Frame 1 & 2: Alice present
    mgr.update(_make_temp_result("Alice", is_stable=True), timestamp=t0)
    mgr.update(_make_temp_result("Alice", is_stable=True), timestamp=t0 + timedelta(seconds=0.5))
    assert mgr.get_presence_state("Alice") == PresenceState.PRESENT
    assert mgr.get_presence_state("Bob") == PresenceState.NOT_PRESENT

    # Frame 3 & 4: Bob present
    mgr.update(_make_temp_result("Bob", is_stable=True), timestamp=t0 + timedelta(seconds=1.0))
    mgr.update(_make_temp_result("Bob", is_stable=True), timestamp=t0 + timedelta(seconds=1.5))
    assert mgr.get_presence_state("Bob") == PresenceState.PRESENT
    assert mgr.get_presence_state("Alice") == PresenceState.GRACE


def test_presence_max_session_duration_safeguard():
    t0 = _mock_time(100.0)
    cfg = PresenceConfig(
        min_entry_observations=2,
        max_session_duration_seconds=30.0  # 30 seconds max duration safeguard
    )
    sm = IdentityPresenceStateMachine(identity="Alice", config=cfg)
    sm.process_observation(is_stable=True, timestamp=t0)
    sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=1.0))
    assert sm.state == PresenceState.PRESENT

    # Observation at t=40.0s (exceeds 30s safeguard) -> ABSENT
    evs = sm.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=40.0))
    assert sm.state == PresenceState.ABSENT
    assert any(e.event_type == PresenceEventType.SESSION_ENDED for e in evs)
    assert "safeguard" in evs[-1].reason


def test_presence_modes_fast_balanced_strict():
    sm_fast = IdentityPresenceStateMachine(identity="Alice", config=PRESENCE_PRESETS[PresenceMode.FAST])
    sm_bal = IdentityPresenceStateMachine(identity="Alice", config=PRESENCE_PRESETS[PresenceMode.BALANCED])
    sm_strict = IdentityPresenceStateMachine(identity="Alice", config=PRESENCE_PRESETS[PresenceMode.STRICT])

    t0 = _mock_time(100.0)
    # 2 observations
    for i in range(2):
        sm_fast.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=i * 0.5))
        sm_bal.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=i * 0.5))
        sm_strict.process_observation(is_stable=True, timestamp=t0 + timedelta(seconds=i * 0.5))

    assert sm_fast.state == PresenceState.PRESENT
    assert sm_bal.state == PresenceState.CANDIDATE
    assert sm_strict.state == PresenceState.CANDIDATE


def test_presence_manager_reset():
    mgr = PresenceManager(config=PRESENCE_PRESETS[PresenceMode.FAST])
    t0 = _mock_time(100.0)
    mgr.update(_make_temp_result("Alice", is_stable=True), timestamp=t0)
    mgr.update(_make_temp_result("Alice", is_stable=True), timestamp=t0 + timedelta(seconds=0.5))
    assert len(mgr.get_active_sessions()) == 1

    mgr.reset()
    assert len(mgr.get_active_sessions()) == 0
    assert len(mgr.get_session_history()) == 0
    assert mgr.get_presence_state("Alice") == PresenceState.NOT_PRESENT
