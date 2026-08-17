import collections
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from ml.temporal.schemas import (
    RecognitionObservation,
    TemporalState,
    TemporalRecognitionResult,
    TemporalPolicyConfig,
    TemporalMode
)

PRESET_POLICIES: Dict[TemporalMode, TemporalPolicyConfig] = {
    TemporalMode.FAST: TemporalPolicyConfig(
        window_size=4,
        min_observations=3,
        min_stable_ratio=0.65,
        max_unknown_observations=2,
        challenger_switch_threshold=2,
        max_gap_seconds=2.0,
        mode="fast"
    ),
    TemporalMode.BALANCED: TemporalPolicyConfig(
        window_size=7,
        min_observations=4,
        min_stable_ratio=0.70,
        max_unknown_observations=3,
        challenger_switch_threshold=3,
        max_gap_seconds=2.0,
        mode="balanced"
    ),
    TemporalMode.STABLE: TemporalPolicyConfig(
        window_size=10,
        min_observations=6,
        min_stable_ratio=0.80,
        max_unknown_observations=4,
        challenger_switch_threshold=5,
        max_gap_seconds=2.5,
        mode="stable"
    )
}


class TemporalIdentityStabilizer:
    """
    Temporal Identity Stabilizer.
    Aggregates sequential frame-level recognition observations over a bounded sliding window
    to eliminate single-frame identity flicker, absorb transient Unknown dropouts, and prevent
    erroneous identity switches without sufficient temporal consensus.
    """

    def __init__(
        self,
        policy: Optional[TemporalPolicyConfig] = None,
        mode: TemporalMode = TemporalMode.BALANCED,
        enabled: bool = True
    ):
        self.mode = mode
        self.enabled = enabled
        self.policy = policy or PRESET_POLICIES.get(mode, PRESET_POLICIES[TemporalMode.BALANCED])
        self.history: collections.deque = collections.deque(maxlen=self.policy.window_size)

        # Internal state tracking
        self.current_stable_identity: Optional[str] = None
        self.consecutive_stable_count: int = 0
        self.consecutive_unknown_count: int = 0
        self.challenger_identity: Optional[str] = None
        self.challenger_count: int = 0
        self.last_observation_time: Optional[float] = None

    @classmethod
    def from_config(cls, temporal_config: Dict[str, Any]) -> "TemporalIdentityStabilizer":
        """Factory method constructing TemporalIdentityStabilizer from system configuration."""
        enabled = temporal_config.get("enabled", True)
        mode_str = temporal_config.get("mode", "balanced").lower()

        try:
            mode = TemporalMode(mode_str)
        except ValueError:
            mode = TemporalMode.BALANCED

        base_policy = PRESET_POLICIES.get(mode, PRESET_POLICIES[TemporalMode.BALANCED])

        custom_policy = TemporalPolicyConfig(
            window_size=int(temporal_config.get("window_size", base_policy.window_size)),
            min_observations=int(temporal_config.get("min_observations", base_policy.min_observations)),
            min_stable_ratio=float(temporal_config.get("min_stable_ratio", base_policy.min_stable_ratio)),
            max_unknown_observations=int(temporal_config.get("max_unknown_observations", base_policy.max_unknown_observations)),
            challenger_switch_threshold=int(temporal_config.get("challenger_switch_threshold", base_policy.challenger_switch_threshold)),
            max_gap_seconds=float(temporal_config.get("max_gap_seconds", base_policy.max_gap_seconds)),
            mode=mode.value
        )

        return cls(policy=custom_policy, mode=mode, enabled=enabled)

    def reset(self):
        """Clears sliding window and resets all temporal state variables."""
        self.history.clear()
        self.current_stable_identity = None
        self.consecutive_stable_count = 0
        self.consecutive_unknown_count = 0
        self.challenger_identity = None
        self.challenger_count = 0
        self.last_observation_time = None

    def update(self, observation: RecognitionObservation) -> TemporalRecognitionResult:
        """
        Updates temporal state with a new frame observation and computes stabilized decision.

        Args:
            observation (RecognitionObservation): Current frame recognition observation.

        Returns:
            TemporalRecognitionResult: Stabilized identity decision and temporal metadata.
        """
        # If disabled, act as pass-through for frame-level recognition
        if not self.enabled:
            is_st = observation.recognized and observation.identity is not None
            return TemporalRecognitionResult(
                stable_identity=observation.identity,
                state=TemporalState.STABLE if is_st else TemporalState.UNKNOWN,
                confidence_score=observation.similarity if observation.recognized else 0.0,
                observations_count=1,
                consecutive_stable_count=1 if is_st else 0,
                active_candidate=observation.best_candidate,
                challenger_identity=None,
                challenger_evidence=0,
                is_stable=is_st,
                latest_observation=observation
            )

        # 1. Temporal gap expiration check
        if self.last_observation_time is not None:
            time_delta = observation.timestamp - self.last_observation_time
            if time_delta > self.policy.max_gap_seconds:
                self.reset()

        self.last_observation_time = observation.timestamp
        self.history.append(observation)

        # 2. Accumulate evidence across sliding window with Quality weighting
        # Quality-rejected frames are ignored for positive evidence
        identity_weights: Dict[str, float] = collections.defaultdict(float)
        total_valid_weight = 0.0
        unknown_weight = 0.0

        for obs in self.history:
            if obs.quality_status == "poor":
                # Poor quality frames carry 0 positive weight
                continue

            if obs.recognized and obs.identity is not None:
                w = 1.0
                identity_weights[obs.identity] += w
                total_valid_weight += w
            else:
                unknown_weight += 1.0
                total_valid_weight += 1.0

        # Determine leading candidate in current window
        leading_candidate = observation.best_candidate
        leading_weight = 0.0
        if identity_weights:
            leading_candidate, leading_weight = max(identity_weights.items(), key=lambda x: x[1])

        consensus_ratio = (leading_weight / total_valid_weight) if total_valid_weight > 0 else 0.0

        # 3. State Transition Logic
        current_state = TemporalState.UNSTABLE
        active_identity = self.current_stable_identity

        if self.current_stable_identity is None:
            # Case A: No active stable identity yet
            if leading_weight >= self.policy.min_observations and consensus_ratio >= self.policy.min_stable_ratio:
                self.current_stable_identity = leading_candidate
                self.consecutive_stable_count = int(leading_weight)
                self.consecutive_unknown_count = 0
                self.challenger_identity = None
                self.challenger_count = 0
                current_state = TemporalState.STABLE
                active_identity = leading_candidate
            elif unknown_weight >= self.policy.min_observations:
                current_state = TemporalState.UNKNOWN
                active_identity = None
            else:
                current_state = TemporalState.UNSTABLE
                active_identity = None

        else:
            # Case B: Currently holding a stable identity A
            stable_id = self.current_stable_identity

            if observation.recognized and observation.identity == stable_id:
                # Direct confirmation of stable identity
                self.consecutive_stable_count += 1
                self.consecutive_unknown_count = 0
                self.challenger_identity = None
                self.challenger_count = 0
                current_state = TemporalState.STABLE
                active_identity = stable_id

            elif not observation.recognized or observation.identity is None:
                # Observation is Unknown or quality rejected
                self.consecutive_unknown_count += 1

                if self.consecutive_unknown_count <= self.policy.max_unknown_observations:
                    # Absorb transient Unknown jitter; retain stable identity
                    current_state = TemporalState.STABLE
                    active_identity = stable_id
                else:
                    # Unknowns exceeded tolerance; expire stable identity
                    self.current_stable_identity = None
                    self.consecutive_stable_count = 0
                    current_state = TemporalState.UNKNOWN
                    active_identity = None

            else:
                # Observation is a competing recognized identity B != A (Potential identity switch)
                self.consecutive_unknown_count = 0
                challenger = observation.identity

                if self.challenger_identity == challenger:
                    self.challenger_count += 1
                else:
                    self.challenger_identity = challenger
                    self.challenger_count = 1

                if self.challenger_count >= self.policy.challenger_switch_threshold:
                    # Challenger accumulated sufficient consecutive evidence -> Commit switch
                    self.current_stable_identity = challenger
                    self.consecutive_stable_count = self.challenger_count
                    self.challenger_identity = None
                    self.challenger_count = 0
                    current_state = TemporalState.STABLE
                    active_identity = challenger
                else:
                    # Suppress transient blip; indicate switching state while preserving stable identity A
                    current_state = TemporalState.SWITCHING
                    active_identity = stable_id

        # Calculate temporal evidence strength [0.0, 1.0]
        confidence_score = float(np.clip(consensus_ratio, 0.0, 1.0)) if active_identity else 0.0

        return TemporalRecognitionResult(
            stable_identity=active_identity,
            state=current_state,
            confidence_score=confidence_score,
            observations_count=len(self.history),
            consecutive_stable_count=self.consecutive_stable_count,
            active_candidate=leading_candidate,
            challenger_identity=self.challenger_identity,
            challenger_evidence=self.challenger_count,
            is_stable=(current_state == TemporalState.STABLE),
            latest_observation=observation
        )

    def process_result(
        self,
        result: Any,
        frame_index: int = 0,
        timestamp: Optional[float] = None
    ) -> TemporalRecognitionResult:
        """Helper to process a ModernRecognitionResult directly."""
        obs = RecognitionObservation.from_recognition_result(
            result=result,
            frame_index=frame_index,
            timestamp=timestamp
        )
        return self.update(obs)

    def get_state(self) -> Dict[str, Any]:
        """Returns current internal tracker state dictionary."""
        return {
            "current_stable_identity": self.current_stable_identity,
            "consecutive_stable_count": self.consecutive_stable_count,
            "consecutive_unknown_count": self.consecutive_unknown_count,
            "challenger_identity": self.challenger_identity,
            "challenger_count": self.challenger_count,
            "window_size": len(self.history),
            "max_window_size": self.policy.window_size,
            "mode": self.policy.mode
        }
