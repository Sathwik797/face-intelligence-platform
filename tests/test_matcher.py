import pytest
import numpy as np
from ml.matcher import EuclideanMatcher, CosineMatcher

# --- EuclideanMatcher Tests (Phase 1 Baseline) ---

def test_euclidean_matcher_known_identity():
    v_alice = np.zeros(128, dtype=np.float64)
    v_alice[0] = 1.0

    v_bob = np.zeros(128, dtype=np.float64)
    v_bob[1] = 1.0

    matcher = EuclideanMatcher(
        known_encodings=[v_alice, v_bob],
        known_names=["Alice", "Bob"],
        threshold=0.6
    )

    query_alice = np.zeros(128, dtype=np.float64)
    query_alice[0] = 0.95
    name, dist, is_rec = matcher.match(query_alice)
    assert name == "Alice"
    assert is_rec is True
    assert dist < 0.6


def test_euclidean_matcher_unknown_identity():
    v_alice = np.zeros(128, dtype=np.float64)
    v_alice[0] = 1.0

    matcher = EuclideanMatcher(
        known_encodings=[v_alice],
        known_names=["Alice"],
        threshold=0.6
    )

    query_charlie = np.zeros(128, dtype=np.float64)
    query_charlie[50] = 1.0

    name, dist, is_rec = matcher.match(query_charlie)
    assert name == "Unknown"
    assert is_rec is False
    assert dist > 0.6


def test_euclidean_matcher_threshold_boundary():
    v_ref = np.zeros(128, dtype=np.float64)
    matcher = EuclideanMatcher(known_encodings=[v_ref], known_names=["User"], threshold=0.5)

    q_pass = np.zeros(128, dtype=np.float64)
    q_pass[0] = 0.4
    name, dist, is_rec = matcher.match(q_pass)
    assert name == "User"
    assert is_rec is True

    q_fail = np.zeros(128, dtype=np.float64)
    q_fail[0] = 0.6
    name, dist, is_rec = matcher.match(q_fail)
    assert name == "Unknown"
    assert is_rec is False


def test_euclidean_matcher_empty_gallery():
    matcher = EuclideanMatcher(known_encodings=[], known_names=[], threshold=0.6)
    query = np.zeros(128, dtype=np.float64)
    name, dist, is_rec = matcher.match(query)
    assert name == "Unknown"
    assert dist == float("inf")
    assert is_rec is False


def test_euclidean_matcher_update_gallery():
    matcher = EuclideanMatcher()
    v1 = np.ones(128, dtype=np.float64)
    matcher.update_gallery([v1], ["EnrolledUser"])
    assert len(matcher.known_names) == 1
    assert matcher.known_names[0] == "EnrolledUser"


# --- CosineMatcher Tests (Phase 4 Modern Model) ---

def test_cosine_matcher_known_identity():
    # Unit vectors
    v_alice = np.zeros(512, dtype=np.float32)
    v_alice[0] = 1.0

    v_bob = np.zeros(512, dtype=np.float32)
    v_bob[1] = 1.0

    matcher = CosineMatcher(
        known_encodings=[v_alice, v_bob],
        known_names=["Alice", "Bob"],
        threshold=0.4
    )

    # Query vector close to Alice (cosine similarity ~ 0.98)
    q_alice = np.zeros(512, dtype=np.float32)
    q_alice[0] = 0.98
    q_alice[2] = 0.2
    q_alice = q_alice / np.linalg.norm(q_alice)

    name, sim, is_rec = matcher.match(q_alice)
    assert name == "Alice"
    assert is_rec is True
    assert sim >= 0.4


def test_cosine_matcher_unknown_identity():
    v_alice = np.zeros(512, dtype=np.float32)
    v_alice[0] = 1.0

    matcher = CosineMatcher(
        known_encodings=[v_alice],
        known_names=["Alice"],
        threshold=0.4
    )

    # Query orthogonal to Alice (cosine similarity == 0.0 < threshold 0.4)
    q_other = np.zeros(512, dtype=np.float32)
    q_other[100] = 1.0

    name, sim, is_rec = matcher.match(q_other)
    assert name == "Unknown"
    assert is_rec is False
    assert sim < 0.4


def test_cosine_matcher_threshold_boundary():
    v_ref = np.zeros(512, dtype=np.float32)
    v_ref[0] = 1.0

    matcher = CosineMatcher(
        known_encodings=[v_ref],
        known_names=["User"],
        threshold=0.5
    )

    # Query with cosine similarity 0.7 >= 0.5
    q_pass = np.zeros(512, dtype=np.float32)
    q_pass[0] = 0.7
    q_pass[1] = np.sqrt(1 - 0.7**2)
    name, sim, is_rec = matcher.match(q_pass)
    assert name == "User"
    assert is_rec is True

    # Query with cosine similarity 0.3 < 0.5
    q_fail = np.zeros(512, dtype=np.float32)
    q_fail[0] = 0.3
    q_fail[1] = np.sqrt(1 - 0.3**2)
    name, sim, is_rec = matcher.match(q_fail)
    assert name == "Unknown"
    assert is_rec is False


def test_cosine_matcher_empty_gallery():
    matcher = CosineMatcher(known_encodings=[], known_names=[], threshold=0.4)
    q = np.ones(512, dtype=np.float32)
    name, sim, is_rec = matcher.match(q)
    assert name == "Unknown"
    assert sim == -1.0
    assert is_rec is False
