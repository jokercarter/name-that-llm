import numpy as np

from ntl_pact18.labels import N_CLASSES
from ntl_pact18.prism import (
    capacity_relaxed_assignment,
    decode,
    hard_assignment,
)


def _clean_logp(m: int = N_CLASSES, margin: float = 10.0) -> np.ndarray:
    logp = np.full((m, N_CLASSES), -margin)
    logp[np.arange(m), np.arange(m)] = 0.0
    return logp


def test_perfect_matrix_recovers_labels():
    logp = _clean_logp()
    assert (hard_assignment(logp) == np.arange(N_CLASSES)).all()
    assert (capacity_relaxed_assignment(logp, lam=1.0) == np.arange(N_CLASSES)).all()


def test_constraint_repairs_a_wrong_argmax():
    logp = _clean_logp(margin=6.0)
    # row 0 prefers class 1, but class 0 must be used by someone: the joint
    # decode should send row 0 back to class 0 and row 1 to class 1.
    logp[0, 0] = -0.2
    logp[0, 1] = 0.0
    assert logp.argmax(axis=1)[0] == 1
    pred = capacity_relaxed_assignment(logp, lam=1.0)
    assert pred[0] == 0
    assert len(set(pred.tolist())) == N_CLASSES


def test_hard_assignment_is_a_permutation():
    rng = np.random.default_rng(0)
    logp = rng.normal(size=(N_CLASSES, N_CLASSES))
    pred = hard_assignment(logp)
    assert sorted(pred.tolist()) == list(range(N_CLASSES))


def test_partial_cluster_gets_distinct_classes():
    rng = np.random.default_rng(1)
    logp = rng.normal(size=(8, N_CLASSES))
    pred = hard_assignment(logp)
    assert len(pred) == 8
    assert len(set(pred.tolist())) == 8


def test_large_lambda_forbids_duplicates_small_lambda_allows_them():
    logp = np.full((4, N_CLASSES), -8.0)
    for r in range(4):  # every row wants class 0
        logp[r, 0] = 0.0
        logp[r, 1] = -1.0
    strict = capacity_relaxed_assignment(logp, lam=50.0)
    assert len(set(strict.tolist())) == 4
    loose = capacity_relaxed_assignment(logp, lam=0.0)
    assert len(set(loose.tolist())) == 1


def test_decode_skips_singletons_and_oversized_clusters():
    probs = np.full((4, N_CLASSES), 1.0 / N_CLASSES)
    probs[:, 3] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    groups = np.array([0, 0, 1, 2])  # cluster 0 has size 2, 1 and 2 are singletons
    preds, stats = decode(probs, groups, mode="hard", lam=1.0, min_size=2, max_size=18)
    assert stats.n_clusters == 1
    assert preds.shape == (4,)
    # the two rows in cluster 0 must not share a class
    assert preds[0] != preds[1]
    # singletons keep their argmax
    assert preds[2] == 3 and preds[3] == 3


def test_relaxed_mode_allows_duplicates_on_overwhelming_evidence():
    probs = np.full((2, N_CLASSES), 1.0 / N_CLASSES)
    probs[:, 7] = 0.95
    probs /= probs.sum(axis=1, keepdims=True)
    preds, stats = decode(probs, np.array([0, 0]), mode="relaxed", lam=1.0)
    # splitting would cost ~5 nats, duplicating costs only lam
    assert (preds == 7).all()
    assert stats.n_duplicate_slots_used == 1


def test_confidence_gate_disables_decoding():
    probs = np.full((2, N_CLASSES), 0.01)
    probs[:, 5] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    groups = np.array([0, 0])
    confidence = np.array([0.01, 0.01])
    preds, stats = decode(probs, groups, confidence=confidence, confidence_threshold=0.5)
    assert stats.n_rows_assigned == 0
    assert (preds == 5).all()
