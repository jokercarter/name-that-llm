import numpy as np

from ntl_pact18.evaluate import (
    balanced_accuracy,
    bootstrap_balanced_accuracy,
    cluster_pair_scores,
    macro_f1,
    per_class_recall,
    top_k_accuracy,
)


def test_balanced_accuracy_bounds():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert balanced_accuracy(y, y) == 1.0
    assert balanced_accuracy(y, np.zeros_like(y)) == 1 / 3


def test_per_class_recall_handles_missing_classes():
    y = np.array([0, 1, 1])
    pred = np.array([0, 1, 0])
    recalls = per_class_recall(y, pred)
    assert np.isclose(recalls[0], 1.0)
    assert np.isclose(recalls[1], 0.5)
    assert np.isnan(recalls[2])


def test_macro_f1_perfect():
    y = np.array([0, 1, 2, 1])
    assert macro_f1(y, y) == 1.0


def test_top_k_accuracy():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.1, 0.4]])
    y = np.array([1, 2])
    assert top_k_accuracy(probs, y, 1) == 0.5
    assert top_k_accuracy(probs, y, 2) == 1.0


def test_cluster_pair_scores_identical_partitions():
    groups = np.array([0, 0, 1, 1, 2])
    p, r, f1 = cluster_pair_scores(groups, groups)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_cluster_pair_scores_detects_merge():
    pred = np.array([0, 0, 0, 0])
    true = np.array([0, 0, 1, 1])
    p, r, _ = cluster_pair_scores(pred, true)
    assert r == 1.0
    assert p < 1.0


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 18, size=200)
    pred = y.copy()
    pred[:40] = (pred[:40] + 1) % 18
    groups = np.repeat(np.arange(20), 10)
    mean, lo, hi = bootstrap_balanced_accuracy(y, pred, groups, n_boot=100, seed=0)
    assert lo <= mean <= hi
