"""PRISM: Prompt-constrained Relaxed Integer Soft Matching.

The competition corpus is a (prompt x model) grid: within one prompt each model
appears at most once. Independent argmax ignores that; Sinkhorn only balances
global marginals. PRISM decodes each inferred prompt cluster jointly:

* primary slots: one per class, cost ``-log p``
* duplicate slots: one per class per row, cost ``-log p + lam``

``lam`` therefore controls how strongly we believe the mutual-exclusion
constraint. ``lam=0`` allows duplicates freely (≈ argmax with a coupling term),
``lam=inf`` is a hard one-per-class assignment. On simulated clusters
``lam ~ 1 nat`` captures most of the gain while degrading gracefully when the
clusterer merges two prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .labels import N_CLASSES
from .utils import EPS


@dataclass
class PrismStats:
    n_clusters: int = 0
    n_rows_in_clusters: int = 0
    n_rows_assigned: int = 0
    n_changed: int = 0
    n_duplicate_slots_used: int = 0
    cluster_sizes: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "n_clusters": self.n_clusters,
            "n_rows_in_clusters": self.n_rows_in_clusters,
            "n_rows_assigned": self.n_rows_assigned,
            "n_changed": self.n_changed,
            "n_duplicate_slots_used": self.n_duplicate_slots_used,
        }


def _log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))


def capacity_relaxed_assignment(logp: np.ndarray, lam: float = 1.0,
                                n_classes: int = N_CLASSES) -> np.ndarray:
    """Solve one cluster. ``logp`` has shape ``(m, n_classes)``."""
    m = logp.shape[0]
    if m <= 1:
        return logp.argmax(axis=1)
    # one primary slot per class + n_classes duplicate slots per class
    cost = np.hstack([-logp] + [-logp + float(lam)] * n_classes)
    rows, cols = linear_sum_assignment(cost)
    pred = np.empty(m, dtype=int)
    pred[rows] = cols % n_classes
    return pred


def hard_assignment(logp: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """One distinct class per row (rectangular Hungarian)."""
    m = logp.shape[0]
    if m <= 1:
        return logp.argmax(axis=1)
    rows, cols = linear_sum_assignment(-logp)
    pred = np.full(m, -1, dtype=int)
    pred[rows] = cols
    # rows beyond n_classes cannot be matched (m > K); fall back for those
    if (pred < 0).any():
        pred[pred < 0] = logp[pred < 0].argmax(axis=1)
    return pred


def solve_cluster(probs: np.ndarray, mode: str = "relaxed", lam: float = 1.0) -> np.ndarray:
    logp = _log(probs)
    if mode == "hard":
        return hard_assignment(logp)
    return capacity_relaxed_assignment(logp, lam=lam)


def decode(probs: np.ndarray, groups: Sequence[int], mode: str = "relaxed",
           lam: float = 1.0, min_size: int = 2, max_size: int = 18,
           confidence: Optional[np.ndarray] = None,
           confidence_threshold: float = 0.0) -> tuple[np.ndarray, PrismStats]:
    """Apply PRISM cluster by cluster and return ``(predictions, stats)``."""
    probs = np.asarray(probs, dtype=np.float64)
    groups = np.asarray(groups)
    pred = probs.argmax(axis=1).astype(int)
    stats = PrismStats()

    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        size = len(idx)
        stats.cluster_sizes[int(g)] = int(size)
        if size < max(min_size, 2) or size > max_size:
            continue
        stats.n_clusters += 1
        stats.n_rows_in_clusters += size
        if confidence is not None and confidence_threshold > 0:
            if float(np.mean(confidence[idx])) < confidence_threshold:
                continue
        local = solve_cluster(probs[idx], mode=mode, lam=lam)
        stats.n_rows_assigned += size
        stats.n_changed += int((local != pred[idx]).sum())
        stats.n_duplicate_slots_used += int(size - len(np.unique(local)))
        pred[idx] = local

    return pred, stats
