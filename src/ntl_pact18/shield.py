"""Orthographic Bayesian shield.

Some orthographic habits are effectively hard constraints: in the training data
Claude and GLM never emit curly quotes, and Gemini almost never emits two or
more em-dashes. Marginal balancing (Sinkhorn) can resurrect those impossible
classes, so after balancing we apply a log-space penalty to any family that has
been observed to violate the marker at a rate below ``max_rate``.

Rules are *discovered from the training split*, not hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .labels import LABELS, N_CLASSES, family_masks
from .utils import EPS

MARKERS: Dict[str, Callable[[str], bool]] = {
    "curly_quote": lambda t: any(c in t for c in ("\u2019", "\u201c", "\u201d")),
    "double_em_dash": lambda t: t.count("\u2014") >= 2,
    "double_space": lambda t: "  " in t,
    "leading_space": lambda t: t[:1] == " ",
}


@dataclass
class ShieldRule:
    marker: str
    family: str
    class_indices: List[int]
    observed_rate: float
    support: int

    def describe(self) -> str:
        return (
            f"{self.marker} -> {self.family} "
            f"(rate={self.observed_rate:.4f}, n={self.support})"
        )


class OrthographicShield:
    def __init__(self, penalty: float = 2.5, max_rate: float = 0.01,
                 min_support: int = 40, min_global_rate: float = 0.02):
        self.penalty = float(penalty)
        self.max_rate = float(max_rate)
        self.min_support = int(min_support)
        self.min_global_rate = float(min_global_rate)
        self.rules: List[ShieldRule] = []

    # ------------------------------------------------------------------- fit
    def fit(self, texts: Sequence[str], y_idx: Sequence[int]) -> "OrthographicShield":
        texts = list(texts)
        y_idx = np.asarray(y_idx, dtype=int)
        self.rules = []
        n_total = max(len(texts), 1)

        for marker_name, fn in MARKERS.items():
            hits = np.asarray([fn(t) for t in texts], dtype=bool)
            global_rate = float(hits.mean())
            # a marker that never fires carries no information: skip it
            if global_rate < self.min_global_rate:
                continue
            for family, idx in family_masks():
                mask = np.isin(y_idx, idx)
                support = int(mask.sum())
                if support < self.min_support:
                    continue
                rate = float(hits[mask].mean()) if support else 1.0
                if rate <= self.max_rate:
                    self.rules.append(
                        ShieldRule(marker_name, family, list(idx), rate, support)
                    )
        return self

    # ----------------------------------------------------------------- apply
    def apply(self, probs: np.ndarray, texts: Sequence[str]) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64).copy()
        if not self.rules:
            return p
        texts = list(texts)
        for rule in self.rules:
            fn = MARKERS[rule.marker]
            rows = np.asarray([i for i, t in enumerate(texts) if fn(t)], dtype=int)
            if not len(rows):
                continue
            logp = np.log(np.clip(p[rows], EPS, 1.0))
            logp[:, rule.class_indices] -= self.penalty
            logp -= logp.max(axis=1, keepdims=True)
            q = np.exp(logp)
            p[rows] = q / np.maximum(q.sum(axis=1, keepdims=True), EPS)
        return p

    def describe(self) -> List[str]:
        return [r.describe() for r in self.rules]
