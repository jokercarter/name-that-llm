"""Two-level family experts (OpenAI / Google / Anthropic / Qwen / ZAI / Moonshot).

The first level is the global 18-way posterior. The second level is a
within-family expert that only rewrites the conditional distribution *inside*
the family the global model already predicted. Nothing outside that family is
touched, which is what makes it safe to train the expert on external passages.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from .labels import N_CLASSES, family_masks
from .utils import EPS


def _stack_features(probs: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(probs, EPS, 1.0))
    ranked = np.sort(probs, axis=1)
    margin = ranked[:, -1] - ranked[:, -2]
    entropy = -(probs * logp).sum(axis=1)
    return np.column_stack([logp, margin, entropy])


class FamilyExperts:
    def __init__(self, min_rows: int = 60, C: float = 1.0, adaptive_rho: bool = True):
        self.min_rows = int(min_rows)
        self.C = float(C)
        self.adaptive_rho = bool(adaptive_rho)
        self.experts: Dict[str, Tuple[np.ndarray, LogisticRegression]] = {}

    def fit(self, probs: np.ndarray, y_idx: Sequence[int]) -> "FamilyExperts":
        y_idx = np.asarray(y_idx, dtype=int)
        feats = _stack_features(probs)
        for family, idx in family_masks():
            mask = np.isin(y_idx, idx)
            if mask.sum() < self.min_rows:
                continue
            local = {label: i for i, label in enumerate(idx)}
            y_local = np.asarray([local[v] for v in y_idx[mask]])
            clf = LogisticRegression(C=self.C, max_iter=1000)
            clf.fit(feats[mask], y_local)
            self.experts[family] = (np.asarray(idx), clf)
        return self

    def apply(self, probs: np.ndarray) -> np.ndarray:
        if not self.experts:
            return probs
        p = np.asarray(probs, dtype=np.float64).copy()
        pred = p.argmax(axis=1)
        feats = _stack_features(p)

        for family, (idx, clf) in self.experts.items():
            member_mask = np.zeros(N_CLASSES, dtype=bool)
            member_mask[idx] = True
            rows = np.flatnonzero(member_mask[pred])
            if not len(rows):
                continue
            q_local = clf.predict_proba(feats[rows])
            q = np.zeros((len(rows), N_CLASSES))
            for col, global_idx in enumerate(idx[: q_local.shape[1]]):
                q[:, global_idx] = q_local[:, col]
            q = np.clip(q, EPS, None)

            block = p[np.ix_(rows, idx)]
            mass = block.sum(axis=1, keepdims=True)
            cond = block / np.maximum(mass, EPS)
            log_cond = np.log(np.clip(cond, EPS, 1.0))
            log_expert = np.log(q[:, idx])
            log_expert -= logsumexp(log_expert, axis=1, keepdims=True)

            if self.adaptive_rho:
                ranked = np.sort(cond, axis=1)
                ratio = ranked[:, -2] / np.maximum(ranked[:, -1], EPS)
                rho = np.clip(0.25 + 0.75 * ratio, 0.25, 1.0)
            else:
                rho = np.ones(len(rows))
            combined = log_cond + rho[:, None] * log_expert
            combined -= logsumexp(combined, axis=1, keepdims=True)
            p[np.ix_(rows, idx)] = mass * np.exp(combined)
        return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)
