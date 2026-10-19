"""Log-probability blending with weights fitted on prompt-grouped OOF predictions."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .utils import EPS, normalize_rows, softmax


def stack_log_probs(member_probs: Mapping[str, np.ndarray],
                    names: Sequence[str] | None = None) -> Tuple[np.ndarray, list]:
    names = list(names or member_probs.keys())
    logp = np.stack([np.log(np.clip(member_probs[n], EPS, None)) for n in names], axis=0)
    return logp, names


def blend_log_probs(member_probs: Mapping[str, np.ndarray],
                    weights: Mapping[str, float]) -> np.ndarray:
    names = list(weights.keys())
    logp, _ = stack_log_probs(member_probs, names)
    w = np.asarray([weights[n] for n in names], dtype=np.float64)
    w = w / max(w.sum(), EPS)
    return np.tensordot(w, logp, axes=(0, 0))


def blend_probs(member_probs: Mapping[str, np.ndarray],
                weights: Mapping[str, float]) -> np.ndarray:
    z = blend_log_probs(member_probs, weights)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)


def fit_blend_weights(member_probs: Mapping[str, np.ndarray], y_idx: np.ndarray,
                      l2: float = 1.0, max_iter: int = 300) -> Dict[str, float]:
    """Minimise NLL over a softmax-parameterised weight vector (L-BFGS)."""
    logp, names = stack_log_probs(member_probs)
    n_members, n_rows, _ = logp.shape
    y_idx = np.asarray(y_idx, dtype=int)
    rows = np.arange(n_rows)

    def objective(theta: np.ndarray) -> float:
        w = softmax(theta)
        z = np.tensordot(w, logp, axes=(0, 0))
        z = z - z.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(z).sum(axis=1))
        log_prob = z[rows, y_idx] - log_norm
        return float(-log_prob.mean() + l2 * float(np.sum(w**2)))

    init = np.zeros(n_members)
    result = minimize(objective, init, method="L-BFGS-B", options={"maxiter": max_iter})
    w = softmax(result.x)
    return {name: float(weight) for name, weight in zip(names, w)}


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: max(v, 0.0) / total for k, v in weights.items()}
