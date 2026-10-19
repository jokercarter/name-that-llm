"""Post-hoc calibration: temperature scaling, Sinkhorn balancing, SLD-EM priors."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from .utils import EPS, normalize_rows

# --------------------------------------------------------------------- temperature


def _nll(probs: np.ndarray, y_idx: np.ndarray) -> float:
    picked = probs[np.arange(len(y_idx)), y_idx]
    return float(-np.log(np.clip(picked, EPS, 1.0)).mean())


def fit_temperature(probs: np.ndarray, y_idx: np.ndarray,
                    bounds: Tuple[float, float] = (0.05, 20.0)) -> float:
    """Find T minimising NLL of ``probs ** (1/T)`` (identity when T == 1)."""

    def objective(log_t: float) -> float:
        t = float(np.exp(log_t))
        return _nll(apply_temperature(probs, t), y_idx)

    result = minimize_scalar(
        objective,
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
        options={"xatol": 1e-3},
    )
    return float(np.exp(result.x))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-3)
    if abs(t - 1.0) < 1e-6:
        return normalize_rows(probs)
    powered = np.power(np.clip(probs, EPS, None), 1.0 / t)
    return normalize_rows(powered)


# ------------------------------------------------------------------------ Sinkhorn


def sinkhorn_balance(probs: np.ndarray, alpha: float = 0.25, iters: int = 50) -> np.ndarray:
    """Partially rescale column mass toward uniform class priors.

    ``alpha=0`` is a no-op, ``alpha=1`` targets a fully uniform marginal.
    """
    if alpha <= 0 or len(probs) < 2:
        return normalize_rows(probs)
    p = normalize_rows(probs).copy()
    n, k = p.shape
    target = (1.0 - alpha) * p.sum(axis=0) + alpha * np.full(k, n / k)
    for _ in range(iters):
        p *= (target / np.maximum(p.sum(axis=0), EPS))[None, :]
        p = normalize_rows(p)
    return p


# ---------------------------------------------------------------------- SLD / EM


def sld_em(probs: np.ndarray, pi_train: Optional[np.ndarray] = None,
           iters: int = 50, tol: float = 1e-6, damping: float = 0.5,
           clip_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Saerens-Latinne-Decaestecker EM estimate of the test class prior.

    Returns ``(adjusted_probs, pi_hat)``. Two safeguards keep the estimate from
    collapsing onto a few classes when the base model is miscalibrated:

    * ``damping``   -- geometric-style averaging with the previous estimate;
    * ``clip_ratio`` -- the estimate is clipped to ``[1/r, r] x uniform``.

    Falls back to the input when the iteration produces non-finite values.
    """
    p = normalize_rows(probs)
    n, k = p.shape
    uniform = 1.0 / k
    pi_train = np.full(k, uniform) if pi_train is None else np.asarray(pi_train, dtype=np.float64)
    pi_train = np.maximum(pi_train, 1e-9)
    pi_train = pi_train / pi_train.sum()
    pi = pi_train.copy()

    lo, hi = uniform / max(clip_ratio, 1.0), uniform * max(clip_ratio, 1.0)

    for _ in range(int(iters)):
        q = normalize_rows(p * (pi / pi_train)[None, :])
        pi_new = q.mean(axis=0)
        if not np.all(np.isfinite(pi_new)):
            return p, pi_train
        pi_new = (1.0 - damping) * pi + damping * pi_new
        pi_new = np.clip(pi_new, lo, hi)
        pi_new = pi_new / pi_new.sum()
        if np.abs(pi_new - pi).max() < tol:
            pi = pi_new
            break
        pi = pi_new

    q = normalize_rows(p * (pi / pi_train)[None, :])
    return q, pi


# ---------------------------------------------------------------------------- ECE


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        ece += gap * mask.mean()
    return float(ece)
