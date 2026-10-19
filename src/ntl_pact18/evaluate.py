"""Metrics and validation helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .calibration import expected_calibration_error
from .labels import LABELS, N_CLASSES


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    support = cm.sum(axis=1)
    hits = np.diag(cm)
    out = np.full(N_CLASSES, np.nan)
    nz = support > 0
    out[nz] = hits[nz] / support[nz]
    return out


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    recalls = per_class_recall(y_true, y_pred)
    return float(np.nanmean(recalls))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), np.nan)
    return float(np.nanmean(f1))


def top_k_accuracy(probs: np.ndarray, y_true: Sequence[int], k: int = 2) -> float:
    y_true = np.asarray(y_true, dtype=int)
    topk = np.argsort(-np.asarray(probs), axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def cluster_pair_scores(pred_groups: Sequence[int], true_groups: Sequence[int]) -> Tuple[float, float, float]:
    """Pairwise precision / recall / F1 of the inferred prompt clusters."""
    pred = np.asarray(pred_groups)
    true = np.asarray(true_groups)
    n = len(pred)
    if n < 2:
        return 1.0, 1.0, 1.0
    same_pred = pred[:, None] == pred[None, :]
    same_true = true[:, None] == true[None, :]
    iu = np.triu_indices(n, 1)
    sp = same_pred[iu]
    st = same_true[iu]
    tp = float(np.sum(sp & st))
    precision = tp / max(float(np.sum(sp)), 1.0)
    recall = tp / max(float(np.sum(st)), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def summarize(probs: Optional[np.ndarray], y_true: Sequence[int], y_pred: Sequence[int],
              pred_groups: Optional[Sequence[int]] = None,
              true_groups: Optional[Sequence[int]] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out: Dict[str, float] = {
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
    }
    if probs is not None:
        out["top2_accuracy"] = top_k_accuracy(probs, y_true, 2)
        out["ece"] = expected_calibration_error(np.asarray(probs), y_true)
    if pred_groups is not None and true_groups is not None:
        precision, recall, f1 = cluster_pair_scores(pred_groups, true_groups)
        out.update({"cluster_pair_precision": precision, "cluster_pair_recall": recall,
                    "cluster_pair_f1": f1})
    return out


def bootstrap_balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int],
                                groups: Sequence[int], n_boot: int = 1000,
                                seed: int = 42) -> Tuple[float, float, float]:
    """Cluster bootstrap over prompts: resample prompts, not passages."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    stats = []
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    for _ in range(int(n_boot)):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        stats.append(balanced_accuracy(y_true[idx], y_pred[idx]))
    stats = np.asarray(stats)
    return float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def per_class_report(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    recalls = per_class_recall(np.asarray(y_true), np.asarray(y_pred))
    support = confusion_matrix(np.asarray(y_true), np.asarray(y_pred)).sum(axis=1)
    lines = [f"{'class':28s} {'n':>5s} {'recall':>7s}"]
    for i, label in enumerate(LABELS):
        r = "  n/a" if np.isnan(recalls[i]) else f"{recalls[i]:.3f}"
        lines.append(f"{label:28s} {support[i]:5d} {r:>7s}")
    return "\n".join(lines)
