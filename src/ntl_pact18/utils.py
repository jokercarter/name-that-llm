"""Small shared helpers: seeding, IO, numerical utilities."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

EPS = 1e-12


def seed_everything(seed: int = 42) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def timer(label: str, verbose: bool = True):
    start = time.perf_counter()
    holder: Dict[str, float] = {}
    try:
        yield holder
    finally:
        elapsed = time.perf_counter() - start
        holder["seconds"] = elapsed
        if verbose:
            print(f"[time] {label}: {elapsed:.2f}s")


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), EPS)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.maximum(np.exp(z).sum(axis=axis, keepdims=True), EPS))


def normalize_rows(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    bad = s[:, 0] <= 0
    if np.any(bad):
        p[bad] = 1.0 / p.shape[1]
        s = p.sum(axis=1, keepdims=True)
    return p / s


def one_hot(labels: Sequence[int], n_classes: int) -> np.ndarray:
    y = np.zeros((len(labels), n_classes), dtype=np.float64)
    y[np.arange(len(labels)), np.asarray(labels, dtype=int)] = 1.0
    return y


def encode_groups(values: Sequence[Any]) -> np.ndarray:
    """Map arbitrary group keys to contiguous integer ids (first-seen order)."""
    mapping: Dict[Any, int] = {}
    out = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(mapping[value])
    return np.asarray(out, dtype=int)


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fmt_table(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    """Render a list of dicts as a plain-text table (no external deps)."""
    if not rows:
        return "(empty)"
    columns = columns or list(rows[0].keys())
    widths = {c: len(str(c)) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(f"{row.get(c, '')}"))
    header = "  ".join(str(c).ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    lines = [header, sep]
    for row in rows:
        lines.append("  ".join(f"{row.get(c, '')}".ljust(widths[c]) for c in columns))
    return "\n".join(lines)
