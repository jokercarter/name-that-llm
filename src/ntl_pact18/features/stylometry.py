"""Stylometric statistics + gradient boosting member.

The feature vector is deliberately hand-built and content-free: sentence and
word length distributions, punctuation rates, casing/digit/non-ASCII ratios and
function-word frequencies. It complements the n-gram member because it encodes
shape rather than vocabulary.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..labels import LABELS

PUNCT = list(',.;:!?()[]"\'-') + ["\u2014", "\u2013", "\u2018", "\u2019", "\u201c", "\u201d", "\u2026", "*", "/", "%"]
FUNC = sorted(ENGLISH_STOP_WORDS)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z\u2019']+")

FEATURE_NAMES: List[str] = (
    [
        "log_words", "n_sent", "sent_len_mean", "sent_len_std", "sent_len_max", "sent_len_min",
        "word_len_mean", "word_len_std", "long_word_ratio", "type_token_ratio",
        "upper_ratio", "digit_ratio", "nonascii_ratio",
        "comma_space_rate", "and_rate", "but_rate",
        "demonstrative_start_rate", "ly_rate", "ing_rate", "tion_rate",
    ]
    + [f"punct::{p}" for p in PUNCT]
    + [f"stop::{w}" for w in FUNC]
)


def stylometric_features(text: str) -> np.ndarray:
    words = WORD_RE.findall(text)
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)
    sents = [s for s in SENT_RE.split(text.strip()) if s]
    sent_lens = np.array([len(WORD_RE.findall(s)) for s in sents]) if sents else np.array([0])
    word_lens = np.array([len(w) for w in words]) if words else np.array([0])
    lowered = [w.lower() for w in words]

    feats = [
        np.log1p(n_words),
        float(len(sents)),
        float(sent_lens.mean()),
        float(sent_lens.std()),
        float(sent_lens.max()),
        float(sent_lens.min()),
        float(word_lens.mean()),
        float(word_lens.std()),
        float((word_lens >= 9).mean()),
        len(set(lowered)) / n_words,
        sum(c.isupper() for c in text) / n_chars,
        sum(c.isdigit() for c in text) / n_chars,
        sum(ord(c) > 127 for c in text) / n_chars,
        text.count(", ") / n_words,
        text.count(" and ") / n_words,
        text.count(" but ") / n_words,
        sum(
            1
            for s in sents
            if s[:1].isupper() and s.split()[0].lower() in ("this", "these", "it", "they")
        )
        / max(len(sents), 1),
        sum(1 for w in words if w.endswith("ly")) / n_words,
        sum(1 for w in words if w.endswith("ing")) / n_words,
        sum(1 for w in words if w.endswith("tion")) / n_words,
    ]
    feats.extend(text.count(q) / n_words for q in PUNCT)

    counts = {}
    for w in lowered:
        counts[w] = counts.get(w, 0) + 1
    feats.extend(counts.get(w, 0) / n_words for w in FUNC)
    return np.asarray(feats, dtype=np.float32)


def _make_estimator(backend: str, n_estimators: int, learning_rate: float,
                    max_leaf_nodes: int, use_gpu: bool = False):
    """Pick a booster. On a GPU machine XGBoost (device='cuda') is preferred,
    then LightGBM with device_type='gpu', then a CPU gradient booster."""
    backend = (backend or "auto").lower()

    if use_gpu and backend in {"auto", "xgboost"}:
        try:
            from xgboost import XGBClassifier

            return (
                XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=6,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.5,
                    reg_lambda=1.0,
                    tree_method="hist",
                    device="cuda",
                    random_state=0,
                    n_jobs=4,
                    verbosity=0,
                ),
                "xgboost-gpu",
            )
        except Exception:
            if backend == "xgboost":
                print("[stylometry] xgboost requested but unavailable, falling back")

    if backend in {"auto", "lightgbm"}:
        try:
            import lightgbm as lgb  # type: ignore

            params = dict(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=max_leaf_nodes,
                min_child_samples=10,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                reg_lambda=1.0,
                random_state=0,
                verbose=-1,
                n_jobs=4,
            )
            if use_gpu:
                # LightGBM GPU build required; falls back to CPU automatically
                # when the binary has no GPU support (it warns and continues).
                params["device_type"] = "gpu"
            return lgb.LGBMClassifier(**params), "lightgbm-gpu" if use_gpu else "lightgbm"
        except Exception:
            if backend == "lightgbm":
                print("[stylometry] lightgbm requested but unavailable, falling back to sklearn")

    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier

            return (
                XGBClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    tree_method="hist", random_state=0, n_jobs=4, verbosity=0,
                ),
                "xgboost-cpu",
            )
        except Exception:
            pass

    return (
        HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=1.0,
            random_state=0,
        ),
        "sklearn-hist",
    )


class StylometryMember:
    name = "stylometry"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 max_leaf_nodes: int = 15, backend: str = "auto", use_gpu: bool = False):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.backend = backend
        self.use_gpu = bool(use_gpu)
        self.model = None
        self.classes_: Optional[np.ndarray] = None
        self.backend_used = "unset"

    def _features(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([stylometric_features(t) for t in texts]) if len(texts) else np.zeros((0, len(FEATURE_NAMES)))

    def fit(self, texts: Sequence[str], y: Sequence[str],
            groups: Optional[Sequence[int]] = None) -> "StylometryMember":
        self.model, self.backend_used = _make_estimator(
            self.backend, self.n_estimators, self.learning_rate,
            self.max_leaf_nodes, use_gpu=self.use_gpu,
        )
        # XGBoost's sklearn wrapper insists on 0..K-1 integer labels, so encode
        # once here and decode by position in predict_proba.
        self.classes_ = np.unique(np.asarray(list(y)))
        index = {label: i for i, label in enumerate(self.classes_)}
        y_enc = np.asarray([index[v] for v in y], dtype=int)
        self.model.fit(self._features(texts), y_enc)
        return self

    def predict_proba(self, texts: Sequence[str],
                      groups: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("StylometryMember.predict_proba called before fit")
        raw = self.model.predict_proba(self._features(texts))
        out = np.full((len(texts), len(LABELS)), 1e-6)
        for col, label in enumerate(self.classes_):
            out[:, LABELS.index(str(label))] = raw[:, col]
        return out / out.sum(axis=1, keepdims=True)
