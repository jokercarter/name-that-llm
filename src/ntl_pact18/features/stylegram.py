"""Style-aware n-gram member (the strongest single member in the public stack).

Four TF-IDF views are concatenated:

* ``char``     3-5 character n-grams       -> orthography, morphology
* ``word``     1-3 word n-grams + punct    -> phrasing, collocations
* ``shape``    Xxx/xxx case-shape n-grams  -> capitalisation habits, no content
* ``function`` content words -> ``CONTENT`` -> function-word skeleton, no topic

A prompt-centred "twin" classifier can be fitted alongside the plain one. At
inference, rows that belong to an inferred prompt cluster of size
``[min_size, max_size]`` use the centred twin, which subtracts the cluster mean
and therefore removes the shared topic signal.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from ..labels import LABELS
from ..utils import EPS, softmax

FUNCTION_WORDS = set(
    """a an the and or but if while although because as so than that which who whom whose
    this these those it its they them their theirs he him his she her hers we us our ours
    you your yours i me my mine in on at to from with without for of by over under between
    through into about is are was were be been being have has had do does did can could
    will would shall should may might must not no nor all any some each every both either
    neither other another such only also just very too more most less least here there when
    where why how what whether then now still yet even much many few little""".split()
)

_DEFAULT_SPEC: Dict[str, Dict] = {
    "char": dict(ngram=(3, 5), max_features=120_000, weight=1.0),
    "word": dict(ngram=(1, 3), max_features=90_000, weight=0.7),
    "shape": dict(ngram=(2, 6), max_features=30_000, weight=0.5),
    "function": dict(ngram=(1, 3), max_features=25_000, weight=0.5),
}


def shape_text(text: str) -> str:
    """Replace each word by its case shape: ``The`` -> ``Xxx``, ``cat`` -> ``xxx``."""
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_text(text: str) -> str:
    """Keep function words and punctuation, collapse content words to CONTENT."""
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return " ".join(
        t.lower() if (t.lower() in FUNCTION_WORDS or not t.isalnum()) else "CONTENT"
        for t in tokens
    )


def _center_sparse(X, groups: Sequence[int], min_size: int, max_size: int):
    """Subtract the per-group mean from eligible rows; leave the rest untouched."""
    X = csr_matrix(X)
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    valid = np.flatnonzero((counts >= min_size) & (counts <= max_size))
    if not len(valid):
        return X.copy()
    remap = np.full(len(counts), -1, dtype=np.int32)
    remap[valid] = np.arange(len(valid), dtype=np.int32)
    mapped = remap[inverse]
    rows = np.flatnonzero(mapped >= 0)
    if not len(rows):
        return X.copy()
    membership = csr_matrix(
        (np.ones(len(rows), dtype=X.dtype), (rows, mapped[rows])),
        shape=(X.shape[0], len(valid)),
        dtype=X.dtype,
    )
    means = diags((1.0 / counts[valid]).astype(X.dtype)) @ (membership.T @ X)
    return (X - membership @ means).tocsr()


class StylegramMember:
    name = "stylegram"

    def __init__(self, spec: Optional[Dict] = None, C: float = 0.5,
                 center_mode: str = "replace", min_size: int = 2, max_size: int = 18):
        self.spec = spec or _DEFAULT_SPEC
        self.C = float(C)
        self.center_mode = center_mode
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.classes_: Optional[np.ndarray] = None
        self.svc_plain: Optional[LinearSVC] = None
        self.svc_center: Optional[LinearSVC] = None
        self._n_features = 0

    # ------------------------------------------------------------- internals
    def _transform(self, texts: Sequence[str]) -> csr_matrix:
        blocks = []
        for key, vec in self.vectorizers.items():
            weight = float(self.spec[key]["weight"])
            blocks.append(vec.transform([self._view(key, t) for t in texts]) * weight)
        return hstack(blocks, format="csr")

    @staticmethod
    def _view(key: str, text: str) -> str:
        if key == "shape":
            return shape_text(text)
        if key == "function":
            return function_text(text)
        return text

    def _build_vectorizers(self, texts: Sequence[str]) -> None:
        for key, conf in self.spec.items():
            analyzer = "char" if key == "shape" else "word" if key == "function" else key
            opts = dict(
                analyzer=analyzer,
                ngram_range=tuple(conf["ngram"]),
                lowercase=False,
                sublinear_tf=True,
                min_df=2,
                max_features=int(conf["max_features"]),
                dtype=np.float64,
            )
            if analyzer == "word":
                opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
            vec = TfidfVectorizer(**opts)
            vec.fit([self._view(key, t) for t in texts])
            self.vectorizers[key] = vec
        self._n_features = int(sum(len(v.vocabulary_) for v in self.vectorizers.values()))

    def _probs(self, model: LinearSVC, X) -> np.ndarray:
        decision = model.decision_function(X)
        if decision.ndim == 1:  # binary edge case
            decision = np.column_stack([-decision, decision])
        probs = softmax(decision)
        out = np.full((X.shape[0], len(LABELS)), 1.0 / len(LABELS))
        for col, label in enumerate(self.classes_):
            out[:, LABELS.index(str(label))] = probs[:, col]
        return out / out.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------- API
    def fit(self, texts: Sequence[str], y: Sequence[str],
            groups: Optional[Sequence[int]] = None) -> "StylegramMember":
        self._build_vectorizers(texts)
        X = self._transform(texts)
        self.svc_plain = LinearSVC(C=self.C, random_state=42, max_iter=4000)
        self.svc_plain.fit(X, list(y))
        self.classes_ = self.svc_plain.classes_

        if self.center_mode != "off" and groups is not None:
            Xc = _center_sparse(X, groups, self.min_size, self.max_size)
            if Xc.nnz:
                self.svc_center = LinearSVC(C=self.C, random_state=42, max_iter=4000)
                self.svc_center.fit(Xc, list(y))
        return self

    def predict_proba(self, texts: Sequence[str],
                      groups: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.svc_plain is None:
            raise RuntimeError("StylegramMember.predict_proba called before fit")
        X = self._transform(texts)
        plain = self._probs(self.svc_plain, X)

        if self.svc_center is None or groups is None or self.center_mode == "off":
            return plain

        groups = np.asarray(groups)
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        eligible_groups = np.flatnonzero((counts >= self.min_size) & (counts <= self.max_size))
        if not len(eligible_groups):
            return plain
        eligible_rows = np.isin(inverse, eligible_groups)
        if not eligible_rows.any():
            return plain

        Xc = _center_sparse(X, groups, self.min_size, self.max_size)
        centered = self._probs(self.svc_center, Xc)
        out = plain.copy()
        if self.center_mode == "replace":
            out[eligible_rows] = centered[eligible_rows]
        else:  # add
            out[eligible_rows] = 0.5 * (plain[eligible_rows] + centered[eligible_rows])
        return out / np.maximum(out.sum(axis=1, keepdims=True), EPS)

    @property
    def n_features(self) -> int:
        return self._n_features
