"""Prompt clustering: a TF-IDF baseline and a learned diagonal-metric clusterer.

Both clusterers work transductively on the texts they are given (the test pool),
which is exactly how the public notebooks infer prompt groups. The learned
variant fits a diagonal metric over n-gram features from same-prompt /
different-prompt pairs, so the notion of "same prompt" is trained rather than
hand-tuned. It is a lightweight stand-in for a bi-encoder that needs no torch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, hstack, issparse, vstack
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .utils import EPS


def _cluster_from_distance(dist: np.ndarray, threshold: float) -> np.ndarray:
    n = dist.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    d = np.clip(dist, 0.0, None)
    np.fill_diagonal(d, 0.0)
    if n > 2:
        upper = np.triu_indices(n, 1)
        median = float(np.median(d[upper]))
        if median > 0:
            d = d / median
    np.fill_diagonal(d, 0.0)
    condensed = squareform(d, checks=False)
    links = linkage(condensed, method="average")
    return fcluster(links, float(threshold), criterion="distance")


def _pair_f1(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float]:
    n = len(pred)
    if n < 2:
        return 1.0, 1.0, 1.0
    same_pred = pred[:, None] == pred[None, :]
    same_true = true[:, None] == true[None, :]
    iu = np.triu_indices(n, 1)
    sp, st = same_pred[iu], same_true[iu]
    tp = float(np.sum(sp & st))
    precision = tp / max(float(np.sum(sp)), 1.0)
    recall = tp / max(float(np.sum(st)), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _calibrate_threshold(sim: np.ndarray, groups: np.ndarray,
                         max_size: int = 18) -> float:
    """Pick the linkage cut that maximises pair-F1, penalising oversized clusters."""
    d = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(d, 0.0)
    n = d.shape[0]
    if n > 2:
        upper = np.triu_indices(n, 1)
        median = float(np.median(d[upper]))
        if median > 0:
            d = d / median
    np.fill_diagonal(d, 0.0)
    links = linkage(squareform(d, checks=False), method="average")

    best_score, best_t = -1.0, 0.9
    for t in np.linspace(0.05, 1.2, 47):
        pred = fcluster(links, float(t), criterion="distance")
        _, _, f1 = _pair_f1(pred, groups)
        sizes = np.bincount(pred)[1:]
        oversized = float(sizes[sizes > max_size].sum()) / max(n, 1)
        score = f1 - 0.5 * oversized
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


class TfidfClusterer:
    """Median-normalised content-word TF-IDF cosine distance (public baseline)."""

    name = "tfidf"

    def __init__(self, threshold: float = 0.90, min_size: int = 2, max_size: int = 18,
                 seed: int = 42):
        self.threshold = float(threshold)
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.seed = int(seed)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.threshold_: Optional[float] = None
        self.calibration_folds_: List[float] = []

    def fit(self, texts: Sequence[str], groups: Optional[Sequence[int]] = None) -> "TfidfClusterer":
        texts = list(texts)
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
        self.vectorizer.fit(texts)
        self.calibration_folds_ = []
        if groups is not None and len(np.unique(groups)) > 1:
            groups = np.asarray(groups)
            X = self.vectorizer.transform(texts)
            uniq = np.unique(groups)
            rng = np.random.default_rng(self.seed)
            if len(uniq) >= 6:
                order = uniq[rng.permutation(len(uniq))]
                for k in range(3):
                    held = order[k::3]
                    mask = np.isin(groups, held)
                    if mask.sum() < 2:
                        continue
                    sim = np.clip((X[mask] @ X[mask].T).toarray(), 0.0, 1.0)
                    self.calibration_folds_.append(
                        _calibrate_threshold(sim, groups[mask], max_size=self.max_size)
                    )
            if self.calibration_folds_:
                self.threshold_ = float(np.median(self.calibration_folds_))
            else:
                sim = np.clip((X @ X.T).toarray(), 0.0, 1.0)
                self.threshold_ = _calibrate_threshold(sim, groups, max_size=self.max_size)
        return self

    def similarity(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
            self.vectorizer.fit(list(texts))
        X = self.vectorizer.transform(list(texts))
        sim = (X @ X.T).toarray()
        return np.clip(sim, 0.0, 1.0)

    def predict_groups(self, texts: Sequence[str]) -> np.ndarray:
        sim = self.similarity(texts)
        threshold = self.threshold_ if self.threshold_ is not None else self.threshold
        return _cluster_from_distance(1.0 - sim, threshold)


class LearnedPairClusterer:
    """Diagonal metric learned from same-prompt / different-prompt pairs."""

    name = "learned"

    def __init__(self, threshold: float = 0.90, min_size: int = 2, max_size: int = 18,
                 n_pairs: int = 40_000, C: float = 1.0, seed: int = 42):
        self.threshold = float(threshold)
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.n_pairs = int(n_pairs)
        self.C = float(C)
        self.seed = int(seed)
        self.word_vec: Optional[TfidfVectorizer] = None
        self.char_vec: Optional[TfidfVectorizer] = None
        self.metric: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self._fitted = False

    # ------------------------------------------------------------- internals
    def _vectorize(self, texts: Sequence[str]) -> csr_matrix:
        w = self.word_vec.transform(list(texts))
        c = self.char_vec.transform(list(texts))
        X = hstack([w, c], format="csr")
        return _l2_normalize(X)

    def _pair_features(self, X: csr_matrix, pairs: np.ndarray) -> csr_matrix:
        rows = [X[i].multiply(X[j]) for i, j in pairs]
        return vstack(rows, format="csr")

    # ------------------------------------------------------------------- API
    def _fit_metric(self, X: csr_matrix, groups: np.ndarray, rows: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
        """Fit the diagonal metric on the pairs that live inside ``rows``."""
        rows = np.asarray(rows)
        by_group = {int(g): rows[groups[rows] == g] for g in np.unique(groups[rows])}
        per_group = max(1, self.n_pairs // max(len(by_group), 1) // 2)

        pos_pairs = []
        for _, idx in by_group.items():
            if len(idx) < 2:
                continue
            for _ in range(per_group):
                a, b = rng.choice(idx, size=2, replace=False)
                pos_pairs.append((int(a), int(b)))

        neg_pairs = []
        attempts = 0
        while len(neg_pairs) < len(pos_pairs) and attempts < 20 * max(len(pos_pairs), 1):
            attempts += 1
            a = int(rng.choice(rows))
            b = int(rng.choice(rows))
            if groups[a] != groups[b]:
                neg_pairs.append((a, b))

        if not pos_pairs or not neg_pairs:
            return np.ones(X.shape[1], dtype=np.float64)

        pairs = np.asarray(pos_pairs + neg_pairs)
        labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])
        clf = LogisticRegression(C=self.C, max_iter=1000, solver="liblinear")
        clf.fit(self._pair_features(X, pairs), labels)
        return clf.coef_.ravel()

    def _sim_with(self, X: csr_matrix, metric: np.ndarray) -> np.ndarray:
        Z = X.multiply(metric[None, :]).tocsr()
        Z = _l2_normalize(Z)
        return np.clip((Z @ Z.T).toarray(), -1.0, 1.0)

    def fit(self, texts: Sequence[str], groups: Optional[Sequence[int]] = None) -> "LearnedPairClusterer":
        if groups is None:
            raise ValueError("LearnedPairClusterer needs prompt groups to fit")
        texts = list(texts)
        groups = np.asarray(groups)
        rng = np.random.default_rng(self.seed)

        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=20_000, sublinear_tf=True, min_df=1,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), max_features=30_000, sublinear_tf=True, min_df=1
        )
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        X = self._vectorize(texts)
        all_rows = np.arange(len(texts))

        self.metric = self._fit_metric(X, groups, all_rows, rng)
        self._fitted = True

        # The linkage cut must be calibrated on prompts the metric has *not*
        # seen: an in-sample threshold over-splits unseen prompts badly.
        uniq = np.unique(groups)
        fold_thresholds: List[float] = []
        if len(uniq) >= 6:
            order = uniq[rng.permutation(len(uniq))]
            for k in range(3):
                held = order[k::3]
                mask = np.isin(groups, held)
                if mask.sum() < 2 or (~mask).sum() < 2:
                    continue
                metric_fold = self._fit_metric(X, groups, np.flatnonzero(~mask), rng)
                sim_held = self._sim_with(X[np.flatnonzero(mask)], metric_fold)
                fold_thresholds.append(
                    _calibrate_threshold(sim_held, groups[mask], max_size=self.max_size)
                )
        self.calibration_folds_ = fold_thresholds
        if fold_thresholds:
            self.threshold_ = float(np.median(fold_thresholds))
        else:
            self.threshold_ = _calibrate_threshold(
                self._sim_with(X, self.metric), groups, max_size=self.max_size
            )
        return self

    def similarity(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LearnedPairClusterer.predict_groups called before fit")
        return self._sim_with(self._vectorize(list(texts)), self.metric)

    def predict_groups(self, texts: Sequence[str]) -> np.ndarray:
        sim = self.similarity(texts)
        threshold = self.threshold_ if self.threshold_ is not None else self.threshold
        return _cluster_from_distance(1.0 - sim, threshold)


def _l2_normalize(X: csr_matrix) -> csr_matrix:
    if not issparse(X):
        X = csr_matrix(X)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return X.multiply(1.0 / norms[:, None]).tocsr()


def build_clusterer(cfg) -> object:
    method = str(cfg.get("clustering.method", "learned")).lower()
    kwargs = dict(
        threshold=float(cfg.get("clustering.threshold", 0.90)),
        min_size=int(cfg.get("clustering.min_size", 2)),
        max_size=int(cfg.get("clustering.max_size", 18)),
    )
    if method == "tfidf":
        return TfidfClusterer(**kwargs)
    return LearnedPairClusterer(seed=int(cfg.get("project.seed", 42)), **kwargs)


def cluster_confidence(sim: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Mean within-cluster similarity per row (higher = purer cluster)."""
    groups = np.asarray(groups)
    conf = np.ones(len(groups), dtype=np.float64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) < 2:
            continue
        block = sim[np.ix_(idx, idx)]
        n = len(idx)
        mean_sim = (block.sum() - np.trace(block)) / max(n * (n - 1), 1)
        conf[idx] = mean_sim
    return conf
