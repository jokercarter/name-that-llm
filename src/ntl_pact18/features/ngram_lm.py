"""LLMDet-flavoured proxy-perplexity member.

LLMDet (Wu et al., EMNLP 2023 Findings) attributes text by comparing its
next-token likelihood under a set of reference models. We cannot query the 18
closed models, but we *can* fit a cheap n-gram language model per class on the
training passages and use the vector of per-class cross-entropies as features.

The member therefore exposes a 21-dimensional signal (18 likelihoods + rank,
margin and entropy) which is orthogonal to both TF-IDF and stylometry, and a
multinomial logistic head turns it into calibrated class probabilities.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..labels import LABELS
from ..utils import softmax

TOKEN_RE = re.compile(r"[a-z']+|[.,;:!?—–\"'()]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class _NgramLM:
    """Add-alpha smoothed unigram + bigram model with a fixed vocabulary."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.uni: Dict[str, float] = {}
        self.bi: Dict[Tuple[str, str], float] = {}
        self.prev_count: Dict[str, float] = {}
        self.unk_uni = -20.0
        self.unk_bi: Dict[str, float] = {}

    def fit(self, docs: Sequence[str]) -> "_NgramLM":
        uni_counts: Counter = Counter()
        bi_counts: Counter = Counter()
        prev_counts: Counter = Counter()
        total = 0
        for doc in docs:
            toks = ["<s>"] + tokenize(doc) + ["</s>"]
            for i, tok in enumerate(toks):
                if i:
                    uni_counts[tok] += 1
                    total += 1
                if i + 1 < len(toks):
                    pair = (toks[i], toks[i + 1])
                    bi_counts[pair] += 1
                    prev_counts[toks[i]] += 1
        vocab = set(uni_counts) | {"<unk>"}
        V = len(vocab)
        denom_uni = total + self.alpha * V
        self.unk_uni = math.log(self.alpha / denom_uni)
        self.uni = {w: math.log((c + self.alpha) / denom_uni) for w, c in uni_counts.items()}
        for (a, b), c in bi_counts.items():
            denom = prev_counts[a] + self.alpha * V
            self.bi[(a, b)] = math.log((c + self.alpha) / denom)
        for a, c in prev_counts.items():
            self.unk_bi[a] = math.log(self.alpha / (c + self.alpha * V))
        self._default_bi = math.log(self.alpha / (self.alpha * V))
        return self

    def logprob(self, text: str) -> float:
        toks = ["<s>"] + tokenize(text) + ["</s>"]
        total = 0.0
        n = 0
        for i in range(1, len(toks)):
            tok = toks[i]
            prev = toks[i - 1]
            total += self.uni.get(tok, self.unk_uni)
            total += self.bi.get((prev, tok), self.unk_bi.get(prev, self._default_bi))
            n += 1
        return total / max(n, 1)


class NgramLMMember:
    name = "ngram_lm"

    def __init__(self, order: int = 2, alpha: float = 0.1, C: float = 1.0):
        self.order = int(order)
        self.alpha = float(alpha)
        self.C = float(C)
        self.lms: Dict[str, _NgramLM] = {}
        self.head: Optional[LogisticRegression] = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------- internals
    def _raw_scores(self, texts: Sequence[str]) -> np.ndarray:
        scores = np.zeros((len(texts), len(LABELS)))
        for j, label in enumerate(LABELS):
            lm = self.lms.get(label)
            if lm is None:
                scores[:, j] = -20.0
                continue
            for i, text in enumerate(texts):
                scores[i, j] = lm.logprob(text)
        return scores

    def _features(self, scores: np.ndarray) -> np.ndarray:
        ranked = np.sort(scores, axis=1)
        margin = ranked[:, -1] - ranked[:, -2]
        temperature = np.maximum(scores.std(axis=1), 1e-3)
        ent = -(softmax(scores / temperature[:, None]) *
                np.log(np.maximum(softmax(scores / temperature[:, None]), 1e-12))).sum(1)
        return np.column_stack([scores, margin, ent, temperature])

    # ------------------------------------------------------------------- API
    def fit(self, texts: Sequence[str], y: Sequence[str],
            groups: Optional[Sequence[int]] = None) -> "NgramLMMember":
        labels = np.asarray(list(y))
        self.lms = {}
        for label in LABELS:
            docs = [t for t, yy in zip(texts, labels) if yy == label]
            if docs:
                self.lms[label] = _NgramLM(self.alpha).fit(docs)
        feats = self._features(self._raw_scores(texts))
        self.head = LogisticRegression(C=self.C, max_iter=2000)
        self.head.fit(feats, list(y))
        self.classes_ = np.asarray(self.head.classes_)
        return self

    def predict_proba(self, texts: Sequence[str],
                      groups: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("NgramLMMember.predict_proba called before fit")
        raw = self.head.predict_proba(self._features(self._raw_scores(texts)))
        out = np.full((len(texts), len(LABELS)), 1e-6)
        for col, label in enumerate(self.classes_):
            out[:, LABELS.index(str(label))] = raw[:, col]
        return out / out.sum(axis=1, keepdims=True)
