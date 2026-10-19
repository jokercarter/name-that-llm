"""Dataset loading, validation and the synthetic corpus generator.

The generator exists so the whole pipeline can be exercised end to end without
the Kaggle download (which needs credentials). It reproduces the structural
properties of the real competition data:

* a ``prompt_id`` grid where every (prompt, model) cell appears at most once,
* unseen prompts at test time,
* partial coverage in the test split,
* provider-specific orthographic invariants (curly quotes / em-dashes) that the
  orthographic shield can discover,
* passage lengths in the 40-250 word range.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .labels import FAMILY_OF_LABEL, LABELS, validate_labels
from .utils import ensure_dir

TOPIC_VOCAB: Dict[str, List[str]] = {
    "philosophy": """consciousness experience meaning ethics virtue reason agency
        identity perception truth knowledge freedom intention value judgment
        wisdom habit practice tradition language thought existence essence
        paradox dialectic intuition reflection discourse norm obligation
        autonomy authenticity narrative self other mind world""".split(),
    "physics": """energy momentum particle field symmetry entropy quantum
        relativity vacuum photon lattice excitation spectrum amplitude
        interference coherence decoherence measurement observer spacetime
        curvature gravity orbit plasma conduction radiation flux""".split(),
    "mathematics": """theorem lemma proof bound manifold topology group ring
        algebra spectrum operator eigenvalue convergence compactness measure
        integral derivative lattice duality morphism functor category
        conjecture counterexample induction recursion""".split(),
    "biology": """cell membrane protein enzyme genome transcription neuron
        synapse metabolism organism evolution selection adaptation ecology
        microbiome immunity receptor pathway signal tissue differentiation
        homeostasis population lineage mutation""".split(),
    "computer_science": """algorithm complexity cache latency compiler parser
        graph traversal recursion memory allocation concurrency thread lock
        protocol packet kernel scheduler hash index query transaction
        replication consistency distributed benchmark""".split(),
    "creative": """lantern harbour whisper gravel orchard copper twilight
        sparrow cellar velvet thunder meadow cipher ember willow mirror
        threshold voyage orchard silence lantern drift""".split(),
}

TOPICS: List[str] = list(TOPIC_VOCAB.keys())

# Every prompt gets its own handful of rare content words. All 18 models answer
# the same prompt, so these words are shared *within* a prompt and unique
# *across* prompts: they carry zero information about the author, which is
# exactly what makes prompt clustering non-trivial but solvable.
_SYL_A = ["ka", "lo", "mi", "ra", "sen", "tor", "vel", "nu", "dra", "phi", "qua", "zel"]
_SYL_B = ["nis", "tor", "vel", "mar", "dun", "pex", "ril", "son", "tia", "vor"]
_SYL_C = ["ak", "en", "is", "or", "ul", "eth", "ix", "om"]
_SYL_D = ["", "a", "e", "i", "o", "u"]
PROMPT_POOL: List[str] = [a + b + c + d for a in _SYL_A for b in _SYL_B for c in _SYL_C for d in _SYL_D]
PROMPT_WORDS_PER_PROMPT = 12
TOPIC_WORDS_PER_PROMPT = 5

CONNECTIVES = [
    "however", "moreover", "therefore", "indeed", "in fact", "that said",
    "notably", "ultimately", "by contrast", "in practice", "on balance",
    "crucially",
]
HEDGES = ["perhaps", "arguably", "clearly", "importantly", "roughly", "precisely"]

# Style is shared *within* a family and only jittered *across* siblings. That
# mirrors the real task: telling a GPT from a Claude is easy, telling
# gpt-5.6-sol from gpt-5.6-terra is the hard part.
FAMILY_STYLE: Dict[str, Dict] = {
    "openai": dict(mu=15.5, comma=.085, semi=.010, colon=.012, dash=.020, paren=.010,
                   excl=.002, curly=.030, long=.62, conn=0, hedge=2),
    "google": dict(mu=12.5, comma=.065, semi=.004, colon=.008, dash=.0005, paren=.006,
                   excl=.006, curly=.012, long=.48, conn=5, hedge=4),
    "anthropic": dict(mu=20.0, comma=.110, semi=.020, colon=.014, dash=.004, paren=.016,
                      excl=.001, curly=.000, long=.70, conn=9, hedge=0),
    "qwen": dict(mu=15.5, comma=.082, semi=.009, colon=.011, dash=.030, paren=.011,
                 excl=.004, curly=.028, long=.56, conn=2, hedge=5),
    "zai": dict(mu=15.0, comma=.088, semi=.011, colon=.013, dash=.010, paren=.012,
                excl=.003, curly=.000, long=.60, conn=3, hedge=0),
    "moonshot": dict(mu=19.5, comma=.100, semi=.015, colon=.015, dash=.022, paren=.017,
                     excl=.002, curly=.022, long=.68, conn=5, hedge=2),
}

# per-sibling jitter: a small length offset, a length preference shift and a
# rare signature phrase. Everything else is inherited from the family.
SIBLING_JITTER: Dict[str, Dict] = {
    "gpt-5.5": dict(dmu=+0.6, dlong=+0.02, sig="the key point is"),
    "gpt-5.6-luna": dict(dmu=-0.6, dlong=-0.04, sig="in short"),
    "gpt-5.6-sol": dict(dmu=+1.1, dlong=+0.05, sig="worth noting"),
    "gpt-5.6-terra": dict(dmu=-1.0, dlong=-0.06, sig="a useful frame"),
    "gpt-6-astra": dict(dmu=+1.6, dlong=+0.07, sig="consider first"),
    "gemini-3.1-pro-preview": dict(dmu=+0.8, dlong=+0.04, sig="put simply"),
    "gemini-3.6-flash": dict(dmu=-0.9, dlong=-0.05, sig="quick take"),
    "gemini-3.7-flash": dict(dmu=-0.2, dlong=-0.01, sig="bottom line"),
    "gemini-3.8-flash": dict(dmu=-1.4, dlong=-0.07, sig="in brief"),
    "claude-fable-5": dict(dmu=-0.5, dlong=-0.03, sig="it is worth saying"),
    "claude-opus-4-6": dict(dmu=+0.3, dlong=+0.01, sig="more precisely"),
    "claude-opus-4-8": dict(dmu=+0.9, dlong=+0.04, sig="the deeper issue"),
    "claude-opus-5": dict(dmu=+1.4, dlong=+0.06, sig="what matters here"),
    "claude-sonnet-5": dict(dmu=-0.8, dlong=-0.05, sig="in other words"),
    "qwen3.8-max": dict(dmu=0.0, dlong=0.0, sig="first, note"),
    "glm-5.3": dict(dmu=+0.7, dlong=+0.03, sig="to be clear"),
    "glm-5.3-flash": dict(dmu=-0.7, dlong=-0.04, sig="in essence"),
    "kimi-k3": dict(dmu=0.0, dlong=0.0, sig="note that"),
}

SIGNATURE_RATE = 0.06


def _model_style(model: str) -> Dict:
    family = FAMILY_OF_LABEL[model]
    base = dict(FAMILY_STYLE[family])
    jitter = SIBLING_JITTER[model]
    base["mu"] = base["mu"] + jitter["dmu"]
    base["long"] = float(np.clip(base["long"] + jitter["dlong"], 0.05, 0.95))
    base["sig"] = jitter["sig"]
    return base


MODEL_STYLE: Dict[str, Dict] = {model: _model_style(model) for model in LABELS}

_WORD_RE = re.compile(r"[A-Za-z']+")


def _pick_words(rng: np.random.Generator, vocab: List[str], n: int, long_pref: float) -> List[str]:
    ordered = sorted(vocab, key=len)
    ranks = np.linspace(-0.5, 0.5, len(ordered))
    logits = (long_pref - 0.5) * 6.0 * ranks
    p = np.exp(logits)
    p = p / p.sum()
    idx = rng.choice(len(ordered), size=n, replace=True, p=p)
    return [ordered[i] for i in idx]


def _capitalize(word: str) -> str:
    return word[:1].upper() + word[1:] if word else word


def _sentence(rng: np.random.Generator, style: Dict, vocab: List[str]) -> str:
    length = int(np.clip(rng.normal(style["mu"], 1.8), 6, 36))
    words = _pick_words(rng, vocab, length, style["long"])

    if rng.random() < 0.35:
        words.insert(0, CONNECTIVES[style["conn"]])
    if rng.random() < 0.25 and len(words) > 2:
        words.insert(1, HEDGES[style["hedge"]])
    if rng.random() < SIGNATURE_RATE:
        words[0:0] = style["sig"].split()

    # punctuation is attached to words, then joined with spaces
    pieces: List[str] = []
    for i, w in enumerate(words):
        suffix = ""
        if i < len(words) - 1:
            if rng.random() < style["semi"]:
                suffix = ";"
            elif rng.random() < style["comma"]:
                suffix = ","
            elif i == 0 and rng.random() < style["colon"]:
                suffix = ":"
        if i > 0 and rng.random() < style["dash"]:
            pieces.append("—")
        pieces.append(w + suffix)

    if rng.random() < style["paren"] and len(pieces) > 6:
        start = int(rng.integers(1, max(2, len(pieces) - 3)))
        end = min(len(pieces) - 1, start + int(rng.integers(2, 4)))
        pieces[start] = "(" + pieces[start]
        pieces[end] = pieces[end] + ")"

    if rng.random() < style["curly"] and len(pieces) > 5:
        start = int(rng.integers(1, max(2, len(pieces) - 3)))
        end = min(len(pieces) - 1, start + int(rng.integers(1, 3)))
        pieces[start] = "\u201c" + pieces[start]
        pieces[end] = pieces[end] + "\u201d"

    text = " ".join(pieces)
    text = _capitalize(text)
    text += "!" if rng.random() < style["excl"] else "."
    return text


def generate_passage(rng: np.random.Generator, model: str, vocab: List[str],
                     min_words: int = 40, max_words: int = 250) -> str:
    style = MODEL_STYLE[model]
    sentences: List[str] = []
    words = 0
    while words < min_words or (words < max_words and rng.random() < 0.55):
        s = _sentence(rng, style, vocab)
        sentences.append(s)
        words += len(_WORD_RE.findall(s))
        if words >= max_words or len(sentences) >= 18:
            break
    text = " ".join(sentences)
    toks = text.split()
    if len(toks) > max_words:
        text = " ".join(toks[:max_words])
        if not text.endswith((".", "!", "?")):
            text += "."
    return text


def make_synthetic_dataset(
    out_dir: str | Path,
    n_train_prompts: int = 150,
    n_test_prompts: int = 60,
    seed: int = 42,
    test_missing_rate: float = 0.15,
    min_words: int = 40,
    max_words: int = 250,
) -> Dict[str, Path]:
    """Write ``train.csv`` / ``test.csv`` / ``test_truth.csv`` and return paths."""
    out = ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    pool = [PROMPT_POOL[i] for i in rng.permutation(len(PROMPT_POOL))]
    cursor = {"i": 0}

    def prompt_vocab(topic: str) -> List[str]:
        start = cursor["i"]
        if start + PROMPT_WORDS_PER_PROMPT > len(pool):
            shuffled = [PROMPT_POOL[i] for i in rng.permutation(len(PROMPT_POOL))]
            pool[:] = shuffled
            start = 0
        cursor["i"] = start + PROMPT_WORDS_PER_PROMPT
        topic_words = rng.choice(
            TOPIC_VOCAB[topic], size=TOPIC_WORDS_PER_PROMPT, replace=False
        ).tolist()
        return topic_words + pool[start:start + PROMPT_WORDS_PER_PROMPT]

    rows = []
    for i in range(n_train_prompts):
        topic = TOPICS[i % len(TOPICS)]
        prompt_id = f"p{i:04d}"
        vocab = prompt_vocab(topic)
        for model in LABELS:
            rows.append(
                dict(
                    id=f"tr{i:04d}{LABELS.index(model):02d}",
                    prompt_id=prompt_id,
                    text=generate_passage(rng, model, vocab, min_words, max_words),
                    model=model,
                    topic=topic,
                )
            )
    train = pd.DataFrame(rows)

    test_rows, truth_rows = [], []
    for i in range(n_test_prompts):
        topic = TOPICS[(i + 3) % len(TOPICS)]
        prompt_id = f"q{i:04d}"
        vocab = prompt_vocab(topic)
        for model in LABELS:
            if rng.random() < test_missing_rate:
                continue
            text = generate_passage(rng, model, vocab, min_words, max_words)
            rid = f"te{i:04d}{LABELS.index(model):02d}"
            test_rows.append(dict(id=rid, text=text))
            truth_rows.append(dict(id=rid, model=model, prompt_id=prompt_id, topic=topic))
    test = pd.DataFrame(test_rows)
    truth = pd.DataFrame(truth_rows)

    paths = {
        "train": out / "train.csv",
        "test": out / "test.csv",
        "test_truth": out / "test_truth.csv",
    }
    train.drop(columns=["topic"]).to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    truth.drop(columns=["topic"]).to_csv(paths["test_truth"], index=False)
    return paths


# --------------------------------------------------------------------------- IO
def load_train(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("id", "prompt_id", "text", "model"):
        if col not in df.columns:
            raise ValueError(f"train file missing column {col!r}: {path}")
    df["text"] = df["text"].fillna("").astype(str)
    validate_labels(df["model"].unique())
    return df.reset_index(drop=True)


def load_test(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("id", "text"):
        if col not in df.columns:
            raise ValueError(f"test file missing column {col!r}: {path}")
    df["text"] = df["text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def load_test_truth(path: str | Path) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "model" in df.columns:
        validate_labels(df["model"].unique())
    return df


def add_group_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Attach a prompt-grouped fold id (never splits a prompt across folds)."""
    df = df.copy()
    n_groups = df["prompt_id"].nunique()
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    folds = np.full(len(df), -1, dtype=int)
    for k, (_, val_idx) in enumerate(gkf.split(df, groups=df["prompt_id"])):
        folds[val_idx] = k
    if (folds < 0).any():  # pragma: no cover - GroupKFold covers all rows
        folds[folds < 0] = 0
    df["fold"] = folds
    df.attrs["n_folds"] = n_splits
    return df


def ensure_dataset(cfg, force: bool = False) -> Dict[str, Path]:
    """Create the synthetic dataset if the configured files are absent."""
    data_dir = cfg.path("paths.data_dir")
    paths = {
        "train": data_dir / cfg.get("data.train_file", "train.csv"),
        "test": data_dir / cfg.get("data.test_file", "test.csv"),
        "test_truth": data_dir / cfg.get("data.test_truth_file", "test_truth.csv"),
    }
    if not force and paths["train"].exists() and paths["test"].exists():
        return paths
    synth = cfg.section("synthetic")
    print(f"[data] generating synthetic corpus -> {data_dir}")
    return make_synthetic_dataset(
        data_dir,
        n_train_prompts=int(synth.get("n_train_prompts", 150)),
        n_test_prompts=int(synth.get("n_test_prompts", 60)),
        seed=int(cfg.get("project.seed", 42)),
        test_missing_rate=float(synth.get("test_missing_rate", 0.15)),
        min_words=int(synth.get("min_words", 40)),
        max_words=int(synth.get("max_words", 250)),
    )
