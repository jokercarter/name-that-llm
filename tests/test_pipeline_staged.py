"""Staged fitting: injected members/OOF and the competition-only eval mask.

The two-GPU trainer computes the expensive member pass in separate processes and
then hands the result to ``PACT18Pipeline.fit``. These tests pin that contract
down without touching a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ntl_pact18.config import load_config
from ntl_pact18.data import add_group_folds, make_synthetic_dataset
from ntl_pact18.labels import LABEL_TO_INDEX, LABELS
from ntl_pact18.pipeline import PACT18Pipeline


def _cfg():
    return load_config(
        overrides=[
            "project.seed=3",
            "data.n_folds=2",
            "clustering.threshold=0.9",
            "evaluation.bootstrap=30",
        ]
    )


class FakeMember:
    """Deterministic stand-in: a fixed probability row per text."""

    def __init__(self, name: str, table: dict[str, np.ndarray]):
        self.name = name
        self.table = table
        self.fit_calls = 0

    def fit(self, texts, y, groups=None):
        self.fit_calls += 1
        return self

    def predict_proba(self, texts, groups=None):
        return np.vstack([self.table[t] for t in texts])


def _dataset(tmp_path, n_prompts=20):
    paths = make_synthetic_dataset(
        tmp_path / "data", n_train_prompts=n_prompts, n_test_prompts=2, seed=5,
        test_missing_rate=0.1, min_words=40, max_words=90,
    )
    df = pd.read_csv(paths["train"])
    return df


def _fake_members(df: pd.DataFrame, seed: int = 0):
    rng = np.random.default_rng(seed)
    table_a, table_b = {}, {}
    for text, label in zip(df["text"], df["model"]):
        a = rng.dirichlet(np.ones(len(LABELS)) * 0.4)
        b = rng.dirichlet(np.ones(len(LABELS)) * 0.4)
        hit = LABEL_TO_INDEX[label]
        a[hit] += 2.0
        b[hit] += 1.0
        table_a[text] = a / a.sum()
        table_b[text] = b / b.sum()
    members = [FakeMember("fake_a", table_a), FakeMember("fake_b", table_b)]
    oof = {
        "fake_a": np.vstack([table_a[t] for t in df["text"]]),
        "fake_b": np.vstack([table_b[t] for t in df["text"]]),
    }
    return members, oof


def test_staged_fit_uses_injected_oof_without_refit(tmp_path):
    df = _dataset(tmp_path)
    members, oof = _fake_members(df)

    pipe = PACT18Pipeline(_cfg()).fit(
        df, verbose=False, members=members, oof=oof, refit=False
    )

    assert [m.name for m in pipe.members] == ["fake_a", "fake_b"]
    assert all(m.fit_calls == 0 for m in members)      # refit skipped
    assert set(pipe.oof_probs_) == {"fake_a", "fake_b"}
    np.testing.assert_allclose(pipe.oof_probs_["fake_a"], oof["fake_a"])
    assert pipe.train_summary_["n_eval_rows"] == len(df)
    assert 0.0 <= pipe.train_summary_["oof_blend_balanced_accuracy"] <= 1.0


def test_eval_mask_restricts_the_reported_metric(tmp_path):
    df = _dataset(tmp_path)
    members, oof = _fake_members(df)
    df = add_group_folds(df, 2, 3)
    df["origin"] = np.where(np.arange(len(df)) % 3 == 0, "competition", "external")
    mask = (df["origin"] == "competition").to_numpy()

    pipe = PACT18Pipeline(_cfg()).fit(
        df, verbose=False, members=members, oof=oof, refit=False, eval_mask=mask
    )
    assert pipe.train_summary_["n_eval_rows"] == int(mask.sum())
    assert len(pipe.oof_probs_["fake_a"]) == len(df)   # OOF itself stays full


def test_missing_oof_member_is_an_error(tmp_path):
    df = _dataset(tmp_path)
    members, oof = _fake_members(df)
    with pytest.raises(ValueError, match="missing members"):
        PACT18Pipeline(_cfg()).fit(
            df, verbose=False, members=members, oof={"fake_a": oof["fake_a"]},
            refit=False,
        )


def test_wrong_oof_shape_is_an_error(tmp_path):
    df = _dataset(tmp_path)
    members, oof = _fake_members(df)
    bad = {"fake_a": oof["fake_a"][:-5], "fake_b": oof["fake_b"]}
    with pytest.raises(ValueError, match="shape"):
        PACT18Pipeline(_cfg()).fit(
            df, verbose=False, members=members, oof=bad, refit=False
        )
