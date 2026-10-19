"""Prompt-grouped inner split used for early stopping."""

from __future__ import annotations

import numpy as np

from ntl_pact18.features.qwen_member import split_by_group


def test_split_holds_out_whole_groups():
    groups = np.repeat(np.arange(20), 5)
    tr, va = split_by_group(groups, len(groups), 0.2, 42)
    assert len(va) > 0
    assert set(groups[tr]).isdisjoint(set(groups[va]))
    assert len(tr) + len(va) == len(groups)


def test_split_with_zero_fraction_keeps_everything():
    groups = np.repeat(np.arange(20), 5)
    tr, va = split_by_group(groups, len(groups), 0.0, 42)
    assert len(tr) == len(groups)
    assert len(va) == 0


def test_split_degrades_gracefully_with_few_groups():
    groups = np.repeat(np.arange(4), 5)
    tr, va = split_by_group(groups, len(groups), 0.25, 42)
    assert len(va) == 0
    assert len(tr) == len(groups)


def test_split_without_groups():
    tr, va = split_by_group(None, 10, 0.2, 42)
    assert len(tr) == 10 and len(va) == 0
