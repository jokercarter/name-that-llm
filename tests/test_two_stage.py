"""Two-stage LoRA bookkeeping: triplet plan and dual-head shapes.

These run on CPU without a model: the triplet plan is pure numpy, and the head
is a plain ``nn.Module``.
"""

from __future__ import annotations

import types

import numpy as np
import torch

from ntl_pact18.config import load_config
from ntl_pact18.features.qlora_head import MeanConcatHead
from ntl_pact18.features.qwen_member import QwenMember


def _member(**overrides):
    cfg = load_config("configs/kaggle_2stage_8b.yaml", profile="off")
    member = QwenMember(types.SimpleNamespace(kind="cpu"))
    del cfg
    for key, value in overrides.items():
        setattr(member, key, value)
    return member


def _toy_groups(n_groups: int = 4, per_group: int = 18):
    groups = np.repeat(np.arange(n_groups), per_group)
    labels = np.tile(np.arange(per_group), n_groups)
    return groups, labels


def test_triplets_never_repeat_a_label():
    member = _member()
    groups, labels = _toy_groups()
    rng = np.random.default_rng(0)
    units = member._triplet_units(groups, labels, rng)
    assert len(units) == 4 * (18 // 3)
    for unit in units:
        assert len(unit) == 3
        assert len({int(labels[i]) for i in unit}) == 3
        assert len({int(groups[i]) for i in unit}) == 1


def test_triplets_skip_groups_that_are_too_small():
    member = _member()
    groups = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    labels = np.array([0, 1, 0, 1, 2, 0, 1, 2, 3])
    units = member._triplet_units(groups, labels, np.random.default_rng(0))
    # group 0 has two rows -> no triplet; group 1 has three distinct -> one
    assert len(units) == 2
    assert all(int(groups[u[0]]) != 0 for u in units)


def test_duplicate_label_in_a_group_is_collapsed():
    member = _member()
    groups = np.array([0, 0, 0, 0])
    labels = np.array([5, 5, 5, 7])   # only two distinct models
    units = member._triplet_units(groups, labels, np.random.default_rng(0))
    assert units == []


def test_batch_plan_is_deterministic_and_rank_disjoint():
    member = _member(triplet_batch=3)
    groups, labels = _toy_groups(n_groups=10)
    a = member._triplet_batches(groups, labels, 2, 7, 0)
    b = member._triplet_batches(groups, labels, 2, 7, 0)
    assert len(a) == len(b) and len(a) > 0
    for x, y in zip(a, b):
        assert [list(u) for u in x] == [list(u) for u in y]
    for batch in a:
        assert len(batch) == 6                     # 3 per rank x 2 ranks
        left = {i for u in batch[:3] for i in u}
        right = {i for u in batch[3:] for i in u}
        assert left.isdisjoint(right)


def test_mean_concat_head_shapes_and_first_token_is_ignored():
    head = MeanConcatHead(hidden_size=16, n_layers=2, n_classes=18, dropout=0.0)
    hidden = [torch.zeros(2, 5, 16), torch.zeros(2, 5, 16)]
    mask = torch.ones(2, 5, dtype=torch.long)
    plain, centred, fused = head(hidden, mask)
    assert plain.shape == (2, 18) and centred.shape == (2, 18)
    assert fused.shape == (2, 32)

    # changing only the first token must not move the pooled vector
    hidden2 = [h.clone() for h in hidden]
    hidden2[0][:, 0, :] = 5.0
    hidden2[1][:, 0, :] = -3.0
    _, _, fused2 = head(hidden2, mask)
    assert torch.allclose(fused, fused2, atol=1e-5)
