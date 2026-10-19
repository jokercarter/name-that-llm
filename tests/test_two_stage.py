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


# ------------------------------------------------------- stage-1 pool capping
def test_stage1_cap_is_stratified_and_deterministic():
    member = _member(stage1_max_rows=300, seed=7)
    labels = np.array([0] * 600 + [1] * 300 + [2] * 100)
    texts = [f"t{i}" for i in range(len(labels))]
    groups = np.arange(len(labels))

    t1, y1, g1 = member._cap_stage1_rows(texts, labels, groups)
    t2, y2, g2 = member._cap_stage1_rows(texts, labels, groups)
    assert len(t1) == len(y1) == len(g1) == 300
    assert t1 == t2 and y1 == y2 and g1 == g2          # same seed -> same rows
    counts = {int(v): int((np.asarray(y1) == v).sum()) for v in (0, 1, 2)}
    # 300/1000 of each class, rounded: 180 / 90 / 30
    assert counts == {0: 180, 1: 90, 2: 30}
    # every kept row stays aligned across the three lists
    for i, txt in enumerate(t1):
        j = texts.index(txt)
        assert y1[i] == labels[j] and g1[i] == groups[j]


def test_stage1_cap_disabled_or_above_pool_is_a_noop():
    member = _member(stage1_max_rows=0, seed=7)
    labels = np.array([0, 1, 2])
    texts = ["a", "b", "c"]
    assert member._cap_stage1_rows(texts, labels, None) == (texts, labels, None)
    member.stage1_max_rows = 10
    assert member._cap_stage1_rows(texts, labels, None) == (texts, labels, None)


def test_stage1_class_weights_are_capped_and_normalised():
    from ntl_pact18.labels import LABELS
    member = _member(stage1_class_weighted=True)
    y = [LABELS[0]] * 100 + [LABELS[1]] * 50 + [LABELS[2]] * 10
    w = member._stage1_class_weights(y)
    assert w is not None and len(w) == 18
    # inverse frequency 160/(3*n): 0.533 / 1.067 / 5.333 -> cap 3.0 ->
    # mean-1 over the three present classes
    expect = np.minimum(np.array([0.5333, 1.0667, 5.3333]), 3.0)
    expect = expect / expect.mean()
    assert np.allclose(w[:3], expect, atol=1e-4)
    assert np.all(w[3:] == 0.0)                     # absent classes get 0
    assert abs(np.mean(w[:3]) - 1.0) < 1e-5
    assert _member(stage1_class_weighted=False)._stage1_class_weights(y) is None


def test_mode_two_stage_implies_the_flag_everywhere():
    """Regression: the trainer and the member must derive the flag alike.

    The shipped bug: the config set only ``mode: two_stage`` (no explicit
    ``two_stage`` key), the member implied it from the mode, the parallel
    trainer read the missing key as ``False`` -> stage 1 never ran and every
    fold raised ``must run fit_stage1() on the pool first``.
    """
    from ntl_pact18.members import build_qwen_member, two_stage_from_config

    cfg = load_config("configs/kaggle_2stage_8b.yaml", profile="off")
    q = cfg.section("features.qwen")
    assert "two_stage" not in q                 # the exact trap that shipped
    assert str(q.get("mode")) == "two_stage"

    # explicit override still wins
    assert two_stage_from_config({"mode": "two_stage", "two_stage": False}) is False
    assert two_stage_from_config({"mode": "probe", "two_stage": True}) is True
    assert two_stage_from_config({"mode": "probe"}) is False
    assert two_stage_from_config(None) is False

    # what the trainer computes ...
    trainer_flag = two_stage_from_config(q)
    # ... must equal what the member constructor receives
    member = build_qwen_member(cfg, types.SimpleNamespace(kind="cuda"))
    assert member is not None
    assert member.two_stage == trainer_flag is True
