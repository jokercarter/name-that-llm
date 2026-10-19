"""Parallel primitives: sharding, batch plans, and the single-GPU no-ops."""

from __future__ import annotations

import numpy as np

from ntl_pact18 import dist_utils
from ntl_pact18.features.qwen_member import QwenMember


def test_shard_indices_partition_the_range():
    n, world = 103, 4
    seen = []
    for rank in range(world):
        idx = dist_utils.shard_indices(n, rank, world)
        seen.append(idx)
    joined = np.concatenate(seen)
    assert len(joined) == n
    assert set(joined.tolist()) == set(range(n))
    # order preserving, so the gather can simply concatenate
    for idx in seen:
        assert np.all(np.diff(idx) == 1)


def test_shard_indices_single_process_is_identity():
    idx = dist_utils.shard_indices(7, 0, 1)
    assert idx.tolist() == list(range(7))


def test_gather_padded_is_identity_without_distributed():
    local = np.arange(12, dtype=np.float32).reshape(6, 2)
    out = dist_utils.gather_padded(local, 1, None)
    assert np.array_equal(out, local)


def test_all_reduce_grads_noop_at_world_one():
    called = dist_utils.all_reduce_grads([], 1)
    assert called == 0


def _batches(lengths, global_batch, seed=42, bucket=True, pool=4):
    class Dummy:
        length_bucket = bucket
        bucket_pool = pool

    return QwenMember._make_batches(
        Dummy(), np.arange(len(lengths)), np.asarray(lengths), global_batch, seed
    )


def test_batches_cover_every_row_when_bucketing():
    lengths = np.random.default_rng(0).integers(40, 400, size=997)
    plan = _batches(lengths, 16)
    flat = np.concatenate(plan)
    # only the final carry (< one batch) may be dropped
    assert len(flat) >= 997 - 15
    assert len(set(flat.tolist())) == len(flat)
    assert all(len(b) == 16 for b in plan)


def test_batches_are_deterministic_across_ranks():
    lengths = np.random.default_rng(1).integers(40, 400, size=500)
    a = _batches(lengths, 8, seed=7)
    b = _batches(lengths, 8, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_length_bucketing_reduces_padding():
    lengths = np.random.default_rng(2).integers(40, 400, size=2000)
    random_plan = _batches(lengths, 16, bucket=False)
    bucketed = _batches(lengths, 16, bucket=True)

    def padded_tokens(plan):
        return sum(int(lengths[b].max()) * len(b) for b in plan)

    plain = padded_tokens(random_plan)
    bucket = padded_tokens(bucketed)
    assert bucket < 0.9 * plain, (bucket, plain)


def test_batches_returns_empty_for_tiny_input():
    assert _batches(np.array([100, 120]), 16) == []


def test_rank_slices_partition_every_global_batch():
    """Each rank trains on a disjoint half of the same global batch."""
    lengths = np.random.default_rng(3).integers(40, 400, size=1500)
    per_rank, world = 8, 2
    plan = _batches(lengths, per_rank * world)
    assert plan, "expected at least one global batch"
    for batch in plan:
        slices = [batch[r * per_rank:(r + 1) * per_rank] for r in range(world)]
        assert all(len(s) == per_rank for s in slices)
        assert sorted(np.concatenate(slices).tolist()) == sorted(batch.tolist())
        # no overlap between the two cards
        assert set(slices[0].tolist()).isdisjoint(slices[1].tolist())


def test_global_batches_do_not_overlap():
    lengths = np.random.default_rng(4).integers(40, 400, size=1200)
    plan = _batches(lengths, 16)
    flat = np.concatenate(plan)
    assert len(set(flat.tolist())) == len(flat)
