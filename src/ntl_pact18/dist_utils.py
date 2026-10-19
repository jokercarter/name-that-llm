"""Two-GPU data-parallel primitives for the QLoRA member.

Why manual gradient all-reduce instead of ``DistributedDataParallel``
--------------------------------------------------------------------
The trainable tensors here are LoRA adapters bolted onto bitsandbytes 4-bit
modules (``Params4bit``), inside a model whose embeddings are tied and whose
decoder layers run under gradient checkpointing. DDP's reducer assumes a stable
bucket layout over plain dense parameters, and that combination is exactly the
one that breaks in practice (unused-parameter errors, reducer rebuilds after
checkpointing, buckets over frozen quantised tensors). A manual all-reduce over
the ~50 trainable tensors is a dozen lines, works with any module mix, and is a
strict no-op at ``world_size == 1`` -- so the laptop path stays byte-identical
to the single-GPU one.

Everything else the two ranks need is here too: deterministic sharding, a
padded gather for predictions (so the OOF pass is not a single-GPU bottleneck)
and a broadcast for the early-stopping decision (so both ranks stop on the same
step; otherwise one rank would block forever in the next collective).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DistState:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    backend: str = ""

    @property
    def enabled(self) -> bool:
        return self.world > 1


_STATE = DistState()


def current() -> DistState:
    """The process-group state, or ``world == 1`` when running single-process."""
    return _STATE


def init_distributed(backend: Optional[str] = None) -> DistState:
    """Join the process group described by ``RANK`` / ``WORLD_SIZE`` env vars.

    Both ``torchrun`` and the spawn launcher in ``scripts/train_parallel.py``
    set those variables, so the same code works under either.
    """
    global _STATE
    try:
        import torch
        import torch.distributed as dist
    except Exception:  # pragma: no cover - torch missing means no distributed
        return _STATE

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1 or not dist.is_available():
        return _STATE

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if torch.cuda.is_available():
        # bind the process to its card *before* the first collective, so NCCL
        # allocates the communication buffers on the right device
        torch.cuda.set_device(local_rank)

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world)

    _STATE = DistState(rank=rank, world=world, local_rank=local_rank, backend=backend)
    return _STATE


def shutdown() -> None:
    global _STATE
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:  # pragma: no cover
        pass
    _STATE = DistState()


def barrier() -> None:
    if _STATE.world <= 1:
        return
    import torch.distributed as dist

    dist.barrier()


# --------------------------------------------------------------------- grads
def all_reduce_grads(params: Sequence["object"], world: int) -> int:
    """Average gradients across ranks, in place. No-op for ``world == 1``.

    Parameters that received no gradient on this rank are reduced as zeros and
    then reset to ``None``: leaving a zero gradient in place would make AdamW
    apply weight decay to a parameter that the forward pass never touched.
    """
    if world <= 1:
        return 0
    import torch
    import torch.distributed as dist

    missing: List["object"] = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
            missing.append(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)
    for p in missing:
        p.grad = None
    return len(params)


def broadcast_tensors(tensors: Iterable["object"]) -> None:
    """Copy rank 0's tensors to every rank (keeps replicas bit-identical)."""
    if _STATE.world <= 1:
        return
    import torch.distributed as dist

    with _guard_no_grad():
        for t in tensors:
            dist.broadcast(t, src=0)


class _guard_no_grad:
    def __enter__(self):
        import torch

        self._ctx = torch.no_grad()
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def broadcast_flag(flag: bool, device) -> bool:
    """Share rank 0's boolean decision with every rank."""
    if _STATE.world <= 1:
        return bool(flag)
    import torch
    import torch.distributed as dist

    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.broadcast(t, src=0)
    return bool(int(t.item()))


# ---------------------------------------------------------------- sharding
def shard_indices(n: int, rank: int, world: int) -> np.ndarray:
    """Contiguous, order-preserving slice of ``range(n)`` for this rank."""
    if world <= 1:
        return np.arange(n)
    bounds = np.linspace(0, n, world + 1).astype(int)
    return np.arange(bounds[rank], bounds[rank + 1])


def gather_padded(local: np.ndarray, world: int, device) -> np.ndarray:
    """All-gather variable-length rows and reassemble them in rank order.

    Used by ``QwenMember.predict_proba``: with two cards a fold's validation
    rows are split in half, and the gather costs ~1.4 MB per fold instead of
    doubling the inference wall-clock.
    """
    if world <= 1:
        return local
    import torch
    import torch.distributed as dist

    local = np.ascontiguousarray(local, dtype=np.float32)
    n_local, dim = local.shape

    width = torch.tensor([n_local], device=device, dtype=torch.long)
    dist.all_reduce(width, op=dist.ReduceOp.MAX)
    max_n = int(width.item())

    me = torch.from_numpy(local).to(device)
    if max_n > n_local:
        pad = torch.zeros((max_n - n_local, dim), device=device, dtype=torch.float32)
        me = torch.cat([me, pad], dim=0)

    chunks = [torch.zeros((max_n, dim), device=device, dtype=torch.float32)
              for _ in range(world)]
    dist.all_gather(chunks, me)

    counts = torch.zeros(world, device=device, dtype=torch.long)
    counts[dist.get_rank()] = n_local
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    total = int(counts.sum().item())
    stacked = torch.cat([c[: int(counts[i])] for i, c in enumerate(chunks)], dim=0)
    return stacked[:total].cpu().numpy()
