"""Head and auxiliary loss for the scaled QLoRA member.

The backbone stays frozen in 4-bit; everything that learns lives here:

* :class:`LayerFusionHead` - one attention-pooled summary per selected decoder
  layer, concatenated and projected to 18 logits. Mean pooling over the last
  layer throws away exactly the signal this task needs (early layers encode
  *how* a passage is written, late layers encode *what* it is about), so the
  head reads several depths and learns how much to trust each.
* :func:`supcon_loss` - supervised contrastive term on the fused embedding.
  The hard errors are sibling models (terra vs sol, 3.7 vs 3.8 flash); a
  cross-entropy term only pushes each row towards its own class, while SupCon
  pulls same-class rows together and pushes different-class rows apart, which
  is what separates near-duplicate styles.

Both are plain ``torch.nn`` code with no transformers dependency, so they can
be unit-tested on CPU without loading an 8B model.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LayerFusionHead(nn.Module):
    """Attention-pool each selected layer, concat, MLP to ``n_classes``."""

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        n_classes: int = 18,
        proj: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.proj = nn.ModuleList(
            [nn.Linear(hidden_size, proj, bias=False) for _ in range(n_layers)]
        )
        self.score = nn.ModuleList([nn.Linear(proj, 1) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(proj * n_layers)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(proj * n_layers, proj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj, n_classes),
        )

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(hidden_states) != self.n_layers:
            raise ValueError(
                f"head expects {self.n_layers} layers, got {len(hidden_states)}"
            )
        mask = attention_mask.to(torch.bool)
        pooled: List[torch.Tensor] = []
        for hidden, proj, score in zip(hidden_states, self.proj, self.score):
            z = torch.tanh(proj(hidden.float()))
            s = score(z).squeeze(-1)
            s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
            w = torch.softmax(s, dim=1)
            pooled.append((z * w.unsqueeze(-1)).sum(dim=1))
        fused = self.norm(torch.cat(pooled, dim=-1))
        return self.out(self.drop(fused)), fused


class MeanConcatHead(nn.Module):
    """Per-layer LayerNorm + masked mean pool, concatenated, two linear heads.

    This is the head used by the current public two-stage LoRA recipe. It is
    cheaper than :class:`LayerFusionHead` (no per-layer projection or attention
    scorer) and, more importantly, it exposes **two** heads over the same pooled
    features: ``plain`` and ``centred``. The centred head is trained on
    triplet-centred features, i.e. on the residual after subtracting the mean of
    three passages that answer the same prompt, which is what teaches the
    representation to ignore the topic.

    The first token is dropped from the mean: in decoder-only models it acts as
    an attention sink and its state is dominated by position, not style.
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        n_classes: int = 18,
        dropout: float = 0.1,
        drop_first_token: bool = True,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.drop_first_token = bool(drop_first_token)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_layers)]
        )
        self.drop = nn.Dropout(dropout)
        width = hidden_size * n_layers
        self.plain = nn.Linear(width, n_classes)
        self.centered = nn.Linear(width, n_classes)

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(hidden_states) != self.n_layers:
            raise ValueError(
                f"head expects {self.n_layers} layers, got {len(hidden_states)}"
            )
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        if self.drop_first_token and mask.size(1) > 1:
            mask = mask.clone()
            mask[:, 0, :] = 0.0
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = [
            norm((hidden.float() * mask).sum(dim=1) / denom)
            for hidden, norm in zip(hidden_states, self.norms)
        ]
        fused = self.drop(torch.cat(pooled, dim=-1))
        return self.plain(fused), self.centered(fused), fused


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020), one view per row."""
    z = F.normalize(features, dim=1)
    sim = z @ z.t() / max(temperature, 1e-6)
    n = z.size(0)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    logits = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.sum() * 0.0
    per_row = -(log_prob * pos).sum(dim=1)[valid] / pos_count[valid]
    return per_row.mean()


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
