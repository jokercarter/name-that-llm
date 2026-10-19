"""Shape, masking and loss sanity for the scaled QLoRA head (CPU only)."""

from __future__ import annotations

import torch

from ntl_pact18.features.qlora_head import (
    LayerFusionHead,
    supcon_loss,
    trainable_parameter_count,
)


def _batch(n_layers=3, hidden=32, batch=2, tokens=7):
    hs = [torch.randn(batch, tokens, hidden) for _ in range(n_layers)]
    mask = torch.ones(batch, tokens, dtype=torch.long)
    mask[1, 4:] = 0
    return hs, mask


def test_head_shapes():
    head = LayerFusionHead(hidden_size=32, n_layers=3, n_classes=18, proj=16, dropout=0.0)
    logits, fused = head(*_batch())
    assert logits.shape == (2, 18)
    assert fused.shape == (2, 3 * 16)


def test_head_ignores_padding():
    torch.manual_seed(0)
    head = LayerFusionHead(32, 3, 18, 16, dropout=0.0).eval()
    hs, mask = _batch()
    with torch.no_grad():
        first = head(hs, mask)[0]
        perturbed = [h.clone() for h in hs]
        perturbed[0][1, 4:] = 1e3
        second = head(perturbed, mask)[0]
    assert torch.allclose(first, second, atol=1e-5)


def test_head_rejects_wrong_layer_count():
    head = LayerFusionHead(32, 3, 18, 16)
    hs, mask = _batch(n_layers=2)
    try:
        head(hs, mask)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a layer-count mismatch")


def test_supcon_is_finite_and_differentiable():
    torch.manual_seed(0)
    z = torch.randn(6, 8, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supcon_loss(z, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_supcon_handles_single_class_and_batch_of_one():
    z = torch.randn(3, 8)
    y = torch.zeros(3, dtype=torch.long)
    assert torch.isfinite(supcon_loss(z, y))
    single = torch.randn(1, 8)
    assert torch.isfinite(supcon_loss(single, torch.tensor([0])))


def test_head_has_trainable_parameters():
    head = LayerFusionHead(32, 3, 18, 16, 0.1)
    assert trainable_parameter_count(head) > 0
