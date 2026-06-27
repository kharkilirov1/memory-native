"""Reversible anchors (acceleration memo M7).

Anchored mode recomputes each chunk forward from a stored anchor instead of inverting, so unlike
the O(1) inverse path it reproduces the TRUE gradient exactly (no float-inverse error) for any
anchor spacing. These tests pin that: anchored grads == plain-autograd grads, and the full model
still trains (inner counters fire once per block).
"""
import pytest
import torch

from memory_native import GPTConfig, ReversibleGPT
from memory_native.reversible import ReversibleCouplingBlock, ReversibleSequence, _couple_fwd


@pytest.mark.parametrize("anchor_every", [1, 2, 3, 5, 8])
def test_anchored_matches_true_gradients(anchor_every):
    torch.manual_seed(0)
    dim, L, N = 8, 5, 4
    blocks = [ReversibleCouplingBlock(dim) for _ in range(L)]
    x = torch.randn(N, dim)

    # reference: plain sequential coupling with grad -> the true gradient
    xr = x.clone().requires_grad_(True)
    h = xr
    for b in blocks:
        h = _couple_fwd(b, h)
    h.pow(2).sum().backward()
    ref_pg = [p.grad.clone() for b in blocks for p in b.parameters()]
    ref_xg = xr.grad.clone()
    for b in blocks:
        for p in b.parameters():
            p.grad = None

    # anchored reversible over the SAME blocks
    xa = x.clone().requires_grad_(True)
    seq = ReversibleSequence(blocks, anchor_every=anchor_every)
    seq(xa).pow(2).sum().backward()
    got_pg = [p.grad for b in blocks for p in b.parameters()]

    assert torch.allclose(xa.grad, ref_xg, atol=1e-5)
    for g, r in zip(got_pg, ref_pg):
        assert torch.allclose(g, r, atol=1e-5)


def test_anchored_model_trains_and_fires_counters():
    torch.manual_seed(0)
    cfg = GPTConfig(48, 16, 6, 2, 32)   # 6 layers so anchors actually chunk
    m = ReversibleGPT(cfg, "counter_packed", anchor_every=2, C=11, act_save_bits=4).train()
    before = [c.state.clone() for c in m.counter_layers()]
    idx = torch.randint(0, 48, (3, 16))
    tgt = torch.randint(0, 48, (3, 16))
    _, loss = m(idx, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    after = [c.state for c in m.counter_layers()]
    assert any(not torch.equal(a, b) for a, b in zip(after, before)), "counters did not update"
