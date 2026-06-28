"""MoDBlock (plan M10): Mixture-of-Depths per-block token routing.

Pinned properties:
  (a) capacity=1.0 reduces to the plain block -- EXACT parity (bit-identical output);
  (b) capacity=0.5 processes ~exactly half the tokens, and the SKIPPED tokens are passed
      through BIT-IDENTICALLY (pure residual identity);
  (c) the router receives finite, nonzero gradients;
  (d) training through the MoD wrapper reduces loss.
"""
import torch
import torch.nn as nn

from memory_native.mod import MoDBlock


class ResBlock(nn.Module):
    """A minimal residual block x -> x + mlp(ln(x)), the contract MoDBlock expects."""

    def __init__(self, d):
        super().__init__()
        self.n_embd = d
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.fc2(torch.nn.functional.gelu(self.fc(self.ln(x))))


def test_capacity_one_is_exact_parity():
    """capacity=1.0: MoDBlock(block)(x) is bit-identical to block(x) (router never engages)."""
    torch.manual_seed(0)
    blk = ResBlock(32)
    mod = MoDBlock(blk, capacity=1.0)
    x = torch.randn(4, 8, 32)
    plain = blk(x)
    routed = mod(x)
    assert torch.equal(plain, routed), "capacity=1.0 must be exact parity with the plain block"
    assert mod.realized_fraction() == 1.0


def test_capacity_half_processes_half_and_skips_identically():
    """capacity=0.5: exactly ceil/round half of tokens processed; skipped tokens unchanged."""
    torch.manual_seed(0)
    d = 16
    blk = ResBlock(d)
    mod = MoDBlock(blk, capacity=0.5)
    b, t = 4, 8
    n = b * t
    x = torch.randn(b, t, d)

    # Recompute the router's selection to know exactly which tokens are skipped.
    flat = x.reshape(n, d)
    scores = mod.router(flat).squeeze(-1)
    k = max(1, round(0.5 * n))
    _, top_idx = torch.topk(scores, k, sorted=False)
    sel_mask = torch.zeros(n, dtype=torch.bool)
    sel_mask[top_idx] = True

    out = mod(x)
    out_flat = out.reshape(n, d)

    # exactly k = n/2 tokens processed
    assert k == n // 2
    assert abs(mod.realized_fraction() - 0.5) < 1e-9

    # SKIPPED tokens must be bit-identical to the input (pure residual bypass)
    skipped = ~sel_mask
    assert torch.equal(out_flat[skipped], flat[skipped]), "skipped tokens were not passed through identically"

    # SELECTED tokens must have changed (the block did real work on them)
    assert not torch.equal(out_flat[sel_mask], flat[sel_mask])


def test_realized_fraction_matches_capacity():
    """The realized processed-fraction tracks capacity (no degenerate all-skip / all-keep)."""
    torch.manual_seed(1)
    blk = ResBlock(24)
    x = torch.randn(5, 10, 24)  # n=50
    for cap in (0.25, 0.5, 0.75):
        mod = MoDBlock(blk, capacity=cap)
        mod(x)
        assert abs(mod.realized_fraction() - cap) <= 0.02, (cap, mod.realized_fraction())


def test_router_receives_gradients():
    """The router's weight gets a finite, nonzero gradient (weighted-residual straight-through)."""
    torch.manual_seed(0)
    blk = ResBlock(32)
    mod = MoDBlock(blk, capacity=0.5).train()
    x = torch.randn(4, 8, 32, requires_grad=True)
    out = mod(x)
    (out ** 2).sum().backward()
    assert mod.router.weight.grad is not None
    assert torch.isfinite(mod.router.weight.grad).all()
    assert mod.router.weight.grad.abs().sum() > 0, "router got no gradient"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_training_reduces_loss():
    """Sanity: training through the MoD wrapper drives a fit-a-target loss down."""
    torch.manual_seed(0)
    d, b, t = 24, 8, 12
    blk = ResBlock(d)
    mod = MoDBlock(blk, capacity=0.5).train()
    opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=3e-3)
    x = torch.randn(b, t, d)
    y = torch.randn(b, t, d) * 0.3
    first = None
    for _ in range(300):
        out = mod(x)
        loss = (out - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.85 * first, (first, loss.item())


def test_capacity_validation():
    """capacity outside (0, 1] is rejected."""
    blk = ResBlock(8)
    for bad in (0.0, -0.5, 1.5):
        try:
            MoDBlock(blk, capacity=bad)
            assert False, f"capacity={bad} should have raised"
        except ValueError:
            pass
