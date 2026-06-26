import torch

from memory_native import ReversibleCouplingBlock, ReversibleSequential


def _ref_block_forward(block, x):
    """Plain (activation-storing) forward, for gradient comparison."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 + block.F(x2)
    y2 = x2 + block.G(y1)
    return torch.cat([y1, y2], dim=-1)


def test_reversible_matches_plain_autograd():
    """Recompute-backward gradients (input + params) must match plain stored autograd."""
    torch.manual_seed(0)
    dim = 16
    block = ReversibleCouplingBlock(dim)
    x = torch.randn(8, dim, requires_grad=True)

    # reversible path
    y = block(x)
    loss = (y ** 2).sum()
    loss.backward()
    g_rev_x = x.grad.clone()
    g_rev_params = [p.grad.clone() for p in block.parameters()]

    # reference path
    x.grad = None
    for p in block.parameters():
        p.grad = None
    y_ref = _ref_block_forward(block, x)
    (y_ref ** 2).sum().backward()
    g_ref_x = x.grad.clone()
    g_ref_params = [p.grad.clone() for p in block.parameters()]

    assert torch.allclose(g_rev_x, g_ref_x, atol=1e-5), (g_rev_x - g_ref_x).abs().max()
    for gr, gf in zip(g_rev_params, g_ref_params):
        assert torch.allclose(gr, gf, atol=1e-5), (gr - gf).abs().max()


def test_reversible_reconstruction_accurate_with_depth():
    torch.manual_seed(0)
    dim, depth = 32, 12
    blocks = [ReversibleCouplingBlock(dim) for _ in range(depth)]
    # small weights so the residual stream stays stable
    for b in blocks:
        for p in b.parameters():
            p.data.mul_(0.1)
    stack = ReversibleSequential(blocks)
    x = torch.randn(4, dim, requires_grad=True)
    y = stack(x)
    # a backward must run end-to-end through the reconstructing chain without error
    (y ** 2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_counters_learn_inside_reversible_no_grad_input():
    """The full method: counter layers inside a reversible stack must self-update through the
    recompute, even when the input carries no gradient (the tap forces backward to run)."""
    import torch.nn as nn
    from memory_native import ReversibleSequential, RMSCounterLinear

    torch.manual_seed(0)
    half, depth = 32, 4
    def cmlp():
        return nn.Sequential(RMSCounterLinear(half, half, C=11, lr=4e-3, lr_scale=2e-4),
                             nn.Tanh(),
                             RMSCounterLinear(half, half, C=11, lr=4e-3, lr_scale=2e-4))
    stack = ReversibleSequential(
        [ReversibleCouplingBlock(2 * half, F=cmlp(), G=cmlp()) for _ in range(depth)]).train()
    counters = [m for m in stack.modules() if isinstance(m, RMSCounterLinear)]
    x = torch.randn(64, 2 * half)        # NOTE: no requires_grad
    target = torch.randn(64, 2 * half)
    first = None
    for _ in range(200):
        y = stack(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()                  # counters self-update via the recompute
        if first is None:
            first = loss.item()
    last = loss.item()
    flips = sum(int(c.weight_flips) for c in counters)
    assert flips > 0, "counters never updated inside reversible (tap missing?)"
    assert last < first, (first, last)
