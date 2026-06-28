import torch

from memory_native import ReversibleCouplingBlock, ReversibleSequential


def _ref_block_forward(block, x):
    """Plain (activation-storing) forward, for gradient comparison."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 + block.F(x2)
    y2 = x2 + block.G(y1)
    return torch.cat([y1, y2], dim=-1)


def _ref_chain_forward(blocks, x):
    """Plain stored-activation forward through a chain of blocks -- the TRUE-gradient reference
    the reversible (recompute) path must match. Storing activations means autograd here is exact."""
    h = x
    for b in blocks:
        h = _ref_block_forward(b, h)
    return h


def _mk_blocks(depth, dim=16, scale=1.0):
    """Deterministic identical blocks (seed reset each call so reference == reversible params)."""
    import torch.nn as nn
    torch.manual_seed(0)
    blocks = [ReversibleCouplingBlock(dim, F=nn.Linear(dim // 2, dim // 2, bias=False),
                                      G=nn.Linear(dim // 2, dim // 2, bias=False))
              for _ in range(depth)]
    if scale != 1.0:
        for b in blocks:
            for p in b.parameters():
                p.data.mul_(scale)
    return blocks


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


def test_inverse_path_matches_true_gradients_at_depth():
    """The DEFAULT path (anchor_every=0, float inverse) must match the true stored-activation
    gradient at depth > 1 -- not just be finite, and not just agree with another inverse impl.
    Well-conditioned weights (the regime training actually runs in); tight tolerance."""
    from memory_native import ReversibleSequence
    depth = 8
    seq = ReversibleSequence(_mk_blocks(depth))                 # O(1) inverse path
    x = torch.randn(8, 16, requires_grad=True)
    g_rev = torch.autograd.grad((seq(x) ** 2).sum(), [x] + list(seq.parameters()))

    ref_blocks = _mk_blocks(depth)                             # identical params, stored-activation
    x2 = x.detach().clone().requires_grad_(True)
    ref_leaves = [x2] + [p for b in ref_blocks for p in b.parameters()]
    g_ref = torch.autograd.grad((_ref_chain_forward(ref_blocks, x2) ** 2).sum(), ref_leaves)

    for a, b in zip(g_rev, g_ref):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (a - b).abs().max()


def test_inverse_and_anchored_exact_across_weight_scales():
    """Both the float-inverse path AND the anchored path match the true input gradient across a
    sweep of weight scales -- including extreme ones (forward magnitude ~1e9). Empirically the
    inverse does NOT accumulate gradient error here (the coupling reconstruction is exact in fp32
    for these maps); anchors are then about activation MEMORY, with no accuracy penalty either way.
    This is the depth/scale coverage the suite previously lacked."""
    from memory_native import ReversibleSequence
    depth = 16
    for scale in (0.5, 1.0, 2.0, 2.5):
        x = torch.randn(8, 16)
        ref_blocks = _mk_blocks(depth, scale=scale)
        x_ref = x.clone().requires_grad_(True)
        g_ref = torch.autograd.grad((_ref_chain_forward(ref_blocks, x_ref) ** 2).sum(), [x_ref])[0]
        for ae in (0, 4):                                      # inverse (0) and anchored (4)
            seq = ReversibleSequence(_mk_blocks(depth, scale=scale), anchor_every=ae)
            xi = x.clone().requires_grad_(True)
            g = torch.autograd.grad((seq(xi) ** 2).sum(), [xi])[0]
            denom = g_ref.abs().max().clamp_min(1e-12)
            rel = (g - g_ref).abs().max() / denom
            assert rel < 1e-4, f"scale={scale} anchor_every={ae}: rel grad err {rel.item():.2e}"


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


def test_o1_reversible_sequence_matches_per_block():
    """ReversibleSequence (O(1) whole-chain) must give identical grads to the per-block version."""
    import torch.nn as nn
    from memory_native import ReversibleSequence

    def mk():
        torch.manual_seed(0)
        return [ReversibleCouplingBlock(16, F=nn.Linear(8, 8, bias=False),
                                        G=nn.Linear(8, 8, bias=False)) for _ in range(5)]
    o1 = ReversibleSequence(mk())
    pb = ReversibleSequential(mk())
    x = torch.randn(8, 16, requires_grad=True)
    g1 = torch.autograd.grad((o1(x) ** 2).sum(), [x] + list(o1.parameters()))
    x2 = x.detach().clone().requires_grad_(True)
    g2 = torch.autograd.grad((pb(x2) ** 2).sum(), [x2] + list(pb.parameters()))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g1, g2))
    # ...and BOTH must match the true stored-activation gradient (not just agree with each other).
    ref_blocks = mk()
    x3 = x.detach().clone().requires_grad_(True)
    ref_leaves = [x3] + [p for b in ref_blocks for p in b.parameters()]
    g_ref = torch.autograd.grad((_ref_chain_forward(ref_blocks, x3) ** 2).sum(), ref_leaves)
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(g1, g_ref))


def test_o1_reversible_stores_one_output():
    """The whole-chain Function must save exactly one activation tensor (the final output) +
    params -- the O(1)-in-depth guarantee."""
    import torch.nn as nn
    from memory_native import ReversibleSequence

    blocks = [ReversibleCouplingBlock(16, F=nn.Linear(8, 8, bias=False),
                                      G=nn.Linear(8, 8, bias=False)) for _ in range(7)]
    seq = ReversibleSequence(blocks)
    x = torch.randn(4, 16, requires_grad=True)
    y = seq(x)
    n_params = sum(1 for _ in seq.parameters())
    # saved_tensors = final output (1) + all params; independent of depth beyond the params
    assert len(y.grad_fn.saved_tensors) == 1 + n_params
    (y ** 2).sum().backward()
