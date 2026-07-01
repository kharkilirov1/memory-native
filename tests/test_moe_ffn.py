"""CounterMoEFFN (plan M4): sparse Mixture-of-counter-Experts FFN.

Defining properties pinned here:
  (a) the router gets grads (fp Parameter, AdamW) AND the experts self-update (counter state moves);
  (b) top_k=1 routes each token to exactly one expert;
  (c) the load-balance aux loss reduces routing imbalance vs no aux over a few steps;
  (d) the MoE can fit a small target (loss drops a lot).
"""
import torch

from memory_native.moe_ffn import CounterMoEFFN


def test_router_grads_and_experts_self_update():
    """Router weight gets a finite grad; at least one expert's counter state moves after backward."""
    torch.manual_seed(0)
    ffn = CounterMoEFFN(32, n_experts=4, top_k=2, C=11, lr=0.06).train()
    # router is the ONLY fp Parameter
    fp_params = [n for n, p in ffn.named_parameters() if p.requires_grad]
    assert fp_params == ["router.weight"], fp_params

    states0 = [(e.fc1.state.clone(), e.fc2.state.clone()) for e in ffn.experts]
    x = torch.randn(4, 8, 32, requires_grad=True)
    out = ffn(x)
    (out ** 2).sum().backward()

    assert ffn.router.weight.grad is not None
    assert torch.isfinite(ffn.router.weight.grad).all()
    assert ffn.router.weight.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()

    moved = any(
        (not torch.equal(e.fc1.state, s0[0])) or (not torch.equal(e.fc2.state, s0[1]))
        for e, s0 in zip(ffn.experts, states0)
    )
    assert moved, "no expert counter state changed -- experts did not self-update"


def test_topk1_routes_each_token_to_one_expert():
    """top_k=1: every token contributes to exactly one (token,slot) assignment, one expert."""
    torch.manual_seed(1)
    ffn = CounterMoEFFN(16, n_experts=8, top_k=1).train()
    N = 5 * 7
    x = torch.randn(5, 7, 16)
    h = x.reshape(-1, 16)
    logits = ffn.router(h)
    probs = torch.softmax(logits, dim=-1)
    top_w, top_idx = probs.topk(1, dim=-1)
    assert top_idx.shape == (N, 1)
    # each token picks exactly one expert; assignments sum to N
    one_hot = torch.nn.functional.one_hot(top_idx.reshape(-1), num_classes=8)
    assert int(one_hot.sum()) == N
    # forward still produces a finite output of the right shape
    out = ffn(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def _imbalance(ffn) -> float:
    """Coefficient of variation of cumulative per-expert token fractions (0 == perfectly balanced)."""
    f = ffn.token_count / ffn.token_count.sum().clamp_min(1)
    return float(f.std() / f.mean().clamp_min(1e-9))


def _train_imbalance(aux_weight: float, steps: int = 60) -> float:
    torch.manual_seed(0)
    d = 24
    ffn = CounterMoEFFN(d, n_experts=8, top_k=2, C=11, lr=0.04,
                        aux_loss_weight=aux_weight).train()
    opt = torch.optim.AdamW([p for p in ffn.parameters() if p.requires_grad], lr=3e-3)
    # a fixed batch with structure, so the router has a real reason to specialize (and collapse)
    x = torch.randn(64, d)
    y = torch.randn(64, d) * 0.3
    ffn.token_count.zero_()
    for _ in range(steps):
        out = ffn(x)
        loss = (out - y).pow(2).mean() + aux_weight * ffn.last_aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return _imbalance(ffn)


def test_aux_loss_reduces_routing_imbalance():
    """With the load-balance aux loss ON, cumulative routing is more balanced than with it OFF."""
    imb_off = _train_imbalance(aux_weight=0.0)
    imb_on = _train_imbalance(aux_weight=0.1)
    assert imb_on < imb_off, (imb_on, imb_off)


def test_moe_can_fit_a_target():
    """Sanity: the counter experts + router actually learn (loss drops a lot)."""
    torch.manual_seed(0)
    d, N = 24, 96
    ffn = CounterMoEFFN(d, n_experts=4, top_k=2, C=11, lr=0.06, lr_scale=3e-4).train()
    opt = torch.optim.AdamW([p for p in ffn.parameters() if p.requires_grad], lr=3e-3)
    x = torch.randn(N, d)
    y = torch.randn(N, d) * 0.3
    first = None
    for _ in range(300):
        out = ffn(x)
        loss = (out - y).pow(2).mean() + 1e-2 * ffn.last_aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.7 * first, (first, loss.item())


def test_active_macs_equal_compute_sizing():
    """Default expert_hidden = 4d/top_k makes top_k experts ~ the dense 2*d*4d active MACs."""
    d = 128
    dense = 2 * d * 4 * d
    for k in (1, 2):
        ffn = CounterMoEFFN(d, n_experts=8, top_k=k)
        expert_macs = ffn.k * (2 * d * ffn.h)
        # within 10% of the dense active MACs (router term is tiny on top)
        assert abs(expert_macs - dense) <= 0.10 * dense, (k, expert_macs, dense)


def test_capacity_grows_without_active_compute():
    """More experts (E up) raises persistent bytes but NOT the per-token expert compute."""
    d = 64
    small = CounterMoEFFN(d, n_experts=4, top_k=2)
    big = CounterMoEFFN(d, n_experts=16, top_k=2)
    assert big.persistent_bytes() > 3 * small.persistent_bytes()
    # the only active-MACs growth is the tiny router term d*E (4x), not the expert term.
    small_expert = small.k * (2 * d * small.h)
    big_expert = big.k * (2 * d * big.h)
    assert small_expert == big_expert


def test_packed_experts_match_unpacked_on_cpu():
    """Prong A: packed experts (fused-update path on CUDA) keep IDENTICAL dynamics on CPU -- packed
    only changes storage + which update kernel fires, not the math. Same seed -> bit-exact forward
    and bit-exact loss after a training step. (Guards 'no CPU regression' from the default switch.)"""
    import torch.nn.functional as F
    d = 32
    def build(packed):
        torch.manual_seed(0)
        return CounterMoEFFN(d, n_experts=4, top_k=2, C=11, packed_experts=packed).train()
    a, b = build(False), build(True)
    torch.manual_seed(1); x = torch.randn(8, 5, d)
    ya = a(x.clone()); yb = b(x.clone())
    assert torch.equal(ya, yb)                       # identical forward
    # the counter update uses torch.rand stochastic rounding -> reseed before each backward so both
    # arms draw the SAME SR stream (otherwise a's draws advance the RNG before b's update).
    torch.manual_seed(2); ya.pow(2).mean().backward()
    torch.manual_seed(2); yb.pow(2).mean().backward()
    # after one self-update step the experts' visible weights must still match bit-for-bit
    for ea, eb in zip(a.experts, b.experts):
        from memory_native.counter import decode_state
        from memory_native.packed import unpack_codes
        ta, _ = decode_state(ea.fc1.state, 11)
        tb, _ = decode_state(unpack_codes(eb.fc1.state, d), 11)
        assert torch.equal(ta, tb)


def test_grouped_stacked_forward_matches_reference_and_trains():
    """Prong B: grouped=True uses stacked experts + torch._grouped_mm (no python expert loop). The
    grouped forward must equal a manual per-expert forward over the SAME stacked weights, and the
    model must train (router grad + stacked counter state moves)."""
    import torch.nn.functional as F
    if not hasattr(torch, "_grouped_mm"):
        import pytest; pytest.skip("torch._grouped_mm unavailable")
    d = 32
    torch.manual_seed(0)
    m = CounterMoEFFN(d, n_experts=4, top_k=2, C=11, grouped=True).eval()
    assert m.grouped and hasattr(m, "stacked")
    torch.manual_seed(1); x = torch.randn(6, 5, d); h = x.reshape(-1, d)
    with torch.no_grad():
        yg = m(x).reshape(-1, d)
        flat_tok, flat_exp, flat_w = m._route(h)                 # deterministic router -> same routing
        W1, W2 = m.stacked.weights()
        ref = torch.zeros_like(h)
        for e in range(m.E):
            sel = (flat_exp == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tok = flat_tok[sel]
            o = F.gelu(h[tok] @ W1[e].t()) @ W2[e].t()           # per-expert reference, same weights
            ref.index_add_(0, tok, flat_w[sel].unsqueeze(-1) * o)
        assert torch.allclose(yg, ref, atol=1e-4)                # grouped forward == per-expert ref

    torch.manual_seed(0)
    m2 = CounterMoEFFN(d, n_experts=4, top_k=2, C=11, lr=0.06, grouped=True).train()
    opt = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=3e-3)
    x = torch.randn(64, d); y = torch.randn(64, d) * 0.3
    st0 = m2.stacked.s1.clone()
    first = None
    for _ in range(120):
        out = m2(x); loss = (out - y).pow(2).mean() + 1e-2 * m2.last_aux_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if first is None: first = loss.item()
    assert loss.item() < 0.7 * first                             # learns
    assert not torch.equal(st0, m2.stacked.s1)                   # stacked counter state self-updated
    assert m2.router.weight.grad is not None                     # router still gets its gradient


def test_batched_update_equals_per_expert():
    """The vectorized [E,out,in] counter update == looping the same update per expert (same SR
    order). Pins that StackedCounterExperts.update keeps the per-expert dynamics exactly."""
    from memory_native.counter import encode_state
    from memory_native.moe_ffn import _batched_rms_update
    C, E, out, in_ = 11, 4, 8, 12
    torch.manual_seed(0)
    st = encode_state(torch.randint(-1, 2, (E, out, in_)).short(),
                      torch.zeros(E, out, in_).short(), C)
    sc = torch.full((E, out, 1), 0.2); v = torch.zeros(E, out, 1)
    gw = torch.randn(E, out, in_); active = torch.ones(E, dtype=torch.bool)
    kw = dict(C=C, lr=0.04, lr_scale=2e-4, beta=0.9, eps=1e-3)
    sb, scb, vb = st.clone(), sc.clone(), v.clone()
    torch.manual_seed(7); _batched_rms_update(sb, scb, vb, gw.clone(), active, **kw)
    torch.manual_seed(7)
    sp = st.clone(); scp = sc.clone(); vp = v.clone()
    for e in range(E):                                           # same SR draw order as the batch
        _batched_rms_update(sp[e:e+1], scp[e:e+1], vp[e:e+1], gw[e:e+1].clone(),
                            active[e:e+1], **kw)
    assert torch.equal(sb, sp)


def test_swiglu_grouped_matches_reference_and_trains():
    """SwiGLU experts (gate/up/down, GLM/Llama-style): grouped forward == per-expert reference on the
    same stacked weights, and it trains. Active MACs use the 3-matrix count."""
    import torch.nn.functional as F
    if not hasattr(torch, "_grouped_mm"):
        import pytest; pytest.skip("torch._grouped_mm unavailable")
    d = 32
    torch.manual_seed(0)
    m = CounterMoEFFN(d, n_experts=4, top_k=2, C=11, grouped=True, swiglu=True).eval()
    torch.manual_seed(1); x = torch.randn(6, 5, d); h = x.reshape(-1, d)
    with torch.no_grad():
        yg = m(x).reshape(-1, d)
        ft, fe, fw = m._route(h)
        Wg, Wu, Wd = m.stacked.weights()
        ref = torch.zeros_like(h)
        for e in range(m.E):
            sel = (fe == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tok = ft[sel]
            o = (F.silu(h[tok] @ Wg[e].t()) * (h[tok] @ Wu[e].t())) @ Wd[e].t()
            ref.index_add_(0, tok, fw[sel].unsqueeze(-1) * o)
        assert torch.allclose(yg, ref, atol=1e-4)
    assert m.active_macs_per_token() == d * m.E + m.k * (3 * d * m.h)   # 3 matmuls per SwiGLU expert

    torch.manual_seed(0)
    mm = CounterMoEFFN(d, n_experts=4, top_k=2, C=11, lr=0.06, grouped=True, swiglu=True).train()
    opt = torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad], lr=3e-3)
    xx = torch.randn(64, d); yy = torch.randn(64, d) * 0.3
    first = None
    for _ in range(120):
        out = mm(xx); loss = (out - yy).pow(2).mean() + 1e-2 * mm.last_aux_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if first is None: first = loss.item()
    assert loss.item() < 0.7 * first


def test_grouped_grad_w_matches_loop():
    """The loop-free per-expert weight-gradient (pad+bmm) == the per-segment matmul loop, bit-exact
    (incl. empty experts). This is the last per-expert loop removed from the MoE backward."""
    from memory_native.moe_ffn import _grouped_grad_w
    torch.manual_seed(0)
    M, h, d, E = 20, 6, 8, 4
    go = torch.randn(M, h); xs = torch.randn(M, d)
    for offs in (torch.tensor([5, 9, 14, 20], dtype=torch.int32),
                 torch.tensor([5, 5, 14, 20], dtype=torch.int32)):        # 2nd: expert 1 empty
        gw, active = _grouped_grad_w(go, xs, offs, E)
        starts = [0] + offs.tolist()[:-1]
        ends = offs.tolist()
        man = torch.stack([(go[s:t].t() @ xs[s:t] if t > s else torch.zeros(h, d))
                           for s, t in zip(starts, ends)])
        assert torch.allclose(gw, man, atol=1e-4)
        assert active.tolist() == [(t > s) for s, t in zip(starts, ends)]
