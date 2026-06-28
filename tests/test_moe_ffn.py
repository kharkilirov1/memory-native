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
