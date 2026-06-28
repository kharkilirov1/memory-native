"""GroupCounterLinear (plan M2): 2:4 structured-sparse VISIBLE weight, error-feedback on the mask.

The visible weight keeps exactly 2 of every 4 (hardware-sparse); the masked weights are NOT dead --
the update ticks them too (error-feedback), so they can flip back into the visible top-2. Hysteresis
makes the visible set sticky so it does not thrash. These pin those properties.
"""
import math

import torch

from memory_native.group_counter import GroupCounterLinear, two_four_mask


def test_two_four_mask_keeps_exactly_two_of_four():
    imp = torch.tensor([[4.0, 1.0, 3.0, 2.0, 9.0, 8.0, 0.0, 1.0]])
    m = two_four_mask(imp, group=4, keep=2)
    assert m.reshape(2, 4).sum(-1).tolist() == [2.0, 2.0]
    assert m.tolist() == [[1, 0, 1, 0, 1, 1, 0, 0]]            # top-2 per group


def test_visible_weight_is_2to4_sparse():
    torch.manual_seed(0)
    lay = GroupCounterLinear(16, 8, C=11).train()
    (lay(torch.randn(4, 16)) ** 2).sum().backward()
    vis = lay.visible_mask()
    assert torch.equal(vis.reshape(8, 4, 4).sum(-1), torch.full((8, 4), 2.0))


def test_masked_weights_keep_accumulating_with_error_feedback():
    """update_all=True: invisible weights still tick (error-feedback) -> not dead."""
    torch.manual_seed(0)
    lay = GroupCounterLinear(16, 8, C=11, lr=0.2, update_all=True).train()
    vis = lay.visible_mask().bool()
    s0 = lay.state.clone()
    (lay(torch.randn(32, 16)) ** 2).sum().backward()
    moved = lay.state != s0
    invisible = ~vis
    assert int((moved & invisible).sum()) > 0                  # masked weights moved


def test_pruning_ablation_freezes_masked_weights():
    """update_all=False: only visible weights tick; masked weights never move (the dead-weight
    failure mode the error-feedback path avoids)."""
    torch.manual_seed(0)
    lay = GroupCounterLinear(16, 8, C=11, lr=0.2, update_all=False).train()
    vis = lay.visible_mask().bool()
    s0 = lay.state.clone()
    (lay(torch.randn(32, 16)) ** 2).sum().backward()
    moved = lay.state != s0
    assert int((moved & ~vis).sum()) == 0                      # masked weights frozen


def test_hysteresis_reduces_visible_set_churn():
    """Higher hysteresis -> the visible 2:4 set is stickier -> fewer visibility changes."""
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    y = torch.randn(64, 16) * 0.3

    def churn(hy):
        torch.manual_seed(0)
        lay = GroupCounterLinear(32, 16, C=11, lr=0.1, hysteresis=hy).train()
        prev = lay.visible_mask().clone()
        tot = 0
        for _ in range(60):
            (lay(x) - y).pow(2).mean().backward()
            v = lay.visible_mask(); tot += int((v != prev).sum()); prev = v.clone()
        return tot

    assert churn(4.0) <= churn(0.0)                            # stickier set churns no more
    assert churn(0.0) > 0                                       # and hyst=0 really does churn


def test_grad_x_consistent_with_masked_forward():
    """grad_x must use the SAME masked weight the forward used (not the dense weight)."""
    torch.manual_seed(0)
    lay = GroupCounterLinear(16, 8, C=11).eval()               # eval: deterministic, committed mask
    x = torch.randn(5, 16, requires_grad=True)
    y = lay(x)
    g = torch.autograd.grad(y.sum(), x)[0]
    # reference grad_x = 1 @ W_vis ; W_vis = scale*t*vis
    t, _ = lay._decode()
    w_vis = (lay.scale * t) * lay.vis
    ref = torch.ones(5, 8) @ w_vis
    assert torch.allclose(g, ref, atol=1e-5)


def test_group_counter_fits_a_target():
    torch.manual_seed(0)
    n = 64
    lay = GroupCounterLinear(n, n, C=11, lr=0.05, lr_scale=2e-4, hysteresis=2.0).train()
    w = torch.randn(n, n) * (0.25 * math.sqrt(2.0 / n))
    x = torch.randn(256, n); y = x @ w.t()
    first = (lay(x) - y).pow(2).mean().item()
    for _ in range(400):
        (lay(x) - y).pow(2).mean().backward()
    assert (lay(x) - y).pow(2).mean().item() < 0.6 * first
