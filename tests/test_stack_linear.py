"""StackCounterLinear (M-STACK): 2:4 group-counter base + slow-fast low-rank residual compose."""
import math

import torch

from memory_native.stack_linear import StackCounterLinear


def test_only_AB_are_fp_parameters():
    """The base is counter-state; only the low-rank residual A,B are fp Parameters."""
    lay = StackCounterLinear(32, 16, rank=8)
    names = {n for n, _ in lay.named_parameters()}
    assert names == {"A", "B"}


def test_base_stays_2to4_sparse():
    torch.manual_seed(0)
    lay = StackCounterLinear(16, 8, rank=4, merge_every=4).train()
    opt = torch.optim.AdamW(lay.fast_parameters(), lr=3e-3)
    x = torch.randn(8, 16)
    for _ in range(12):
        (lay(x) ** 2).sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    vis = lay.base.visible_mask()
    assert torch.equal(vis.reshape(8, 4, 4).sum(-1), torch.full((8, 4), 2.0))


def test_merge_fires_on_schedule_and_base_changes():
    torch.manual_seed(0)
    lay = StackCounterLinear(16, 16, rank=8, merge_every=5).train()
    opt = torch.optim.AdamW(lay.fast_parameters(), lr=5e-3)
    x = torch.randn(16, 16)
    s0 = lay.base.state.clone()
    for _ in range(20):
        (lay(x) ** 2).sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    # merge is LAZY (a step that hits the schedule folds at the START of the next forward), so the
    # pending merge after the final step never fires -> 3 merges over 20 steps at merge_every=5.
    assert int(lay.merge_count) == 3
    assert not torch.equal(lay.base.state, s0)             # base absorbed the residual


def test_rank0_is_plain_group_counter():
    """rank=0 disables the fast path -> the base self-updates like a plain group-counter."""
    lay = StackCounterLinear(16, 8, rank=0).train()
    assert lay.fast_parameters() == []
    assert lay.base.update_enabled is True
    s0 = lay.base.state.clone()
    (lay(torch.randn(8, 16)) ** 2).sum().backward()
    assert not torch.equal(lay.base.state, s0)             # base ticks itself


def test_stack_recovers_a_teacher():
    """The composed lever trains: 2:4 base + slow-fast residual reach a low MSE together."""
    torch.manual_seed(0)
    n = 64
    lay = StackCounterLinear(n, n, rank=16, merge_every=16, C=11, lr=0.04, lr_scale=2e-4).train()
    opt = torch.optim.AdamW(lay.fast_parameters(), lr=3e-3)
    w = torch.randn(n, n) * (0.25 * math.sqrt(2.0 / n))
    x = torch.randn(256, n); y = x @ w.t()
    first = (lay(x) - y).pow(2).mean().item()
    for _ in range(300):
        (lay(x) - y).pow(2).mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    assert (lay(x) - y).pow(2).mean().item() < 0.5 * first
