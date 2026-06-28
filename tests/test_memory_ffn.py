"""CounterMemoryFFN (plan M1): retrieval memory with counter-state values, exact-for-active update.

The defining property: a cell a token did NOT retrieve has EXACTLY zero gradient, so updating only
retrieved rows is the exact gradient (not a sparse approximation). These pin that, plus the
capacity-without-active-FLOPs claim and that the router still learns by autograd.
"""
import math

import torch

from memory_native.memory_ffn import CounterMemoryFFN, CounterValueMemory


def test_only_retrieved_rows_update():
    """Exact-for-active: rows that were read change; every other row is byte-identical."""
    torch.manual_seed(0)
    ffn = CounterMemoryFFN(32, n_cells=1024, k=8, key_dim=16).train()
    x = torch.randn(2, 8, 32)
    state0 = ffn.values.state.clone()
    # capture exactly which cells get read this step
    q = ffn.query(x.reshape(-1, 32))
    _, ids = ffn._retrieve(q)
    read = torch.unique(ids.reshape(-1))
    (ffn(x) ** 2).sum().backward()
    changed = torch.unique(torch.nonzero((ffn.values.state != state0).any(dim=1)).reshape(-1))
    # every changed row must be one that was read; unread rows are untouched
    assert set(changed.tolist()).issubset(set(read.tolist()))
    not_read = torch.ones(ffn.E, dtype=torch.bool); not_read[read] = False
    assert torch.equal(ffn.values.state[not_read], state0[not_read])


def test_shared_cell_accumulates_then_ticks_once():
    """A cell read by many tokens must aggregate their grads (sum) and tick ONCE, not per read."""
    torch.manual_seed(0)
    mem = CounterValueMemory(64, 16, C=11, lr=0.1)
    flat = torch.tensor([5, 5, 5, 5])                 # one cell, read 4x
    grad = torch.randn(4, 16)
    state0 = mem.state.clone()
    mem._update_active(flat, grad)
    assert int(mem.rows_touched) == 1                 # one unique cell touched (not 4)
    assert not torch.equal(mem.state[5], state0[5])   # cell 5 moved
    mask = torch.ones(64, dtype=torch.bool); mask[5] = False
    assert torch.equal(mem.state[mask], state0[mask]) # every other cell byte-identical


def test_router_receives_gradients():
    torch.manual_seed(0)
    ffn = CounterMemoryFFN(32, n_cells=256, k=4, key_dim=8).train()
    x = torch.randn(3, 4, 32, requires_grad=True)
    (ffn(x) ** 2).sum().backward()
    assert ffn.query.weight.grad is not None and torch.isfinite(ffn.query.weight.grad).all()
    assert ffn.k1.grad is not None and ffn.k2.grad is not None
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_capacity_grows_without_active_flops():
    """Bigger E (more capacity / persistent bytes) at fixed k,key_dim must NOT raise active MACs
    more than the sqrt(E) sub-key term -- the whole point vs a dense FFN (linear in width)."""
    small = CounterMemoryFFN(64, n_cells=4096, k=8, key_dim=16)
    big = CounterMemoryFFN(64, n_cells=65536, k=8, key_dim=16)   # 16x the cells
    assert big.values.persistent_bytes() > 15 * small.values.persistent_bytes()
    # 16x the cells: a dense layer would be 16x the active MACs. Here only the 2*sqrt(E)*dk term
    # grows (by sqrt(16)=4x) and the rest is constant, so total active MACs grow < 4x, not 16x.
    assert big.active_macs_per_token() < 4 * small.active_macs_per_token()
    assert big.active_macs_per_token() < 0.5 * 16 * small.active_macs_per_token()


def test_memory_ffn_can_fit_a_target():
    """Sanity: the counter value table + router actually learn (loss drops a lot)."""
    torch.manual_seed(0)
    d, N = 32, 128
    ffn = CounterMemoryFFN(d, n_cells=1024, k=8, key_dim=16, lr=0.06, lr_scale=3e-4).train()
    opt = torch.optim.AdamW([p for p in ffn.parameters() if p.requires_grad], lr=3e-3)
    x = torch.randn(N, d)
    y = torch.randn(N, d) * 0.3
    first = None
    for _ in range(300):
        out = ffn(x)
        loss = (out - y).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.7 * first, (first, loss.item())
