"""Witnesses for the GROUP-LOCAL update — the one-pass fused-epilogue enabler.

The row-scope update keys the RMS denominator and the grad-norm clip to the whole output
row, so every weight's tick depends on a full-row reduction of the CURRENT gradient — the
structural blocker for a one-pass tiled-GEMM epilogue (FUSION_PLAN lever #1: "exact blocks
it"). Group-local statistics (``v`` per scale group, denom/clip over the group) remove the
blocker BY CONSTRUCTION — every statistic lives inside one group-aligned GEMM tile. These
tests pin the three CPU-checkable claims the Triton kernel relies on:

  * locality — perturbing grad_w[0, 5] changes state only inside column 5's GROUP of row 0
    (the tile is self-contained); row scope spreads the change across the whole row;
  * degeneracy — with a single group (group == in_features) group-local IS the row math
    (identical codes/scale/v): the change is statistic granularity, not a new update rule;
  * learning — a PackedGroupScaleCounterLinear(stats_scope="group") still recovers a
    ternary teacher end-to-end, with the recovery recipe's clip=1.0 in the loop.

The Triton fused kernel (`triton_group_counter_update_fused`) mirrors
`group_counter_update_grouplocal_hashsr` and is gated on GPU at quanta-level, like the
dense kernels; those runs live in scripts/benchmark_group_kernels.py, not here.
"""
import pytest
import torch

from memory_native.counter import decode_state, encode_state
from memory_native.group_scale_kernels import (
    group_counter_update_grouplocal_hashsr,
    group_counter_update_hashsr,
)
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
from memory_native.packed import unpack_codes


def _changed_positions(scope, seed, out=4, in_=16, group=4, C=11, clip=1.0):
    """State positions that change when ONE grad_w element [0, 5] is perturbed."""
    torch.manual_seed(seed)
    codes0 = encode_state(torch.randint(-1, 2, (out, in_), dtype=torch.int16),
                          torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16), C)
    groups = (in_ + group - 1) // group
    scale0 = torch.full((out, groups), 0.3)
    v0 = torch.zeros(out, groups if scope == "group" else 1)
    perm = torch.arange(in_)
    gw = torch.randn(out, in_)
    fn = group_counter_update_grouplocal_hashsr if scope == "group" else group_counter_update_hashsr
    kw = dict(group=group, C=C, lr=0.1, lr_scale=1e-3, rms_beta=0.9, rms_eps=1e-3,
              seed=7, clip=clip)

    def run(g):
        return fn(codes0.clone(), scale0.clone(), v0.clone(), g, perm, **kw)

    base = run(gw.clone())
    g2 = gw.clone()
    g2[0, 5] += 30.0
    pert = run(g2)
    return torch.nonzero(base != pert).tolist()


def test_grouplocal_is_group_local():
    """Perturbing gw[0,5] must stay inside row 0, group 1 (cols 4..7) — the tile owns it —
    for EVERY seed; this is the invariant the fused kernel's tile independence rests on."""
    seen_any = False
    for seed in range(10):
        changed = _changed_positions("group", seed)
        seen_any = seen_any or bool(changed)
        for r, c in changed:
            assert r == 0 and 4 <= c < 8, f"seed {seed}: leaked outside row0/group1: ({r},{c})"
    assert seen_any, "perturbation must be visible for at least one seed (lr is large)"


def test_row_scope_couples_whole_row():
    """The same perturbation under row scope (clip=1.0) reaches other groups of the row —
    the documented reason a one-pass epilogue cannot serve the row math."""
    groups_touched = set()
    for seed in range(10):
        for _, c in _changed_positions("row", seed):
            groups_touched.add(c // 4)
    assert len(groups_touched) > 1, f"row scope should couple groups, got {sorted(groups_touched)}"


def test_grouplocal_degenerates_to_row_math():
    """group == in_features ⇒ bit-identical codes/scale/v to the row-scope reference."""
    torch.manual_seed(1)
    out, in_, C = 6, 32, 11
    perm = torch.randperm(in_)
    codes = encode_state(torch.randint(-1, 2, (out, in_), dtype=torch.int16),
                         torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16), C)
    scale_r = torch.rand(out, 1) * 0.2 + 0.05
    scale_g = scale_r.clone()
    v_r = torch.zeros(out, 1)
    v_g = torch.zeros(out, 1)
    codes_r, codes_g = codes.clone(), codes.clone()
    kw = dict(group=in_, C=C, lr=0.05, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3,
              residual_alpha=0.35, clip=1.0)
    for step in range(3):
        codes_r = group_counter_update_hashsr(
            codes_r, scale_r, v_r, torch.randn(out, in_, generator=torch.Generator().manual_seed(step)),
            perm, seed=step, **kw)
    for step in range(3):
        codes_g = group_counter_update_grouplocal_hashsr(
            codes_g, scale_g, v_g, torch.randn(out, in_, generator=torch.Generator().manual_seed(step)),
            perm, seed=step, **kw)
    assert torch.equal(codes_r, codes_g)
    assert torch.equal(scale_r, scale_g)
    assert torch.equal(v_r, v_g)


def test_grouplocal_decimation_is_exact_restriction():
    """active_groups leaves inactive groups BIT-untouched (state, scale, v), and the active
    groups get BIT-identically what a full update would give them — decimation is the full
    math restricted to a subset, which is only well-defined because nothing crosses a
    group boundary. This is the contract the fused-kernel grid restriction will mirror."""
    torch.manual_seed(4)
    out, in_, group, C = 8, 32, 8, 11
    groups = in_ // group
    codes0 = encode_state(torch.randint(-1, 2, (out, in_), dtype=torch.int16),
                          torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16), C)
    scale0 = torch.rand(out, groups) * 0.2 + 0.05
    v0 = torch.rand(out, groups) * 0.01
    gw = torch.randn(out, in_)
    perm = torch.randperm(in_)
    kw = dict(group=group, C=C, lr=0.05, lr_scale=2e-4, rms_beta=0.9, rms_eps=1e-3,
              seed=11, residual_alpha=0.35, clip=1.0)
    active = torch.tensor([True, False, True, False])

    sc_f, v_f = scale0.clone(), v0.clone()
    full = group_counter_update_grouplocal_hashsr(codes0.clone(), sc_f, v_f, gw.clone(),
                                                  perm, **kw)
    sc_d, v_d = scale0.clone(), v0.clone()
    dec = group_counter_update_grouplocal_hashsr(codes0.clone(), sc_d, v_d, gw.clone(),
                                                 perm, active_groups=active, **kw)
    col_active = active[torch.arange(in_) // group]
    assert torch.equal(dec[:, ~col_active], codes0[:, ~col_active])
    assert torch.equal(sc_d[:, ~active], scale0[:, ~active])
    assert torch.equal(v_d[:, ~active], v0[:, ~active])
    assert torch.equal(dec[:, col_active], full[:, col_active])
    assert torch.equal(sc_d[:, active], sc_f[:, active])
    assert torch.equal(v_d[:, active], v_f[:, active])


def test_grouplocal_rejects_row_shaped_v():
    codes = encode_state(torch.zeros(4, 16, dtype=torch.int16),
                         torch.zeros(4, 16, dtype=torch.int16), 11)
    with pytest.raises(ValueError, match="group-local v"):
        group_counter_update_grouplocal_hashsr(
            codes, torch.full((4, 4), 0.1), torch.zeros(4, 1), torch.randn(4, 16),
            torch.arange(16), group=4, C=11, lr=0.01, lr_scale=0.0,
            rms_beta=0.9, rms_eps=1e-3, seed=0)


def test_grouplocal_layer_recovers_teacher():
    """End-to-end: stats_scope='group' with the recovery recipe (clip=1.0) still learns."""
    torch.manual_seed(0)
    in_, out, group, C = 64, 32, 16, 11
    ts = 0.25
    tw = torch.randint(-1, 2, (out, in_)).to(torch.int16)
    x = torch.randn(256, in_)
    y = x @ (ts * tw.float()).t()
    lay = PackedGroupScaleCounterLinear(
        in_, out, group=group, C=C, lr=0.02, lr_scale=2e-4,
        stats_scope="group", kernel_mode="torch", local_grad_clip=1.0,
    ).train()
    lay.load_group_state(torch.full((out, in_ // group), ts),
                         torch.zeros(out, in_, dtype=torch.int16))
    assert lay.v.shape == (out, in_ // group)
    for _ in range(500):
        (lay(x) - y).pow(2).mean().backward()
    with torch.no_grad():
        mse = (lay(x) - y).pow(2).mean().item()
    t, _ = decode_state(unpack_codes(lay.state, in_), C)
    assert mse < 0.05, f"group-local layer failed to learn: mse={mse:.4f}"
    assert int(lay.sr_step) == 500


def test_grouplocal_layer_salient_stays_frozen():
    torch.manual_seed(2)
    in_, out, group, C = 32, 8, 8, 11
    lay = PackedGroupScaleCounterLinear(
        in_, out, group=group, C=C, stats_scope="group", kernel_mode="torch",
        local_grad_clip=1.0,
    ).train()
    sal_idx = torch.tensor([3, in_ + 7, 5 * in_ + 1], dtype=torch.int32)
    sal_val = torch.tensor([0.5, -0.25, 0.125], dtype=torch.float16)
    lay.load_group_state(torch.full((out, in_ // group), 0.2),
                         torch.randint(-1, 2, (out, in_), dtype=torch.int16),
                         salient_idx=sal_idx, salient_val=sal_val)
    x = torch.randn(64, in_)
    for _ in range(5):
        lay(x).pow(2).mean().backward()
    codes = unpack_codes(lay.state, in_)
    t, c = decode_state(codes, C)
    w = torch.empty_like(t)
    w[:, lay.perm.long()] = t
    cr = torch.empty_like(c)
    cr[:, lay.perm.long()] = c
    flat_t = w.reshape(-1)[sal_idx.long()]
    flat_c = cr.reshape(-1)[sal_idx.long()]
    assert (flat_t == 0).all() and (flat_c == 0).all(), "salient base codes must stay zero"


def test_grouplocal_state_dict_roundtrip():
    torch.manual_seed(3)
    in_, out, group = 32, 8, 8
    a = PackedGroupScaleCounterLinear(in_, out, group=group, stats_scope="group",
                                      kernel_mode="torch").train()
    a.load_group_state(torch.full((out, in_ // group), 0.2),
                       torch.randint(-1, 2, (out, in_), dtype=torch.int16))
    x = torch.randn(16, in_)
    for _ in range(3):
        a(x).pow(2).mean().backward()
    b = PackedGroupScaleCounterLinear(in_, out, group=group, stats_scope="group",
                                      kernel_mode="torch")
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.state, b.state) and torch.equal(a.v, b.v)
    with torch.no_grad():
        assert torch.equal(a.eval()(x), b.eval()(x))
    # Scopes are checkpoint-incompatible on purpose: v shapes differ.
    row = PackedGroupScaleCounterLinear(in_, out, group=group, stats_scope="row",
                                        kernel_mode="torch")
    with pytest.raises(RuntimeError):
        row.load_state_dict(a.state_dict())


def test_stats_scope_validation():
    with pytest.raises(ValueError, match="stats_scope"):
        PackedGroupScaleCounterLinear(16, 8, stats_scope="column")
