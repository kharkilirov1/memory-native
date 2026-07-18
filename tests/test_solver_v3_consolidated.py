"""Consolidated solver-v3 gates: Stage-A ingredients on the agent v3 refine cycle plus
the packed salient channel.

Chain of claims, each pinned by a separate test:
  * grid='itf' (A5) fits skewed blocks strictly better than the symmetric grid and its
    coordinate descent is monotone;
  * scale_refit='align' (A7) solves the exact joint per-row scale problem in the
    H-metric and never loses to the greedy hdiag refit on the same support;
  * salient_first (A4.1) splits heavy hitters out BEFORE the sweep and reduces the
    H-weighted error on heavy-tailed weights;
  * the solver's salient return is an exact decomposition: Q = S*t + salient overrides
    in ORIGINAL column order, with t = 0 at salient entries;
  * PackedGroupScaleCounterLinear stores the salient channel bit-exactly: base codes
    are zero at salient entries, visible_weight equals the solver reconstruction up to
    fp16 rounding of the override values, and the sparse-add path (what the Triton
    kernels add) matches the dense reference;
  * salient entries stay FROZEN through strict/reference updates;
  * ptq_warm_start wires it end-to-end, and an itf start is packed through the exact
    sym re-solve (the packed format is sym-scale).
"""
import copy

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from memory_native.counter import decode_state
from memory_native.donor.ptq import (
    align_scales_output,
    gptq_group_ternary,
    group_residual_counter,
    itf_grid,
    optimal_ternary,
    ptq_warm_start,
)
from memory_native.group_scale_counter import GroupScaleCounterLinear
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear
from memory_native.packed import unpack_codes


def _corr_H(seed, in_f, n=1024):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_f, generator=g) @ torch.randn(in_f, in_f, generator=g) * 0.1
    return X.t() @ X


def _herr(w, q, H):
    e = w - q
    return float(((e @ H) * e).sum())


def _permute_recon(S, t, perm, group, grid="sym"):
    """Dense reconstruction from PERMUTED-group scales + original-order t."""
    cols = t.shape[1]
    gidx = torch.arange(cols) // group
    t_perm = t[:, perm].float()
    if grid == "itf":
        rec_perm = S[:, gidx, 0] * t_perm.clamp(min=0) + S[:, gidx, 1] * t_perm.clamp(max=0)
    else:
        rec_perm = S[:, gidx] * t_perm
    rec = torch.empty_like(rec_perm)
    rec[:, perm] = rec_perm
    return rec


# --- solver ingredients ------------------------------------------------------


def test_itf_grid_skewed_block_wins_and_is_monotone():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.05
    w[:, 64:] = (w[:, 64:] - 0.30) * 3.0          # heavy negative lobe, small positive lobe
    s_sym, t_sym = optimal_ternary(w)
    e_sym = float(((w - s_sym * t_sym) ** 2).sum())
    errs = []
    for iters in (1, 2, 3):
        sp, sn, t_itf = itf_grid(w, iters=iters)
        q = torch.where(t_itf > 0, sp.unsqueeze(1), torch.zeros_like(w))
        q = torch.where(t_itf < 0, -sn.unsqueeze(1), q)
        errs.append(float(((w - q) ** 2).sum()))
    assert errs[-1] < e_sym, (errs, e_sym)                # strict win on skewed blocks
    assert errs[1] <= errs[0] + 1e-6 and errs[2] <= errs[1] + 1e-6   # monotone descent


def test_itf_grid_per_lobe_init_escapes_shared_scale_trap():
    torch.manual_seed(1)
    w = torch.randn(8, 128) * 0.05
    w[:, :100] = (w[:, :100] - 0.4) * 4.0        # dominant negative lobe
    w[:, 100:] = w[:, 100:] * 0.2 + 0.02         # tiny positive lobe
    sp, sn, t = itf_grid(w, iters=3)
    assert (t > 0).any(), "opposite-sign lobe must not collapse to zero support"
    assert sn.mean() > 5 * sp.mean(), (sn.mean(), sp.mean())


def test_align_refit_never_loses_to_hdiag_on_same_cycle():
    torch.manual_seed(2)
    in_f, out_f = 128, 32
    H = _corr_H(3, in_f)
    w = torch.randn(out_f, in_f) * 0.05
    _, _, _, perm, _ = gptq_group_ternary(w, H, group=64, refine_scale=False, return_perm=True)
    base_q, _, base_t, _, _ = gptq_group_ternary(w, H, group=64, refine_scale=False,
                                                 return_perm=True)
    w_perm = w[:, perm]
    Hp = H[perm][:, perm]
    t_perm = base_t[:, perm].float()
    s_h, q_h = align_scales_output(w_perm, t_perm, Hp, group=64, grid="sym")
    e_align = _herr(w_perm, q_h, Hp)
    e_base = _herr(w_perm, base_q[:, perm], Hp)
    assert e_align <= e_base * 1.001, (e_align, e_base)
    # and through the solver API: align refit <= hdiag refit on the same cycle
    q_hd, _, _ = gptq_group_ternary(w, H, group=64, scale_refit="hdiag")
    q_al, _, _ = gptq_group_ternary(w, H, group=64, scale_refit="align")
    assert _herr(w, q_al, H) <= _herr(w, q_hd, H) * 1.001


def test_salient_first_reduces_error_on_heavy_tail():
    torch.manual_seed(4)
    in_f, out_f = 128, 32
    H = _corr_H(5, in_f)
    w = torch.randn(out_f, in_f) * 0.05
    hot = torch.rand(out_f, in_f) < 0.02
    w = torch.where(hot, w * 12.0, w)            # 2% heavy hitters
    q0, _, _ = gptq_group_ternary(w, H, group=64)
    q1, _, _ = gptq_group_ternary(w, H, group=64, salient_first=0.02)
    assert _herr(w, q1, H) < _herr(w, q0, H), (_herr(w, q1, H), _herr(w, q0, H))


def test_salient_return_is_exact_decomposition():
    torch.manual_seed(6)
    in_f, out_f, group = 64, 16, 16
    H = _corr_H(7, in_f)
    w = torch.randn(out_f, in_f) * 0.05
    w[0, 3] = 0.8
    q, S, t, perm, _, (idx, val) = gptq_group_ternary(
        w, H, group=group, salient_first=0.05, scale_refit="align",
        return_perm=True, return_salient=True)
    assert idx.dtype == torch.int32 and idx.numel() > 0
    rec = _permute_recon(S, t, perm, group)
    qsal = torch.zeros_like(w).reshape(-1)
    qsal[idx.long()] = val
    rec = rec + qsal.view_as(w)
    assert torch.allclose(rec, q, atol=1e-6)     # Q = S*t + salient, bit-exact
    assert (t.reshape(-1)[idx.long()] == 0).all()  # salient owns its entries


def test_itf_reconstruction_identity_through_solver():
    torch.manual_seed(8)
    in_f, out_f, group = 64, 16, 16
    H = _corr_H(9, in_f)
    w = torch.randn(out_f, in_f) * 0.05
    q, S, t, perm, _ = gptq_group_ternary(w, H, group=group, grid="itf",
                                          scale_refit="align", return_perm=True)
    assert S.shape == (out_f, in_f // group, 2)
    rec = _permute_recon(S, t, perm, group, grid="itf")
    assert torch.allclose(rec, q, atol=1e-6)


# --- packed salient channel ----------------------------------------------------


def _packed_with_salient(seed=0, group=8, in_f=32, out_f=12, salient_first=0.05):
    torch.manual_seed(seed)
    H = _corr_H(seed + 1, in_f, n=256)
    w = torch.randn(out_f, in_f) * 0.05
    w[0, 3] = 0.8
    q, S, t, perm, wadj, (idx, val) = gptq_group_ternary(
        w, H, group=group, salient_first=salient_first, scale_refit="align",
        return_perm=True, return_salient=True)
    c = group_residual_counter(wadj, S, t, perm, group, 11)
    c = c.clone()
    c.reshape(-1)[idx.long()] = 0
    layer = PackedGroupScaleCounterLinear(in_f, out_f, group=group, C=11,
                                          perm=perm, kernel_mode="torch")
    layer.load_group_state(S, t, c, perm, salient_idx=idx, salient_val=val)
    return layer, q, w, idx, val, perm


def test_packed_salient_roundtrip_bit_exact():
    layer, q, w, idx, val, perm = _packed_with_salient()
    vw = layer.visible_weight()
    # fp16 rounding of the overrides is the only allowed deviation
    assert torch.allclose(vw, q, atol=2e-3)
    # base codes are zero at salient entries (the salient channel owns them)
    td, cd = decode_state(unpack_codes(layer.state, layer.in_features), layer.C)
    inv = torch.argsort(perm)
    o, j = idx.long() // layer.in_features, idx.long() % layer.in_features
    assert (td[o, inv[j]] == 0).all() and (cd[o, inv[j]] == 0).all()
    # forward equals the dense reference
    x = torch.randn(5, layer.in_features)
    with torch.no_grad():
        assert torch.allclose(layer(x), x @ vw.t(), atol=1e-6)


def test_packed_salient_sparse_correction_matches_dense():
    layer, q, w, idx, val, perm = _packed_with_salient()
    x = torch.randn(5, layer.in_features)
    with torch.no_grad():
        vw = layer.visible_weight()
        base = vw.clone()
        base.reshape(-1)[idx.long()] = 0           # what the Triton base kernel sees
        A = layer._salient_sparse()
        y_emulated = x @ base.t() + torch.sparse.mm(A, x.t()).t()
        assert torch.allclose(y_emulated, layer(x), atol=1e-6)
        go = torch.randn(5, layer.out_features)
        gx_base = go @ base
        gx_emulated = gx_base + torch.sparse.mm(A.t(), go.t()).t()
        assert torch.allclose(gx_emulated, go @ vw, atol=1e-6)


def test_packed_salient_frozen_during_updates():
    layer, q, w, idx, val, perm = _packed_with_salient()
    inv = torch.argsort(perm)
    o, j = idx.long() // layer.in_features, idx.long() % layer.in_features
    x = torch.randn(16, layer.in_features)
    layer.train()
    for _ in range(3):
        (layer(x).square().mean()).backward()
    layer.eval()
    td, cd = decode_state(unpack_codes(layer.state, layer.in_features), layer.C)
    assert (td[o, inv[j]] == 0).all() and (cd[o, inv[j]] == 0).all()
    with torch.no_grad():
        kept = layer.visible_weight().reshape(-1)[idx.long()]
    assert torch.allclose(kept, val.half().float(), atol=1e-3)
    assert int(layer.sr_step) == 3               # review-fold: SR counter mirrored


def test_persistent_bytes_and_stats_account_salient():
    layer, _, _, idx, _, _ = _packed_with_salient()
    n = idx.numel()
    plain = PackedGroupScaleCounterLinear(layer.in_features, layer.out_features,
                                          group=layer.group, C=layer.C)
    assert layer.persistent_bytes() == plain.persistent_bytes() + n * (4 + 2)
    stats = layer.state_statistics()
    assert stats["salient_fraction"] == pytest.approx(n / (layer.out_features * layer.in_features))
    assert "flip_rate_alt" in stats and "sr_step" in stats


def test_reference_layer_salient_roundtrip_and_freeze():
    torch.manual_seed(11)
    in_f, out_f, group = 24, 8, 8
    H = _corr_H(12, in_f, n=256)
    w = torch.randn(out_f, in_f) * 0.05
    w[0, 3] = 0.6
    q, S, t, perm, wadj, (idx, val) = gptq_group_ternary(
        w, H, group=group, salient_first=0.06, scale_refit="align",
        return_perm=True, return_salient=True)
    c = group_residual_counter(wadj, S, t, perm, group, 11)
    c = c.clone()
    c.reshape(-1)[idx.long()] = 0
    layer = GroupScaleCounterLinear(in_f, out_f, group=group, C=11, perm=perm)
    layer.load_group_state(S, t, c, perm, salient_idx=idx, salient_val=val)
    assert torch.allclose(layer.visible_weight(), q, atol=2e-3)
    x = torch.randn(16, in_f)
    layer.train()
    (layer(x).square().mean()).backward()
    layer.eval()
    t2, c2 = decode_state(layer.state, layer.C)
    assert (t2.reshape(-1)[idx.long()] == 0).all()
    assert (c2.reshape(-1)[idx.long()] == 0).all()


# --- warm start end-to-end -----------------------------------------------------


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 12, bias=True)

    def forward(self, x):
        return self.proj(x)


def test_warm_start_group_salient_end_to_end():
    torch.manual_seed(0)
    model = _Tiny()
    original = copy.deepcopy(model)
    calib = [torch.randn(4, 5, 16), torch.randn(3, 5, 16)]
    xcal = torch.cat([x.reshape(-1, 16) for x in calib], dim=0)
    q, _, _, _, _, (idx, val) = gptq_group_ternary(
        original.proj.weight, xcal.t() @ xcal, group=8, refine_iters=2,
        salient_first=0.06, scale_refit="align", return_perm=True, return_salient=True)
    report = ptq_warm_start(
        model, calib, mode="gptq_group", kind="counter_packed",
        group=8, C=11, progress=False, kernel_mode="torch",
        salient_first=0.06, scale_refit="align",
    )
    assert report.coeffs == 16 * 12
    counter = model.proj.counter
    assert isinstance(counter, PackedGroupScaleCounterLinear)
    assert counter.salient_idx.numel() == idx.numel()
    assert torch.allclose(counter.visible_weight(), q, atol=2e-3)
    x = torch.randn(2, 16)
    with torch.no_grad():
        assert torch.allclose(model(x), x @ counter.visible_weight().t() + original.proj.bias,
                              atol=1e-6)


def test_warm_start_itf_packs_through_exact_sym_resolve():
    torch.manual_seed(1)
    model = _Tiny()
    original = copy.deepcopy(model)
    calib = [torch.randn(4, 5, 16), torch.randn(3, 5, 16)]
    xcal = torch.cat([x.reshape(-1, 16) for x in calib], dim=0)
    H = xcal.t() @ xcal
    w = original.proj.weight
    # solver-side reference: itf support, then exact sym re-solve
    _, _, t, perm, wadj, _ = gptq_group_ternary(
        w, H, group=8, refine_iters=2, grid="itf", scale_refit="align",
        return_perm=True, return_salient=True)
    Hp = H[perm][:, perm]
    S_sym, q_sym = align_scales_output(w[:, perm], t[:, perm].float(), Hp, group=8, grid="sym")
    rec = _permute_recon(S_sym, t, perm, 8)
    assert torch.allclose(rec, q_sym[:, torch.argsort(perm)], atol=1e-6)
    report = ptq_warm_start(
        model, calib, mode="gptq_group", kind="counter_packed",
        group=8, C=11, progress=False, kernel_mode="torch",
        grid="itf", scale_refit="align",
    )
    counter = model.proj.counter
    assert isinstance(counter, PackedGroupScaleCounterLinear)
    # packed visible weight IS the exact sym re-solve on the itf support
    assert torch.allclose(counter.visible_weight(), rec, atol=1e-5)
    # and it does not regress vs the plain sym-grid packed start on these weights
    q_plain, _, _ = gptq_group_ternary(w, H, group=8, refine_iters=2)
    assert _herr(w, counter.visible_weight(), H) <= _herr(w, q_plain, H) * 1.05


def test_packed_salient_state_dict_roundtrip_into_fresh_layer():
    """Checkpoints with a salient channel must load into freshly built layers (the resume
    path constructs layers with EMPTY salient buffers) and keep behaving identically:
    same visible weight, rebuilt non-persistent act-order positions, salient still frozen."""
    layer, _, _, idx, val, perm = _packed_with_salient(seed=11)
    x_warm = torch.randn(9, layer.in_features)
    layer.train()
    (layer(x_warm).square().mean()).backward()          # sr_step=1, state moved
    layer.eval()
    sd = copy.deepcopy(layer.state_dict())

    fresh = PackedGroupScaleCounterLinear(
        layer.in_features, layer.out_features, group=layer.group, C=layer.C,
        perm=perm, kernel_mode="torch")
    fresh.load_state_dict(sd)                            # was: RuntimeError size mismatch
    assert fresh.salient_idx.numel() == idx.numel()
    assert torch.equal(fresh._salient_perm_flat, layer._salient_perm_flat)
    assert int(fresh.sr_step) == int(layer.sr_step)
    with torch.no_grad():
        assert torch.equal(fresh.visible_weight(), layer.visible_weight())

    # identical updates after the restore: bit-exact state AND salient stays frozen
    x2 = torch.randn(7, layer.in_features)
    go2 = torch.randn(7, layer.out_features)
    layer._update_from_io(x2, go2)
    fresh._update_from_io(x2, go2)
    assert torch.equal(layer.state, fresh.state)
    assert torch.equal(layer.scale, fresh.scale)
    with torch.no_grad():
        kept = fresh.visible_weight().reshape(-1)[idx.long()]
    assert torch.allclose(kept, val.half().float(), atol=1e-3)


def test_reference_salient_state_dict_roundtrip_into_fresh_layer():
    layer, _, _, idx, val, perm = _packed_with_salient(seed=12)
    ref = GroupScaleCounterLinear(layer.in_features, layer.out_features,
                                  group=layer.group, C=layer.C, perm=perm)
    t, c = decode_state(unpack_codes(layer.state, layer.in_features), layer.C)
    inv = torch.argsort(perm)
    ref.load_group_state(layer.scale, t[:, inv].to(torch.int16), c[:, inv].to(torch.int16),
                         perm, salient_idx=idx, salient_val=val)
    sd = copy.deepcopy(ref.state_dict())
    fresh = GroupScaleCounterLinear(layer.in_features, layer.out_features,
                                    group=layer.group, C=layer.C, perm=perm)
    fresh.load_state_dict(sd)
    assert fresh.salient_idx.numel() == idx.numel()
    with torch.no_grad():
        assert torch.equal(fresh.visible_weight(), ref.visible_weight())
