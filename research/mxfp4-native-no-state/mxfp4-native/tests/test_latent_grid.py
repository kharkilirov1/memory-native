"""mx6/latent_grid.py (LatentGrid, LatentGridLinear) and the probes/lg_ab.py wrapper. CPU, fast, mechanism-level;
every test that asserts an effect has a null twin (the mechanism removed / the legacy variant must FAIL it).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mx6 import update as mx_update                                    # noqa: E402
from mx6.latent_grid import (E2M1_MAGS, E2M1_SIGNED, LatentGrid, LatentGridLinear,   # noqa: E402
                             get_grid, pack_bits, unpack_bits)
from mx6.layer import MXNLinear                                        # noqa: E402
from mx6.packed_na import PackedNA                                     # noqa: E402

V = torch.tensor(E2M1_SIGNED, dtype=torch.float64)
KM = {8: 18, 7: 9, 6: 4}


def ref_rtn_rung(a: torch.Tensor) -> torch.Tensor:
    """Independent reference: nearest signed E2M1 value, exact ties AWAY from zero (larger |V| wins)."""
    score = (a.double().unsqueeze(-1) - V).abs() - 1e-9 * V.abs()
    return score.argmin(-1)


def mc_shift(grid: LatentGrid, start_value: float, delta: float, mode: str, n: int = 200_000, seed: int = 1):
    """(mean, standard error) of G[new] - G[start] over n independent stochastic roundings of one move."""
    T = grid.tables("cpu")
    i0 = E2M1_SIGNED.index(start_value) * grid.m
    gen = torch.Generator().manual_seed(seed)
    new = grid.move(torch.full((n,), i0), torch.full((n,), float(delta)), gen, mode)
    sh = (T.values[new] - T.values[i0]).double()
    return float(sh.mean()), max(float(sh.std()) / math.sqrt(n), 1e-9)


# ── grid ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
def test_grid_contains_every_e2m1_point_and_is_odd_symmetric(k):
    g = LatentGrid(k)
    m, n = g.m, g.n_pts
    assert n == 14 * m + 1 <= 2 ** k < 14 * (m + 1) + 1               # the largest m that fits k bits
    assert bool((g.values[1:] > g.values[:-1]).all())                 # strictly increasing
    assert torch.equal(g.values, -g.values.flip(0)) and float(g.values[7 * m]) == 0.0   # odd, 0 in the middle
    for r, v in enumerate(E2M1_SIGNED):
        assert float(g.values[r * m]) == v                            # every E2M1 value is a grid point (exact)
    sub = g.values[1:] - g.values[:-1]
    for j in range(14):                                               # equal sub-steps inside each gap
        gap = E2M1_SIGNED[j + 1] - E2M1_SIGNED[j]
        assert torch.allclose(sub[j * m:(j + 1) * m], torch.full((m,), gap / m), atol=1e-5)
    assert float(sub[7 * m]) < float(sub[-1])                         # relative resolution: finer near 0 than at 6
    # null twin: a uniform grid with the same number of points does NOT contain the E2M1 values
    uni = torch.linspace(-6, 6, n)
    assert not all(bool((uni - v).abs().min() < 1e-6) for v in E2M1_SIGNED)


def test_k_to_m_and_capacity():
    for k, m in KM.items():
        g = get_grid(k)
        assert (g.m, g.n_pts, g.bits) == (m, 14 * m + 1, k)
    with pytest.raises(ValueError):                                    # null twin: m too large for k bits
        LatentGrid(8, m=19)
    with pytest.raises(ValueError):
        LatentGrid(3)
    assert LatentGrid(8, m=18).n_pts == 253


def test_k4_grid_is_plain_e2m1_stateless():
    g = LatentGrid(4)
    assert g.m == 1 and torch.equal(g.values, torch.tensor(E2M1_SIGNED)) and torch.equal(g.rung, torch.arange(15))
    assert torch.equal(g.deploy, g.values)
    layer = LatentGridLinear(32, 16, k=4)
    assert torch.equal(layer.weight(), layer.latent_weight())          # no sub-grid: latent == deployed


# ── deploy == RTN(latent) ────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [6, 7, 8])
def test_deploy_table_is_rtn_to_e2m1_ties_away_from_zero(k):
    g = get_grid(k)
    ref = ref_rtn_rung(g.values)
    assert torch.equal(g.rung, ref)
    assert torch.equal(g.rung, g.na.round_idx(g.values))                # = PackedNA.round_idx semantics
    assert torch.equal(g.deploy, torch.tensor(E2M1_SIGNED)[g.rung])
    assert torch.equal(g.deploy, -g.deploy.flip(0))                    # odd map: deploy(-a) == -deploy(a)
    if k % 2 == 0:                                                     # m even: exact midpoints are grid points
        m = g.m
        for lo, hi, expect in [(0.0, 0.5, 0.5), (2.0, 3.0, 3.0), (4.0, 6.0, 6.0)]:
            mid = int(E2M1_SIGNED.index(lo) * m + m // 2)
            assert float(g.values[mid]) == (lo + hi) / 2 and float(g.deploy[mid]) == expect   # tie up (away)
            assert float(g.deploy[14 * m - mid]) == -expect                                    # and mirrored


def _set_random_index(layer, seed=0):
    gen = torch.Generator().manual_seed(seed)
    layer.set_index_(torch.randint(0, layer.fmt.n_pts, layer.idx.shape, generator=gen))


def _hysteretic_rung(latent, prev_rung, frac=0.75):
    """Null twin of the deploy: keep the previous rung until the latent is `frac` of the way to the new one."""
    rtn = ref_rtn_rung(latent)
    Vf = V.to(latent.dtype)
    keep = (rtn != prev_rung) & ((latent - Vf[prev_rung]).abs() < frac * (Vf[rtn] - Vf[prev_rung]).abs())
    return torch.where(keep, prev_rung, rtn)


def test_layer_weight_is_rtn_of_the_latent_after_every_update_not_stale_not_hysteretic():
    torch.manual_seed(0)
    layer = LatentGridLinear(64, 32, k=8, lr=0.01)
    _set_random_index(layer)
    s = layer.fmt.na.expand_scale(layer._scale, layer._tscale)
    T = layer.fmt.tables("cpu")
    lat = T.values[layer.idx.long()]
    assert torch.equal(layer.weight(), V.float()[ref_rtn_rung(lat)] * s)
    w_before, rung_before = layer.weight().clone(), ref_rtn_rung(lat)
    layer.generator = torch.Generator().manual_seed(3)
    layer.apply_update_(torch.randn(32, 64, generator=torch.Generator().manual_seed(4)))
    lat2 = T.values[layer.idx.long()]
    fresh = V.float()[ref_rtn_rung(lat2)] * s
    assert torch.equal(layer.weight(), fresh)                           # cache invalidated: deploy follows the latent
    # null twins: a stale deploy (cache not refreshed) and a hysteretic deploy both disagree with RTN(latent)
    assert float((w_before != fresh).float().mean()) > 0.10
    hyst = V.float()[_hysteretic_rung(lat2.double(), rung_before)] * s
    assert float((hyst != fresh).float().mean()) > 0.002


def test_init_rung_equals_mxnlinear_init_and_fine_init_agrees_up_to_double_rounding():
    na = PackedNA(E2M1_MAGS, C=8, scale_format="nvfp4")
    torch.manual_seed(5)
    mx = MXNLinear(64, 32, na, lr=0.01)
    torch.manual_seed(5)
    lg = LatentGridLinear(64, 32, k=8, init="rung")
    assert torch.equal(lg._scale, mx._scale) and torch.equal(lg._tscale, mx._tscale)   # same NVFP4 fit
    assert torch.equal(lg.weight(), mx.weight())                                       # same deployed init
    torch.manual_seed(5)
    fine = LatentGridLinear(64, 32, k=8, init="fine")
    mismatch = float((fine.weight() != mx.weight()).float().mean())
    assert 0.0 < mismatch < 1.5 / 18                                                   # only at rung midpoints


# ── update law ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [6, 8])
@pytest.mark.parametrize("start,delta", [(2.0, 0.7), (1.5, 1.2), (0.0, 0.3), (-1.0, -2.2), (3.0, -1.7),
                                         (0.5, 4.3), (-4.0, 5.0), (-2.0, -0.7), (1.0, -0.45)])
def test_expected_latent_step_equals_the_value_step_exact(k, start, delta):
    g = get_grid(k)
    mean, se = mc_shift(g, start, delta, "exact")
    assert abs(mean - delta) < 5 * se + 1e-4, (mean, delta, se)


def test_null_twins_dir_and_min_gap_ticks_are_biased():
    g = get_grid(8)
    # inside one gap the directional sub-step is exact too ...
    mean, se = mc_shift(g, 2.0, 0.7, "dir")
    assert abs(mean - 0.7) < 5 * se + 1e-4
    # ... but across gap boundaries it is not (0.5-gap sub-step applied to the 1.0-gap above 2.0 etc.)
    for start, delta in [(1.5, 1.2), (-1.0, -2.2), (3.0, -1.7)]:
        mean, se = mc_shift(g, start, delta, "dir")
        assert abs(mean - delta) > 0.25 and abs(mean - delta) > 50 * se
    # legacy min-gap tick: moving out of a 0.5-gap into a 1.0-gap overshoots by x2 (outward drift), both signs
    for start, delta in [(2.0, 0.7), (-2.0, -0.7), (4.0, 0.9), (-4.0, -0.9)]:
        mean, se = mc_shift(g, start, delta, "min")
        assert abs(mean - delta) > 0.3 and abs(mean - delta) > 50 * se


def _layer_step_error(update_mode, seed=0):
    """Realised minus expected latent step (E2M1 units) of apply_update_, over weights that cannot clip."""
    torch.manual_seed(seed)
    layer = LatentGridLinear(128, 128, k=8, lr=0.01, update_mode=update_mode)
    layer.generator = torch.Generator().manual_seed(seed + 100)
    grad = torch.randn(128, 128, generator=torch.Generator().manual_seed(seed + 7)) * torch.rand(128, 1) * 3 + 0.05
    _, g_hat = mx_update.rms_update(grad, layer.v_rms.clone(), layer.rms_beta, layer.rms_eps)
    s = layer.fmt.na.expand_scale(layer._scale, layer._tscale)
    T = layer.fmt.tables("cpu")
    w0 = T.values[layer.idx.long()]
    want = -layer.lr * g_hat / s
    ok = (w0 + want).abs() < 5.9
    layer.apply_update_(grad)
    got = T.values[layer.idx.long()] - w0
    err = (got - want)[ok].double()
    outward = err * torch.sign(w0[ok]).double()
    n = err.numel()
    return (float(err.mean() / (err.std() / math.sqrt(n))), float(outward.mean() / (outward.std() / math.sqrt(n))),
            float(want[ok].abs().mean()))


def test_apply_update_has_unbiased_expected_step_through_the_layer_and_legacy_does_not():
    z_all, z_out, mag = _layer_step_error("exact")
    assert mag > 0.2                                                    # the test moves weights across several gaps
    assert abs(z_all) < 5 and abs(z_out) < 5
    z_all, z_out, _ = _layer_step_error("min")                          # null twin: legacy min-gap tick
    assert z_out > 20
    z_all, z_out, _ = _layer_step_error("dir")                          # null twin: no boundary handling
    assert abs(z_all) > 8 or abs(z_out) > 8


def test_index_never_leaves_the_grid_and_dtype_is_kept():
    for k in (6, 7, 8):
        g = get_grid(k)
        gen = torch.Generator().manual_seed(0)
        idx = torch.randint(0, g.n_pts, (50_000,), generator=gen)
        delta = torch.randn(50_000, generator=gen) * 8
        delta[:6] = torch.tensor([float("inf"), float("-inf"), float("nan"), 1e30, -1e30, 0.0])
        for mode in ("exact", "dir", "min"):
            new, over = g.move(idx, delta, gen, mode, return_clip=True)
            assert int(new.min()) >= 0 and int(new.max()) <= g.n_pts - 1
            assert bool(over.any())                                     # null twin: the unclamped move DOES leave
            assert int(new[2]) == int(idx[2]) and int(new[5]) == int(idx[5])   # NaN / 0 do not move
        assert int(g.move(idx[:1] * 0 + g.n_pts - 1, torch.tensor([5.0]), gen)) == g.n_pts - 1    # clamp, no wrap
        assert int(g.move(idx[:1] * 0, torch.tensor([-5.0]), gen)) == 0
    layer = LatentGridLinear(64, 32, k=7, lr=5.0)
    for scale in (1e-6, 1.0, 1e6):
        layer.apply_update_(torch.randn(32, 64) * scale)
        assert layer.idx.dtype == torch.uint8 and int(layer.idx.max()) <= 126 and bool(torch.isfinite(layer.weight()).all())


def test_zero_lr_leaves_the_state_unchanged_and_nonzero_lr_does_not():
    torch.manual_seed(0)
    layer = LatentGridLinear(64, 32, k=8, lr=0.0)
    snap = {n: b.clone() for n, b in layer.named_buffers() if n != "v_rms"}
    w0 = layer.weight().clone()
    layer.generator = torch.Generator().manual_seed(1)
    layer.apply_update_(torch.randn(32, 64) * 100)
    for n, b in layer.named_buffers():
        if n != "v_rms":
            assert torch.equal(b, snap[n]), n
    assert torch.equal(layer.weight(), w0)
    assert float(layer.v_rms.min()) > layer.rms_eps                    # the RMS EMA (not the weights) did move
    torch.manual_seed(0)
    twin = LatentGridLinear(64, 32, k=8, lr=0.01)
    before = twin.idx.clone()
    twin.apply_update_(torch.randn(32, 64))
    assert float((twin.idx != before).float().mean()) > 0.5


def test_weight_decay_pulls_the_latent_towards_zero_and_zero_wd_is_inert():
    for wd, moves in [(0.0, False), (0.5, True)]:
        torch.manual_seed(0)
        layer = LatentGridLinear(64, 32, k=8, lr=0.05, weight_decay=wd)
        layer.set_index_(torch.full((32, 64), 11 * 18))                  # all at +2.0
        layer.generator = torch.Generator().manual_seed(2)
        for _ in range(5):
            layer.apply_update_(torch.zeros(32, 64))                    # zero gradient: only the decay acts
        mean = float(layer.fmt.tables("cpu").values[layer.idx.long()].mean())
        assert (mean < 1.9) if moves else (mean == 2.0)


# ── bits ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [6, 7, 8])
def test_bits_accounting_and_bit_packing(k):
    out_f, in_f = 32, 64
    p = out_f * in_f
    layer = LatentGridLinear(in_f, out_f, k=k)
    want = k + 0.5 + 32.0 / in_f + 32.0 / p                          # idx + E4M3/16 + fp32 RMS per row + fp32 tscale
    assert abs(layer.bits_per_param() - want) < 1e-9
    assert abs(layer.fmt.bits_per_param(p) - (k + 0.5 + 32.0 / p)) < 1e-9
    packed = layer.pack_state()
    assert packed.dtype == torch.uint8 and packed.shape == (out_f, in_f * k // 8)
    assert layer.persistent_bytes() == packed.numel() + layer._scale.numel() + out_f * 4 + 4
    assert torch.equal(unpack_bits(packed, k, in_f), layer.idx.long())
    other = LatentGridLinear(in_f, out_f, k=k)
    _set_random_index(other, 9)
    layer.load_state(other.pack_state())
    assert torch.equal(layer.idx, other.idx)
    # null twin: an index that does not fit the k bits cannot be packed losslessly
    over = torch.full((1, 16), 2 ** k)
    assert not torch.equal(unpack_bits(pack_bits(over, k), k, 16), over)


def test_latent_grid_has_no_redundant_codes_unlike_packed_na():
    """PackedNA(rung, counter) overlaps neighbouring rungs' counter ranges; the latent grid has none."""
    def na_positions(C):
        Vq = [Fraction(v) for v in E2M1_SIGNED]
        pos = set()
        for r in range(15):
            for a in range(-(C - 1), C):
                gap = (Vq[min(r + 1, 14)] - Vq[r]) if a >= 0 else (Vq[r] - Vq[max(r - 1, 0)])
                if r == 14 and a >= 0:
                    gap = Vq[14] - Vq[13]
                if r == 0 and a < 0:
                    gap = Vq[1] - Vq[0]
                x = Vq[r] + Fraction(a, C) * gap
                if -6 <= x <= 6:
                    pos.add(x)
        return pos
    for C, k in [(8, 8), (4, 7)]:
        na = PackedNA(E2M1_MAGS, C=C)
        assert na.n_codes == 15 * (2 * C - 1) and na.bits == k
        assert len(na_positions(C)) == 15 + 14 * (C - 1) < 0.55 * na.n_codes     # ~half the codes are duplicates
    assert get_grid(7).n_pts > len(na_positions(8)) > len(na_positions(4))        # lg7 (7 bit) > na c8 (8 bit)
    assert get_grid(8).n_pts == 253 == len(set(get_grid(8).values.tolist()))      # all 253 distinct


# ── forward / backward ───────────────────────────────────────────────────────

def test_forward_backward_parity_with_a_dense_layer_on_the_deployed_weights():
    torch.manual_seed(0)
    layer = LatentGridLinear(64, 32, k=7).train()
    captured = {}
    layer.apply_update_ = lambda gw: captured.setdefault("gw", gw.clone())    # keep the weights fixed for the check
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(3, 5, 64, generator=gen)
    c = torch.randn(3, 5, 32, generator=gen)
    xl = x.clone().requires_grad_()
    out = layer(xl)
    (out * c).sum().backward()
    W = layer.weight().detach().clone().requires_grad_()
    xr = x.clone().requires_grad_()
    out_ref = F.linear(xr, W)
    (out_ref * c).sum().backward()
    assert torch.allclose(out, out_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(xl.grad, xr.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(captured["gw"], W.grad, atol=1e-5, rtol=1e-5)
    assert layer._pending is False
    # null twin: a dense layer on the LATENT (fine) weights is a different function
    assert not torch.allclose(out, F.linear(x, layer.latent_weight()), atol=1e-3)


def test_one_forward_per_backward_guard_fires_and_changed_weights_are_caught():
    layer = LatentGridLinear(32, 16, k=8).train()
    x = torch.randn(4, 32)
    y = layer(x)
    with pytest.raises(RuntimeError, match="forwarded twice"):
        layer(x)
    y.sum().backward()                                                   # the first forward is still usable
    layer(x).sum().backward()                                            # ... and the next pair is fine again
    # null twins: no-grad / eval forwards do not arm the guard
    with torch.no_grad():
        layer(x), layer(x)
    layer.eval()
    layer(x), layer(x)
    layer.train()
    assert layer._pending is False
    y = layer(x)
    layer.set_index_(layer.idx)                                          # any in-place write between fwd and bwd
    with pytest.raises(RuntimeError, match="weights changed"):
        y.sum().backward()
    with pytest.raises(ValueError):
        LatentGridLinear(24, 16)                                         # in_features must be a multiple of 16


def test_training_reduces_a_regression_loss_and_zero_lr_does_not():
    def run(lr):
        torch.manual_seed(0)
        teacher = LatentGridLinear(32, 16, k=8).weight().detach()
        layer = LatentGridLinear(32, 16, k=8, lr=lr).train()
        layer.generator = torch.Generator().manual_seed(5)
        gen = torch.Generator().manual_seed(6)
        losses = []
        for _ in range(150):
            x = torch.randn(64, 32, generator=gen)
            loss = F.mse_loss(layer(x), x @ teacher.t())
            loss.backward()
            losses.append(float(loss.detach()))
        return sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    first, last = run(0.01)
    assert last < 0.4 * first
    first0, last0 = run(0.0)
    assert abs(last0 - first0) < 0.2 * first0 and last0 > 0.8 * first0


# ── e1_quality wrapper (probes/lg_ab.py) ─────────────────────────────────────

def _load_wrapper():
    spec = importlib.util.spec_from_file_location("lg_ab", ROOT / "probes" / "lg_ab.py")
    lg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lg)
    e1 = lg.load_e1(register=False)
    return lg, e1


def test_arm_name_parsing():
    lg, _ = _load_wrapper()
    assert lg.parse_lg("lg8_fp4") == (8, None, False)
    assert lg.parse_lg("lg7_fp4_h1.25") == (7, 1.25, False)
    assert lg.parse_lg("lg6_fp4_h2_rung") == (6, 2.0, True)
    assert lg.parse_lg("na_fp4_c8") is None and lg.parse_lg("qat_fp4") is None
    for bad in ("lg8", "lg8_fp4_x", "lg_fp4", "lg8_fp4_h"):
        with pytest.raises(ValueError):
            lg.parse_lg(bad)


def test_wrapper_registers_the_arms_without_touching_e1_and_delegates_the_rest():
    lg, e1 = _load_wrapper()
    orig = e1.make_linear
    lg.install(e1)
    assert e1.make_linear is not orig and lg.install(e1) is e1 and e1.LADDERS["fp4"][-1] == 6.0
    e1.SCALE_FORMAT = "nvfp4"
    VOCAB, D = 24, 64
    model = e1.CharLM("lg8_fp4_h1.5", VOCAB, d=D, blocks=1)
    lins = [model.blocks[0].qkv, model.blocks[0].proj, model.blocks[0].fc, model.blocks[0].out]
    assert all(isinstance(m, LatentGridLinear) and m.headroom == 1.5 and m.fmt.k == 8 for m in lins)
    assert isinstance(e1.make_linear("lg7_fp4", D, D, 0.01), LatentGridLinear)       # delegation: the rest is unchanged
    assert isinstance(e1.make_linear("na_fp4_c4", D, D, 0.01), MXNLinear)
    assert isinstance(e1.make_linear("dense_nobias", D, D, 0.01), torch.nn.Linear)
    # bits: measured == expected == k + 0.5 + RMS + tscale, both through the harness' own functions
    for k in (6, 7, 8):
        mdl = e1.CharLM(f"lg{k}_fp4", VOCAB, d=D, blocks=1)
        bits = e1.measured_bits(mdl, f"lg{k}_fp4")
        assert abs(bits - e1.expected_bits(mdl, f"lg{k}_fp4")) < 1e-6
        assert k + 0.5 < bits < k + 0.5 + 32.0 / D + 0.02
    # the lg arms are NVFP4-only and have a frozen block scale: anything else must fail loudly, not silently differ
    e1.SCALE_FORMAT = "e8m0"
    with pytest.raises(ValueError, match="nvfp4"):
        e1.make_linear("lg8_fp4", D, D, 0.01)
    e1.SCALE_FORMAT, e1.SCALE_LR = "nvfp4", 0.01
    with pytest.raises(ValueError, match="scale-lr"):
        e1.make_linear("lg8_fp4", D, D, 0.01)
    e1.SCALE_LR = 0.0
    e1.HEADROOM_MODE, e1.HEADROOM = "ladder", 1.2                       # --headroom-mode ladder --headroom 1.2
    assert e1.make_linear("lg8_fp4", D, D, 0.01).headroom == 1.2
    assert e1.make_linear("lg8_fp4_h2", D, D, 0.01).headroom == 2.0     # the arm suffix wins


def test_wrapper_telemetry_flip_edge_hist_and_a_short_run():
    lg, e1 = _load_wrapper()
    lg.install(e1)
    e1.SCALE_FORMAT = "nvfp4"
    VOCAB, D = 24, 64
    model = e1.CharLM("lg7_fp4", VOCAB, d=D, blocks=1)
    snap = e1.snapshot_values(model)
    assert len(snap) == 4 and all(nv == 15 and a.dtype == torch.int8 for a, nv in snap)
    assert e1.flip_edge(snap, snap)[0] == 0.0                                   # null twin: no change, no flips
    vals, fracs = e1.magnitude_histogram(model)
    assert vals == E2M1_SIGNED and abs(sum(fracs) - 1.0) < 1e-5 and len(fracs) == 15
    ws = e1.weight_stats(model)
    assert all(math.isfinite(v) for v in ws.values())
    ids = torch.randint(0, VOCAB, (4000,), generator=torch.Generator().manual_seed(0))
    r = e1.run_arm("lg7_fp4", 0, D, 4, 4, 32, ids, VOCAB, lr=0.01, adam_lr=3e-3, eval_every=2)
    assert math.isfinite(r["val"]) and math.isfinite(r["curve"][-1][3]) and r["curve"][-1][3] > 0.0   # flips happened
    assert math.isfinite(r["edge"]) and abs(sum(r["hist"][1]) - 1.0) < 1e-5
    twin = e1.run_arm("lg7_fp4", 0, D, 4, 4, 32, ids, VOCAB, lr=0.0, adam_lr=3e-3, eval_every=2)
    assert twin["curve"][-1][3] == 0.0                                          # null twin: lr = 0 -> zero flips
