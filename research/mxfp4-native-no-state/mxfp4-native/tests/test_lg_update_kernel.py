"""mx6/lg_update_kernel.py: CPU part = torch transcription of the kernel arithmetic (closed-form G/U/deploy, same order
of ops) vs the reference torch path; GPU part (skipped without CUDA+Triton) = the Triton kernels vs the reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mx6 import lg_update_kernel as K                                  # noqa: E402
from mx6.latent_grid import LatentGridLinear, get_grid                 # noqa: E402

GPU = pytest.mark.skipif(not (torch.cuda.is_available() and K.HAS_TRITON), reason="needs CUDA + Triton")
KM = {8: 18, 7: 9, 6: 4, 5: 2, 4: 1}


def make_layer(rows=8, cols=64, k=8, seed=0, backend="torch", device="cpu", **kw):
    torch.manual_seed(seed)
    lay = LatentGridLinear(cols, rows, k=k, backend=backend, **kw).to(device)
    return lay


# ── CPU: closed forms ────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
def test_closed_form_grid_and_deploy_match_tables(k):
    g = get_grid(k)
    i = torch.arange(g.n_pts)
    assert torch.allclose(K.grid_value_cf(i, g.m), g.values, rtol=0, atol=2e-6)
    assert torch.equal(K.deploy_cf(i, g.m), g.deploy)                 # RTN to E2M1, ties away from zero, exact
    # U inverts G, and equals the table-based U (unclamped) on arbitrary values incl. beyond the grid
    assert torch.allclose(K.u_map_cf(g.values, g.m), i.float(), atol=2e-4)
    w = torch.linspace(-20, 20, 4001)
    ref = g._u(w, g.tables("cpu"), clamp=False)
    assert torch.allclose(K.u_map_cf(w, g.m), ref, rtol=1e-6, atol=1e-3)


def test_e4m3_scale_formula_matches_decode():
    b = torch.arange(1, 127, dtype=torch.uint8)
    e = (b.int() >> 3)
    pow2 = torch.pow(2.0, e.float() - 7)
    mm = (b.int() & 7).float()
    val = torch.where(e == 0, mm * 2 ** -9, (8 + mm) * 0.125 * pow2)
    assert torch.equal(val, K.e4m3_decode(b))


# ── CPU: emulation vs the reference torch path ──────────────────────────────────────────────────────────────

def _ref_and_emu(k=8, wd=0.0, lr=0.01, seed=3, rows=8, cols=64, gscale=1e-2):
    lay = make_layer(rows, cols, k, seed, lr=lr, weight_decay=wd)
    torch.manual_seed(seed + 100)
    g = torch.randn(rows, cols) * gscale
    idx0, v0 = lay.idx.clone(), lay.v_rms.clone()
    gen = torch.Generator().manual_seed(77)
    lay.generator = gen
    u = torch.rand(rows, cols, generator=torch.Generator().manual_seed(77))     # the very draw `move` will make
    new, v_new, over = K.emulate_update(g, idx0, lay._scale, lay._tscale, v0, u, lr=lr, beta=lay.rms_beta,
                                        eps=lay.rms_eps, weight_decay=wd, m=lay.fmt.m)
    lay.apply_update_(g)
    return lay, idx0, new, v_new, over


@pytest.mark.parametrize("k", [8, 6, 4])
@pytest.mark.parametrize("wd", [0.0, 0.05])
def test_emulation_matches_reference_given_identical_uniforms(k, wd):
    lay, idx0, new, v_new, over = _ref_and_emu(k=k, wd=wd)
    assert torch.allclose(v_new, lay.v_rms, rtol=1e-6)
    n_diff = (new != lay.idx.long()).sum().item()
    # closed-form vs table arithmetic differs by ulps; with the same uniform a different integer needs
    # |u - uniform| < ~1e-5, so mismatches must be (almost) absent and never off by more than one
    assert n_diff <= 2, n_diff
    assert (new - lay.idx.long()).abs().max() <= 1
    assert (new != idx0.long()).float().mean() > 0.05          # the update actually moves things (non-vacuous)


def test_emulation_lr0_is_noop_and_range_holds():
    lay = make_layer(rows=16, cols=64, lr=0.0)
    g = torch.randn(16, 64)
    u = torch.rand(16, 64)
    new, _, _ = K.emulate_update(g, lay.idx, lay._scale, lay._tscale, lay.v_rms, u, lr=0.0, beta=0.9, eps=1e-3,
                                 weight_decay=0.0, m=lay.fmt.m)
    assert torch.equal(new, lay.idx.long())
    big = torch.randn(16, 64) * 1e3                              # huge steps: clamp at the grid ends, no wrap
    new, _, over = K.emulate_update(big, lay.idx, lay._scale, lay._tscale, lay.v_rms, u, lr=5.0, beta=0.9, eps=1e-3,
                                    weight_decay=0.0, m=lay.fmt.m)
    assert int(new.min()) >= 0 and int(new.max()) <= 14 * lay.fmt.m and bool(over.any())


@pytest.mark.parametrize("k", [8, 6])
def test_emulation_expected_latent_step_is_minus_lr_ghat(k):
    """MC unbiasedness of the kernel arithmetic: E[G[idx'] - G[idx]] * s = -lr * g_hat on unclipped weights, z < 5."""
    rows, cols, n = 4, 32, 3000
    lay = make_layer(rows, cols, k, seed=5, lr=0.02)
    g = torch.randn(rows, cols) * 0.05
    m = lay.fmt.m
    s = K.block_scale_cf(lay._scale, lay._tscale).repeat_interleave(16, dim=1)
    w0 = K.grid_value_cf(lay.idx, m)
    gq = g.square().mean(1, keepdim=True)
    v_new = lay.v_rms * 0.9 + gq * 0.1
    ghat = g / v_new.sqrt().clamp_min(lay.rms_eps)
    target = -lay.lr * ghat / s                                   # in block-scale units
    gen = torch.Generator().manual_seed(11)
    acc = torch.zeros(rows, cols, dtype=torch.float64)
    acc2 = torch.zeros_like(acc)
    over_any = torch.zeros(rows, cols, dtype=torch.bool)
    for _ in range(n):
        u = torch.rand(rows, cols, generator=gen)
        new, _, over = K.emulate_update(g, lay.idx, lay._scale, lay._tscale, lay.v_rms, u, lr=lay.lr, beta=0.9,
                                        eps=lay.rms_eps, weight_decay=0.0, m=m)
        d = (K.grid_value_cf(new, m) - w0).double()
        acc += d
        acc2 += d * d
        over_any |= over
    mean = acc / n
    se = ((acc2 / n - mean ** 2).clamp_min(0) / n).sqrt().clamp_min(1e-9)
    z = ((mean - target.double()) / se).abs()
    keep = ~over_any
    assert keep.float().mean() > 0.75          # block maxima sit on the outer rung: outward requests clip (deterministic)
    assert float(z[keep].max()) < 5.0, float(z[keep].max())
    # null twin: ignoring the multi-gap structure (a fixed min-gap tick) would NOT pass this; here the targets are
    # non-trivial: the mean shift is far from zero relative to its standard error
    assert float((target.double().abs() / se)[keep].max()) > 5.0


# ── CPU: module-level behaviour without Triton ──────────────────────────────────────────────────────────────

def test_backend_arg_and_cpu_fallback():
    with pytest.raises(ValueError):
        LatentGridLinear(32, 4, backend="nope")
    with pytest.raises(ValueError):
        LatentGridLinear(32, 4, backend="triton", update_mode="dir")
    lay = LatentGridLinear(32, 4, backend="auto")                 # on CPU: torch path
    x = torch.randn(3, 32, requires_grad=True)
    lay(x).sum().backward()
    assert lay.idx.dtype == torch.uint8
    with pytest.raises(RuntimeError):
        LatentGridLinear(32, 4, backend="triton").weight()        # CUDA + Triton required


def test_resident_bytes_torch_path_documents_the_inflation():
    lay = make_layer(rows=16, cols=64, backend="torch")
    lay.weight()
    p = 16 * 64
    assert lay.resident_bytes() >= p * 9                          # idx 1 + fp32 scale cache 4 + fp32 weight 4
    lay.invalidate()
    assert lay.resident_bytes() < p * 1.5                         # state only


def test_double_forward_guard_still_works():
    lay = make_layer(rows=4, cols=32)
    x = torch.randn(2, 32, requires_grad=True)
    lay(x)
    with pytest.raises(RuntimeError):
        lay(x)


# ── GPU ──────────────────────────────────────────────────────────────────────────────────────────────────

def _pair(rows, cols, k=8, **kw):
    a = make_layer(rows, cols, k, backend="triton", device="cuda", **kw)
    b = make_layer(rows, cols, k, backend="torch", device="cuda", **kw)
    b.load_state_dict(a.state_dict())
    return a, b


@GPU
@pytest.mark.parametrize("gdtype", [torch.float32, torch.bfloat16])
def test_gpu_v_rms_parity_and_decode_exact(gdtype):
    a, b = _pair(64, 512, lr=0.01)
    g = (torch.randn(64, 512, device="cuda") * 0.02).to(gdtype)
    assert torch.equal(a.weight(torch.float32), b.weight())                   # decode kernel bit-exact vs table
    a.apply_update_(g)
    b.apply_update_(g)
    assert torch.allclose(a.v_rms, b.v_rms, rtol=1e-5)
    assert int(a.idx.max()) <= 14 * a.fmt.m
    # same distribution of moves: mean |shift| and mean signed latent shift agree within MC noise
    d_a = (a.latent_weight() - make_layer(64, 512, 8, backend="torch", device="cuda").latent_weight()).double()
    d_b = (b.latent_weight() - make_layer(64, 512, 8, backend="torch", device="cuda").latent_weight()).double()
    se = (d_a.std() / (d_a.numel() ** 0.5) * 3).item()
    assert abs(d_a.mean().item() - d_b.mean().item()) < max(se, 1e-6) * 5


@GPU
def test_gpu_deterministic_given_seed_and_lr0_noop():
    a, b = _pair(32, 256, lr=0.02)
    g = torch.randn(32, 256, device="cuda") * 0.05
    torch.manual_seed(9)
    a.apply_update_(g)
    torch.manual_seed(9)
    b.backend = "triton"
    b.apply_update_(g)
    assert torch.equal(a.idx, b.idx)
    c, _ = _pair(32, 256, lr=0.0)
    idx0 = c.idx.clone()
    c.apply_update_(g)
    assert torch.equal(c.idx, idx0)


@GPU
@pytest.mark.parametrize("k", [8, 6])
def test_gpu_unbiased_expected_step(k):
    """Rows are identical copies (same idx, same grad) => independent Philox noise per row = MC over rows."""
    n, cols = 8192, 64
    lay = make_layer(1, cols, k, backend="triton", device="cuda", lr=0.02)
    idx = lay.idx.repeat(n, 1)
    sb = lay._scale.repeat(n, 1)
    g1 = torch.randn(1, cols, device="cuda") * 0.05
    g = g1.repeat(n, 1)
    v = torch.full((n, 1), lay.rms_eps, device="cuda")
    m = lay.fmt.m
    vals = lay.fmt.tables("cuda").values
    w0 = vals[lay.idx.long()]
    s = lay.fmt.na.expand_scale(lay._scale, lay._tscale)
    v_new = v[:1] * 0.9 + g1.square().mean(1, keepdim=True) * 0.1
    target = (-lay.lr * g1 / v_new.sqrt().clamp_min(lay.rms_eps) / s).double()
    idx0, v_old = lay.idx.clone(), v[:1].clone()           # the kernel updates v in place
    clip = K.lg_update_(g, idx, sb, lay._tscale, v, lr=lay.lr, beta=0.9, eps=lay.rms_eps, weight_decay=0.0, m=m,
                        seed=123)
    d = (vals[idx.long()] - w0).double()
    std = d.std(0)
    mean, se = d.mean(0), (std / n ** 0.5).clamp_min(1e-9)
    z = ((mean - target[0]) / se).abs()

    # Which columns may legitimately differ from the unclipped target? Those whose request leaves the grid: the
    # excess is dropped BEFORE the rounding (u is clamped to [0, top]) => deterministic, zero variance, biased by design.
    # (The old mask `0 < idx < top` missed e.g. idx = 12m asking for +3 gap units: z ~ 1e9 with se clamped at 1e-9.)
    # `over` comes from the CPU-validated torch transcription of the same arithmetic, not from the kernel under test.
    _, _, over = K.emulate_update(g1, lay.idx, lay._scale, lay._tscale, v_old, torch.zeros_like(g1), lr=lay.lr,
                                  beta=0.9, eps=lay.rms_eps, weight_decay=0.0, m=m)
    keep = ~over[0]
    u0 = lay.idx[0].float() + (K.u_map_cf(w0[0] + target[0].float(), m) - K.u_map_cf(w0[0], m))
    frac = u0 - torch.floor(u0)
    pmin = torch.minimum(frac, 1 - frac)
    stochastic = keep & (n * pmin >= 50)                      # enough expected "up" events for a z-test / non-zero spread
    determ = keep & (pmin < 1e-6)                            # on-grid target: no randomness, must match to fp noise
    testable = stochastic | determ

    def report(mask):
        cols = torch.nonzero(mask).flatten().tolist()
        return "; ".join(f"col {c}: idx {int(idx0[0, c])} target {float(target[0, c]):+.4f} mean {float(mean[c]):+.4f} "
                         f"std {float(std[c]):.4f} over {bool(over[0, c])} z {float(z[c]):.3g}" for c in cols[:12])

    bad = (stochastic & (z >= 5.5)) | (determ & ((mean - target[0]).abs() > 1e-3))
    assert not bool(bad.any()), "biased columns (unclipped): " + report(bad)
    assert int(testable.sum()) >= cols // 4, "too few testable columns: " + str(int(testable.sum()))
    # RNG must vary across rows: every column whose rounding is genuinely random has nonzero spread
    frozen = stochastic & (std == 0)
    assert not bool(frozen.any()), "zero variance across rows although the rounding is random: " + report(frozen)
    # clipped columns: deterministic at the end point (the dropped excess), never beyond it
    assert int(idx.max()) <= 14 * m and int(idx.min()) >= 0
    assert clip.shape == (n,)
    assert int(keep.sum()) >= cols // 4, "mask excludes too many columns to be meaningful"


@GPU
def test_gpu_rng_rows_are_decorrelated():
    """Philox counter = row*cols+col: neighbouring rows / columns must not repeat each other's random draws."""
    n, cols = 4096, 64
    lay = make_layer(1, cols, 8, backend="triton", device="cuda", lr=0.02)
    idx = lay.idx.repeat(n, 1)
    sb = lay._scale.repeat(n, 1)
    g = (torch.randn(1, cols, device="cuda") * 0.05).repeat(n, 1)
    v = torch.full((n, 1), lay.rms_eps, device="cuda")
    K.lg_update_(g, idx, sb, lay._tscale, v, lr=lay.lr, beta=0.9, eps=lay.rms_eps, weight_decay=0.0, m=lay.fmt.m, seed=7,
                 track_clip=False)
    d = (idx.float() - lay.idx.float())
    mov = d.std(0) > 0
    assert int(mov.sum()) > cols // 4
    d = d - d.mean(0)                                           # remove the per-column mean (different targets)
    c_rows = torch.corrcoef(torch.stack([d[:-1, mov].flatten(), d[1:, mov].flatten()]))[0, 1].abs()
    assert float(c_rows) < 0.05, float(c_rows)                  # adjacent rows independent
    a = d[:, mov]
    cc = torch.corrcoef(a.t())
    off = cc - torch.diag(torch.diag(cc))
    assert float(off.abs().max()) < 0.2, float(off.abs().max())  # distinct columns independent (n=4096 => ~0.016 sd)


@GPU
def test_gpu_end_to_end_training_step_and_memory():
    lay = make_layer(256, 512, 8, backend="triton", device="cuda", lr=0.01)
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = lay(x)
    assert y.dtype == torch.bfloat16
    before = lay.idx.clone()
    y.float().square().mean().backward()
    assert not torch.equal(before, lay.idx)
    assert lay._scale_cache is None and lay._weight_cache is None
    assert lay.resident_bytes() < 256 * 512 * 1.1
    assert lay.last_clip_frac is not None


@GPU
def test_gpu_bench_smoke():
    out = K.bench_update(128, 256, iters=5, compare_torch=True)
    assert out["triton_us"] > 0 and out["torch_us"] > 0
