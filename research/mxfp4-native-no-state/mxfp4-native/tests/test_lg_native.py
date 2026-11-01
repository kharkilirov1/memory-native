"""LatentGridLinear native low-precision path (`_LGNativeFunction`): CPU tests force it with native=True on the torch
backend (same autograd code, torch decode); GPU tests (skipped without CUDA + Triton) run the real triton path."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))

from mx6 import fp4_gemm as fg                                         # noqa: E402
from mx6 import lg_update_kernel as K                                  # noqa: E402
from mx6.latent_grid import LatentGridLinear                           # noqa: E402

GPU = pytest.mark.skipif(not (torch.cuda.is_available() and K.HAS_TRITON), reason="needs CUDA + Triton")


def _bf16_cpu_ok() -> bool:
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            torch.randn(4, 4) @ torch.randn(4, 4)
        return True
    except Exception:
        return False


needs_bf16 = pytest.mark.skipif(not _bf16_cpu_ok(), reason="bf16 autocast matmul unavailable on CPU in this torch")


def mk(rows=32, cols=64, k=8, seed=0, device="cpu", **kw):
    torch.manual_seed(seed)
    return LatentGridLinear(cols, rows, k=k, **kw).to(device)


def count_updates(lay):
    calls = []
    orig = lay.apply_update_
    lay.apply_update_ = lambda g: (calls.append((g.dtype, tuple(g.shape))), orig(g))[1]
    return calls


# ── CPU: nibble packing ──────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
def test_pack_nibbles_cf_is_lossless_and_matches_old_pack(k):
    lay = mk(16, 64, k)
    nib = K.pack_nibbles_cf(lay.idx, lay.fmt.m)
    old = fg._pack_values(lay.weight().detach(), lay._scale, lay._tscale)          # the previous route via the fp32 weight
    assert torch.equal(nib, old.nibbles)
    q = lay.fp4_weight()
    assert torch.equal(q.dequantize(), lay.weight())                                 # lossless: bit-equal deployed weight
    assert torch.equal(fg.pack_fp4(lay).nibbles, nib)


# ── CPU: native path on the torch backend ────────────────────────────────────────────────────────────────────

def test_native_fp32_matches_legacy_exactly_and_updates_once():
    a = mk(native=True)
    b = mk(native=False)
    b.load_state_dict(a.state_dict())
    ca = count_updates(a)
    x = torch.randn(10, 64)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    dy = torch.randn(10, 32)
    a.generator, b.generator = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    ya, yb = a(xa), b(xb)
    assert torch.equal(ya, yb)
    ya.backward(dy)
    yb.backward(dy)
    assert torch.equal(xa.grad, xb.grad) and torch.equal(a.idx, b.idx)               # same grads, same noise
    assert len(ca) == 1 and ca[0] == (torch.float32, (32, 64))
    assert not a._pending


def test_native_double_forward_guard_and_save_weight_equivalence():
    lay = mk(rows=16, cols=32)
    lay.native = True
    x = torch.randn(3, 32, requires_grad=True)
    lay(x)
    with pytest.raises(RuntimeError):
        lay(x)
    outs = []
    for sw in (False, True):
        l2 = mk(rows=16, cols=32, native=True, save_weight=sw)
        l2.generator = torch.Generator().manual_seed(1)
        xx = torch.randn(3, 32, generator=torch.Generator().manual_seed(2), requires_grad=True)
        l2(xx).square().sum().backward()
        outs.append((l2.idx.clone(), xx.grad.clone(), l2.v_rms.clone()))
    for u, v in zip(*outs):
        assert torch.equal(u, v)


@needs_bf16
def test_native_under_cpu_autocast_bf16_close_to_fp32_reference():
    lay = mk(64, 128, lr=0.01, native=True)
    ref = mk(64, 128, lr=0.01, native=False)
    ref.load_state_dict(lay.state_dict())
    calls = count_updates(lay)
    x = torch.randn(24, 128)
    xn, xr = x.clone().requires_grad_(), x.clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = lay(xn)
    assert y.dtype == torch.bfloat16                                    # no fp32 island needed
    yr = ref(xr)
    scale = yr.abs().max()
    assert (y.float() - yr).abs().max() / scale < 2e-2
    dy = torch.randn(24, 64)
    y.backward(dy.to(torch.bfloat16))
    yr.backward(dy)
    assert xn.grad.dtype == torch.float32 and len(calls) == 1 and calls[0][0] == torch.bfloat16
    assert (xn.grad - xr.grad).abs().max() / xr.grad.abs().max() < 3e-2
    assert not lay._pending


def test_fp4_gemm_modes_fall_back_on_cpu_and_are_counted():
    base = mk(16, 32, native=True)
    for g in ("triton_bf16", "nvfp4_scaled_mm"):
        lay = mk(16, 32, native=True, gemm=g, dx_gemm="nvfp4")
        x = torch.randn(4, 32, requires_grad=True)
        y = lay(x)
        with torch.no_grad():
            assert torch.equal(y, base(x))                               # fell back to the exact path
        y.sum().backward()
        assert lay.fp4_fallbacks >= 1
    with pytest.raises(ValueError):
        mk(gemm="nope")
    with pytest.raises(ValueError):
        mk(native="maybe")


def test_native_auto_is_off_on_cpu_so_legacy_islands_stay():
    assert not mk().native_enabled()
    assert mk(native=True).native_enabled()


# ── CPU: bench / e1 integration ──────────────────────────────────────────────────────────────────────────────

def _bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_train_step_native", ROOT / "probes" / "bench_train_step.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _e1():
    import lg_ab
    e1 = lg_ab.install(lg_ab.load_e1(register=False))
    e1.DEV, e1.SCALE_FORMAT, e1.HEADROOM_MODE, e1.HEADROOM = "cpu", "nvfp4", "ladder", 2.0
    return e1


def _ns(**kw):
    base = dict(lr=0.01, tern_lr=3e-2, adam_lr=3e-3, headroom=2.0, layers=1, heads=0, batch=2, seq=16, warmup=1,
                steps=2, phase_steps=0, no_profile=True, topk=3, fused="off", seed=0)
    base.update(kw)
    return SimpleNamespace(**base)


@needs_bf16
def test_fp32_islands_skips_native_lg_layers_and_bench_runs():
    bench, e1 = _bench(), _e1()
    r = bench.bench_one("lg6_fp4_h2", 32, "eager", _ns(precision="bf16", adam_impl="fused", lg_native="on"), e1, 64, "cpu")
    assert r["error"] is None, r.get("traceback")
    assert r["fp32_islands"] == 0 and r["native_lg"] == r["lg_layers"] == 4 and r["finite"]
    legacy = bench.bench_one("lg6_fp4_h2", 32, "eager", _ns(precision="bf16", adam_impl="fused", lg_native="off"), e1, 64, "cpu")
    assert legacy["error"] is None and legacy["fp32_islands"] == 4 and legacy["native_lg"] == 0
    f = bench.bench_one("lg6_fp4_h2_nvfp4fwd", 32, "eager",
                        _ns(precision="bf16", adam_impl="fused", lg_native="on"), e1, 64, "cpu")
    assert f["error"] is None and f["lg_gemm"] == "nvfp4_scaled_mm"               # CPU: falls back to the exact path


def test_tern_counter_compile_is_reported_unsupported_not_failed():
    bench, e1 = _bench(), _e1()
    r = bench.bench_one("tern_counter", 32, "compile", _ns(), e1, 64, "cpu")
    assert r["error"].startswith("UNSUPPORTED") and r["unsupported"] is True


@needs_bf16
def test_e1_run_arm_bf16_with_native_lg():
    e1 = _e1()
    orig = e1.make_linear

    def make(arm, i, o, lr):
        lay = orig(arm, i, o, lr)
        if isinstance(lay, LatentGridLinear):
            lay.native = True
        return lay
    e1.make_linear = make
    ids = torch.randint(0, 24, (4000,))
    r = e1.run_arm(arm="lg6_fp4_h2", seed=0, d=32, steps=6, batch=4, seq=16, ids=ids, vocab=24, lr=0.01, adam_lr=3e-3,
                   eval_every=3, blocks=1, heads=2, precision="bf16")
    assert r["val"] == r["val"]


# ── GPU ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _pair(rows, cols, k=8, **kw):
    a = mk(rows, cols, k, backend="triton", device="cuda", **kw)
    b = mk(rows, cols, k, backend="torch", device="cuda", native=False, **kw)
    b.load_state_dict(a.state_dict())
    return a, b


@GPU
def test_gpu_native_is_active_and_matches_reference_in_bf16():
    a, ref = _pair(512, 1024, lr=0.01)
    assert a.native_enabled()
    calls = count_updates(a)
    x = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    y = a(x)
    yr = ref(xr)
    assert y.dtype == torch.bfloat16
    assert ((y.float() - yr).abs().max() / yr.abs().max()) < 1.5e-2
    dy = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    idx0 = a.idx.clone()
    y.backward(dy)
    yr.backward(dy.float())
    assert ((x.grad.float() - xr.grad).abs().max() / xr.grad.abs().max()) < 2e-2
    assert len(calls) == 1 and calls[0][0] == torch.bfloat16 and calls[0][1] == (512, 1024)      # exactly one bf16 update
    assert not torch.equal(a.idx, idx0) and not a._pending
    # the bf16 grad handed to the kernel equals dy^T x within bf16 rounding
    gw = (dy.float().t() @ x.detach().float())
    g_native = dy.t() @ x.detach()
    assert ((g_native.float() - gw).abs().max() / gw.abs().max()) < 1e-2


@GPU
def test_gpu_autocast_fp32_input_and_no_island():
    a, _ = _pair(128, 256)
    x = torch.randn(64, 256, device="cuda", requires_grad=True)             # fp32 activations (LayerNorm output under autocast)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = a(x)
    assert y.dtype == torch.bfloat16
    y.float().square().mean().backward()
    assert x.grad.dtype == torch.float32 and torch.isfinite(x.grad).all()
    # eval / no_grad under autocast, and plain fp32 (no autocast) still work
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert a(x).dtype == torch.bfloat16
    a2, _ = _pair(128, 256)
    y32 = a2(torch.randn(8, 256, device="cuda"))
    assert y32.dtype == torch.float32
    y32.sum().backward()


@GPU
def test_gpu_no_fp32_copies_resident_or_in_flight():
    rows = cols = 2048
    n = rows * cols
    a = mk(rows, cols, 8, backend="triton", device="cuda", lr=0.01)
    assert a.resident_bytes() < n * 1.1
    x = torch.randn(512, cols, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(512, rows, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):                                                        # warm cuBLAS / Triton workspaces
        a(x).backward(dy)
    x.grad = None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = a(x)
    fwd_held = torch.cuda.memory_allocated() - base
    y.backward(dy)
    x.grad = None
    del y
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated() - base
    peak = torch.cuda.max_memory_allocated() - base
    assert a._weight_cache is None and a._scale_cache is None and a.resident_bytes() < n * 1.1
    assert after < 1 << 20, after                                             # nothing left behind between steps
    # forward holds only x (bf16, already counted) + y; the decoded weight (2 B/param) is freed after the GEMM
    assert fwd_held < 4 * (1 << 20), fwd_held
    # peak: re-decoded bf16 W (2) + bf16 grad (2) + small activations; an fp32 weight or grad copy (4 B each) would bust it
    assert peak < 4.6 * n + (16 << 20), peak / n
    # save_weight=True keeps the bf16 weight (2 B/param) between forward and backward
    a.save_weight = True
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    y = a(x)
    assert torch.cuda.memory_allocated() - base >= 1.9 * n
    y.backward(dy)


@GPU
def test_gpu_save_weight_gives_identical_update():
    a, _ = _pair(256, 512, lr=0.02)
    b, _ = _pair(256, 512, lr=0.02)
    b.load_state_dict(a.state_dict())
    b.save_weight = True
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    outs = []
    for lay in (a, b):
        torch.manual_seed(3)
        xx = x.detach().clone().requires_grad_()
        lay(xx).backward(dy)
        outs.append((lay.idx.clone(), xx.grad.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


@GPU
def test_gpu_pack_kernel_matches_closed_form_and_dequantizes_exactly():
    for k in (4, 6, 8):
        lay = mk(64, 256, k, backend="triton", device="cuda")
        nib = K.lg_pack_nibbles(lay.idx, lay.fmt.m)
        assert torch.equal(nib, K.pack_nibbles_cf(lay.idx, lay.fmt.m))
        torch.testing.assert_close(lay.fp4_weight().dequantize(torch.float32), lay.weight(torch.float32),
                                   rtol=1e-6, atol=0)


@GPU
def test_gpu_triton_bf16_gemm_is_exact_weights():
    a, _ = _pair(256, 512)
    c = mk(256, 512, backend="triton", device="cuda", gemm="triton_bf16")
    c.load_state_dict(a.state_dict())
    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xc = x.detach().clone().requires_grad_()
    ya, yc = a(x), c(xc)
    assert ((ya.float() - yc.float()).abs().max() / ya.float().abs().max()) < 1e-2
    dy = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    ya.backward(dy)
    yc.backward(dy)
    assert ((x.grad.float() - xc.grad.float()).abs().max() / x.grad.float().abs().max()) < 1e-2


@GPU
def test_gpu_nvfp4_scaled_mm_forward_is_approximate_but_close():
    ok, why = fg.backend_status("torch_scaled_mm")
    if not ok:
        pytest.skip(why)
    a, _ = _pair(512, 1024)
    c = mk(512, 1024, backend="triton", device="cuda", gemm="nvfp4_scaled_mm")
    c.load_state_dict(a.state_dict())
    x = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ya, yc = a(x.detach()), c(x)
    rel = ((ya.float() - yc.float()).abs().mean() / ya.float().abs().mean()).item()
    assert 1e-3 < rel < 0.30, rel                    # activation NVFP4 quantization: ~9-14%, never exact
    assert c.fp4_fallbacks == 0
    yc.backward(torch.randn_like(yc))                # dx exact bf16 by default, update applied
    assert not c._pending
