"""NVFP4 scaling (opt-in): E4M3 scale per 16 + fp32 per-tensor scale.

w ~= ladder[idx] * e4m3(byte) * tensor_scale,   tensor_scale = amax / (L * 448),
byte = RTN_e4m3(block_amax / (L * tensor_scale)),   L = outer rung of the ladder.

Covers: E4M3 rounding against an independent reference table, round-trip error vs E8M0,
refresh symmetry, exact bit accounting, and the directional-gap martingale under a
non-power-of-two scale.  The default (e8m0) behaviour is covered by the other test files.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mx6 import packed_na as P                                  # noqa: E402
from mx6.layer import MXNLinear                                 # noqa: E402
from mx6.packed_na import PackedNA, e4m3_decode, e4m3_encode    # noqa: E402

LADDERS = {
    "wide": ([0.0, 0.5, 3.0], 3),
    "e2m1": ([0.0, 0.5, 1.0, 6.0], 5),
    "full": ([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], 3),   # the real E2M1 value set
}


def nv(name):
    mags, C = LADDERS[name]
    return PackedNA(mags, C, scale_format="nvfp4")


def e8(name):
    mags, C = LADDERS[name]
    return PackedNA(mags, C)


# ── E4M3 ──────────────────────────────────────────────────────────────────────

def _e4m3_table():
    """(bytes, values) of all positive finite E4M3FN codes, built from the definition."""
    vals, bytes_ = [], []
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue                      # NaN
            if e == 0 and m == 0:
                continue                      # zero: encode() floors at 2^-9 instead
            v = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
            vals.append(v)
            bytes_.append((e << 3) | m)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    return [bytes_[i] for i in order], [vals[i] for i in order]


def _ref_rtn(x: float, bytes_, vals) -> int:
    """Nearest table value; an exact tie goes to the even mantissa (even byte)."""
    best = min(range(len(vals)), key=lambda i: (abs(vals[i] - x), bytes_[i] & 1))
    return bytes_[best]


def test_e4m3_decode_matches_torch_fp8_for_every_byte():
    b = torch.tensor([x for x in range(128) if x != 0x7F], dtype=torch.uint8)
    ref = b.view(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(e4m3_decode(b), ref)
    assert float(e4m3_decode(torch.tensor([0x7E], dtype=torch.uint8))) == 448.0
    assert float(e4m3_decode(torch.tensor([1], dtype=torch.uint8))) == 2.0 ** -9


def test_e4m3_rounding_vs_reference_table():
    bytes_, vals = _e4m3_table()
    assert len(vals) == 126 and max(vals) == 448.0
    # every representable value, every adjacent midpoint (exact ties), neighbours of the
    # midpoints, and random points on a log grid
    pts = list(vals)
    for lo, hi in zip(vals[:-1], vals[1:]):
        mid = (lo + hi) / 2
        pts += [mid, math.nextafter(mid, 0.0), math.nextafter(mid, 1e9)]
    g = torch.Generator().manual_seed(0)
    pts += (torch.exp(torch.rand(2000, generator=g) * math.log(448 / 2.0 ** -9)) * 2.0 ** -9).tolist()
    x32 = torch.tensor(pts, dtype=torch.float32)
    got = e4m3_encode(x32).tolist()
    for xi, gi in zip(x32.tolist(), got):
        assert gi == _ref_rtn(xi, bytes_, vals), (xi, gi, _ref_rtn(xi, bytes_, vals))


def test_e4m3_saturates_and_never_returns_zero_or_nan():
    x = torch.tensor([0.0, 1e-12, 2.0 ** -11, 447.0, 448.0, 1e4, 1e30])
    b = e4m3_encode(x)
    assert b[0] == 1 and b[1] == 1 and b[2] == 1       # floored at 2^-9, not 0 (w / 0)
    assert b[-1] == 0x7E and b[-2] == 0x7E and b[-3] == 0x7E   # clipped to 448, not NaN
    assert not torch.isnan(e4m3_decode(b)).any()


# ── round trip vs E8M0 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["wide", "e2m1", "full"])
def test_roundtrip_mse_nvfp4_below_e8m0(name):
    """NVFP4 (block 16, 4-bit-mantissa scale) beats E8M0 (block 32, power-of-two scale).

    Fair baseline: E8M0 with the same ladder-aware target (amax -> L, headroom = L).
    A power-of-two scale can only reach amax/s in (L/2, L]; E4M3 gets within 6 %.
    """
    torch.manual_seed(0)
    w = torch.randn(256, 512)
    a, b = nv(name), e8(name)
    L = a.ladder_max
    codes, sb, ts = a.quantize_scaled(w)
    mse_nv = float((a.dequantize(codes, sb, ts) - w).square().mean())
    codes8, sb8 = b.quantize(w, headroom=L)
    mse_e8 = float((b.dequantize(codes8, sb8) - w).square().mean())
    codes8l, sb8l = b.quantize(w, headroom=1.5)
    mse_legacy = float((b.dequantize(codes8l, sb8l) - w).square().mean())
    print(f"\n[{name}] MSE nvfp4={mse_nv:.5f}  e8m0(amax->L)={mse_e8:.5f}  e8m0(legacy 1.5)={mse_legacy:.5f}"
          f"  ratio nv/e8={mse_nv / mse_e8:.3f}")
    assert mse_nv < 0.95 * mse_e8, (name, mse_nv, mse_e8)


def test_block_amax_lands_on_the_outer_rung():
    torch.manual_seed(1)
    w = torch.randn(64, 256)
    for name in ("wide", "e2m1"):
        f = nv(name)
        codes, sb, ts = f.quantize_scaled(w)
        idx = codes.long() // f.acc_levels
        top = f.n_vals - 1
        blocks = w.reshape(64, -1, 16)
        am = blocks.abs().argmax(-1)
        sel = torch.gather(idx.reshape(64, -1, 16), 2, am[..., None])[..., 0]
        sgn = torch.gather(blocks, 2, am[..., None])[..., 0] > 0
        frac = float(torch.where(sgn, sel == top, sel == 0).float().mean())
        assert frac > 0.85, (name, frac)       # RTN of the scale may fall just under L
        assert int(sb.max()) <= 0x7E


# ── refresh symmetry ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["wide", "e2m1"])
def test_nvfp4_refresh_is_sign_symmetric(name):
    mags, C = LADDERS[name]
    torch.manual_seed(2)
    w = torch.randn(16, 128) * 0.3
    pos = MXNLinear.from_dense(w, nv(name), scale_headroom=1.5)
    neg = MXNLinear.from_dense(-w, nv(name), scale_headroom=1.5)
    assert torch.equal(pos.weight(), -neg.weight())
    # perturb codes identically (mirrored) so refresh has work to do
    with torch.no_grad():
        f = pos.fmt
        idx = pos.codes.long() // f.acc_levels
        mirror = f.n_vals - 1 - idx
        acc0 = torch.zeros_like(idx)
        pos._state.copy_(f.pack_state(f.encode(idx, acc0).to(torch.uint8)))
        neg._state.copy_(f.pack_state(f.encode(mirror, acc0).to(torch.uint8)))
        pos._codes_cache = pos._weight_cache = neg._codes_cache = neg._weight_cache = None
    assert torch.equal(pos.weight(), -neg.weight())
    pos.refresh_scale_()
    neg.refresh_scale_()
    assert torch.equal(pos._scale, neg._scale) and torch.equal(pos._tscale, neg._tscale)
    assert torch.equal(pos.weight(), -neg.weight())


# ── bit accounting ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["wide", "e2m1"])
def test_bit_accounting_is_exact(name):
    a, b = nv(name), e8(name)
    out_f, in_f = 8, 4096
    n = out_f * in_f
    exp_nv = out_f * a.packed_width(in_f) + out_f * (in_f // 16) + out_f * 4 + 4
    assert a.persistent_bytes(out_f, in_f) == exp_nv
    assert b.persistent_bytes(out_f, in_f) == out_f * b.packed_width(in_f) + out_f * (in_f // 32) + out_f * 4
    # codes + 8/16 scale bits + 32/numel tensor scale, exactly
    assert a.bits_per_param(n) == pytest.approx(a.bits + 0.5 + 32.0 / n, abs=1e-12)
    assert b.bits_per_param() == pytest.approx(b.bits + 0.25, abs=1e-12)
    assert a.bits_per_param() == a.bits + 0.5
    # per-param including the per-row RMS, from the byte count
    assert exp_nv * 8.0 / n == pytest.approx(a.bits + 0.5 + 32.0 / in_f + 32.0 / n, abs=1e-12)
    print(f"\n[{name}] bits/param (codes+scales+tensor scale, no RMS): "
          f"e8m0={b.bits_per_param():.4f}  nvfp4={a.bits_per_param(n):.6f}")


@pytest.mark.parametrize("fmt_name", ["e8m0", "nvfp4"])
def test_persistent_bytes_equals_buffers_held(fmt_name):
    f = nv("e2m1") if fmt_name == "nvfp4" else e8("e2m1")
    layer = MXNLinear(256, 32, f)
    held = sum(t.numel() * t.element_size()
               for t in (layer._state, layer._scale, layer.v_rms, layer._tscale) if t is not None)
    assert layer.persistent_bytes() == held
    assert (layer._tscale is None) == (fmt_name == "e8m0")


def test_e8m0_state_dict_unchanged():
    keys = set(MXNLinear(64, 8, e8("wide")).state_dict())
    assert keys == {"_state", "_scale", "v_rms"}


# ── update path uses the actual decoded scale ─────────────────────────────────

def test_step_tick_uses_decoded_nonpow2_scale():
    f = nv("e2m1")
    torch.manual_seed(3)
    out, nb = 4, 8
    sb = torch.randint(40, 70, (out, nb), dtype=torch.uint8)
    ts = torch.tensor(0.7391)
    mag = torch.randint(0, f.n_vals, (out, nb * 16))
    acc = torch.randint(-(f.C - 1), f.C, (out, nb * 16))
    codes = f.encode(mag, acc).to(torch.uint8)
    g = torch.randn(out, nb * 16)
    got = P._step_pre_sr_dir(codes, sb, g, 0.05, f.C, f._gaps_up, f._gaps_down,
                             f.acc_levels, f.n_vals, 16, True, ts)
    s = (e4m3_decode(sb) * ts).repeat_interleave(16, dim=1)
    assert not torch.allclose(s, torch.pow(2.0, torch.round(torch.log2(s))))   # really not 2^k
    gu, gd = f._gaps_up[mag] * s, f._gaps_down[mag] * s
    v = acc.float() / f.C * torch.where(acc >= 0, gu, gd) - 0.05 * g
    want = v * f.C / torch.where(v >= 0, gu, gd)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    # the step requires tscale for nvfp4
    with pytest.raises(ValueError):
        f.step(codes, sb, g, 0.05)


# ── directional-gap martingale with a non-power-of-two scale ──────────────────

N = 20000
STEPS = 100
LR = 0.05


def _run(fmt, start_val, seed=0):
    """Frozen NVFP4 scale: bytes 54..58 (0.875..1.25) x 0.8137 -> s in ~[0.71, 1.02]."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    nb = N // 16
    sb = (54 + torch.arange(nb) % 5).to(torch.uint8).reshape(1, nb)
    ts = torch.tensor(0.8137)
    s = (e4m3_decode(sb) * ts).repeat_interleave(16, dim=1)[0]
    i = int((fmt.signed - start_val).abs().argmin())
    c = torch.full((1, N), i * fmt.acc_levels + fmt.C - 1, dtype=torch.uint8)
    for _ in range(STEPS):
        c = fmt.step(c, sb, torch.randn(1, N, generator=gen), LR, gen=gen, tscale=ts)
    mag = (c.long() // fmt.acc_levels).flatten()
    acc = ((c.long() % fmt.acc_levels) - (fmt.C - 1)).flatten().float()
    gap = torch.where(acc >= 0, fmt._gaps_up[mag], fmt._gaps_down[mag])
    v = fmt.signed[mag]
    return v, v + acc / fmt.C * gap, s, mag


@pytest.mark.parametrize("name,cap", [("e2m1", 0.03), ("wide", 0.2)])
def test_directional_gap_martingale_under_nonpow2_scale(name, cap):
    """E[(value + pending)/s] stays at the start rung; outer-rung mass stays bounded.

    The pre-SR tick is gap * s with the decoded scale; if the update assumed s = 2^k the
    tick would be mis-sized by up to 1.25/1.0 per block and the mean would drift.
    """
    fmt = nv(name)
    for v0 in [float(v) for v in fmt.signed.tolist() if 0.0 <= v < fmt.signed.max() - 1e-6]:
        v, true, s, mag = _run(fmt, v0)
        # value space: sum over weights of s*(true) is the martingale; compare in rung units
        tol = 5 * float(true.std()) / N ** 0.5 + 0.03
        assert abs(float(true.mean()) - v0) < tol, (name, v0, float(true.mean()), tol)
        outer = float((v.abs() == float(fmt.signed.abs().max())).float().mean())
        assert outer < cap, (name, v0, outer)
