"""Saturation telemetry, ladder-aware headroom, pressure-driven scale growth (all opt-in).

* telemetry: counts updates whose carry would leave the ladder; numerics unchanged.
* headroom_mode="ladder": scale_headroom is h relative to the ladder max L (target
  amax/s = L/h), so h=1 makes the outer rung reachable; legacy 1.5 does not (null twin).
* scale_growth="pressure": a block that keeps getting clipped at the outer rung bumps its
  scale one step (E4M3 code / x2) and is re-encoded without losing the accumulator.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mx6.layer import MXNLinear                 # noqa: E402
from mx6.packed_na import PackedNA              # noqa: E402

WIDE = ([0.0, 0.5, 3.0], 3)
E2M1 = ([0.0, 0.5, 1.0, 6.0], 5)


def fmt_of(ladder, scale_format="e8m0", **kw):
    mags, C = ladder
    return PackedNA(mags, C, scale_format=scale_format, **kw)


# ── telemetry ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["directional", "min"])
def test_clip_mask_hand_case(mode):
    """wide ladder, scale 1, 32 weights. Push hard: up (g<0) or down (g>0)."""
    f = fmt_of(WIDE, gap_mode=mode)
    n = f.n_vals                                   # 5: -3 -0.5 0 0.5 3
    idx = torch.tensor([4, 4, 0, 0, 2, 3, 1, 4] * 4)
    codes = f.encode(idx, torch.zeros_like(idx)).to(torch.uint8)[None]
    sb = torch.full((1, 1), 127, dtype=torch.uint8)
    # per weight: huge push direction (+1 up / -1 down), so carry is >= 1 rung
    push = torch.tensor([+1, -1, -1, +1, +1, +1, -1, +1] * 4, dtype=torch.float32)[None]
    grad = -push * 400.0                           # update = -lr * grad = push * 4 value units
    gen = torch.Generator().manual_seed(0)
    new, clip = f.step(codes, sb, grad, 0.01, gen=gen, return_clip=True)
    # Directional ticks (gap in the direction of motion): 4 units = 4.8 ticks of the 2.5 gap
    # (carry 1) or 24 ticks of the 0.5 gap (carry 8).
    #  idx 4 (+3) up   -> carry +1 past the top  -> clip;  idx 4 down -> carry -1 -> idx 3, no clip
    #  idx 0 (-3) down -> clip;                          idx 0 up   -> idx 1, no clip
    #  idx 2 (0) up    -> 24 ticks of the 0.5 gap -> carry 8 -> clip
    #  idx 3 (+.5) up  -> 4.8 ticks of the 2.5 gap -> idx 4, no clip (legacy "min" ticks use
    #  the 0.5 gap: 24 ticks -> carry 8 -> clip; that is the old outward bias);  idx 1 down alike
    if mode == "directional":
        want = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1] * 4, dtype=torch.bool)[None]
    else:
        want = torch.tensor([1, 0, 1, 0, 1, 1, 1, 1] * 4, dtype=torch.bool)[None]
    assert torch.equal(clip, want), (clip.int().tolist(), want.int().tolist())
    # the clipped weights ended on the outer rung in the pushed direction
    end = new.long() // f.acc_levels
    assert (end[0, 0] == n - 1) and (end[0, 2] == 0)


def test_telemetry_does_not_change_numerics():
    f = fmt_of(E2M1)
    torch.manual_seed(0)
    codes = torch.randint(0, f.n_codes, (8, 128)).to(torch.uint8)
    sb = torch.full((8, 4), 127, dtype=torch.uint8)
    g = torch.randn(8, 128) * 5
    a = f.step(codes, sb, g, 0.05, gen=torch.Generator().manual_seed(1))
    b, clip = f.step(codes, sb, g, 0.05, gen=torch.Generator().manual_seed(1), return_clip=True)
    assert torch.equal(a, b) and clip.any()


def test_layer_telemetry_counts():
    torch.manual_seed(0)
    layer = MXNLinear(64, 8, fmt_of(WIDE), track_clip=True)
    x = torch.randn(16, 64)
    layer(x).sum().backward()
    assert layer.last_clip_blocks.shape == (8, 2)
    assert float(layer.last_clip_frac) == pytest.approx(layer.last_clip_blocks.sum().item() / (8 * 64))
    assert torch.equal(layer._clip_acc, layer.last_clip_blocks)


# ── ladder-aware headroom ─────────────────────────────────────────────────────

def _outer_frac(layer):
    idx = layer.codes.long() // layer.fmt.acc_levels
    top = layer.fmt.n_vals - 1
    return float(((idx == top) | (idx == 0)).float().mean())


@pytest.mark.parametrize("ladder", [WIDE, E2M1])
def test_ladder_headroom_makes_outer_rung_reachable(ladder):
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    legacy = MXNLinear.from_dense(w, fmt_of(ladder), scale_headroom=1.5)
    lad = MXNLinear.from_dense(w, fmt_of(ladder), scale_headroom=1.0, headroom_mode="ladder")
    print(f"\nouter-rung mass: legacy(1.5)={_outer_frac(legacy):.4f}  ladder(h=1)={_outer_frac(lad):.4f}")
    assert _outer_frac(legacy) == 0.0                 # null twin: unreachable
    assert _outer_frac(lad) > 0.01
    # target amax/s = L: every block's scaled amax is in (L/2, L]
    s = lad.fmt.expand_scale(lad._scale)
    r = (w.abs() / s).reshape(64, -1, 32).amax(-1)
    assert float(r.max()) <= lad.fmt.ladder_max * (1 + 1e-6) and float(r.min()) > lad.fmt.ladder_max / 2
    # default construction is untouched
    d = MXNLinear(64, 8, fmt_of(ladder))
    assert d.headroom_mode == "absolute" and d._eff_headroom() == 1.5


def test_ladder_headroom_nvfp4_keeps_amax_to_L():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    for ladder in (WIDE, E2M1):
        a = MXNLinear.from_dense(w, fmt_of(ladder, "nvfp4"))
        b = MXNLinear.from_dense(w, fmt_of(ladder, "nvfp4"), scale_headroom=1.0, headroom_mode="ladder")
        assert torch.equal(a._scale, b._scale) and torch.equal(a._tscale, b._tscale)
        assert _outer_frac(a) > 0.01


# ── pressure-driven growth ────────────────────────────────────────────────────

def _pressure_layer(scale_format, growth):
    fmt = fmt_of(WIDE, scale_format)
    layer = MXNLinear(32, 2, fmt, scale_growth=growth, headroom_mode="ladder",
                      scale_headroom=1.0, lr=0.05)
    return layer


def _drive(layer, steps=60):
    """Constant outward gradient: loss = -sum(out) pushes every weight up (x = 1)."""
    x = torch.ones(4, 32)
    top = []
    for _ in range(steps):
        out = layer(x)
        (-out.sum()).backward()
        top.append(float(layer.weight().max()))
    return top


@pytest.mark.parametrize("scale_format", ["e8m0", "nvfp4"])
def test_pressure_growth_lets_weights_keep_growing(scale_format):
    torch.manual_seed(0)
    on = _pressure_layer(scale_format, "pressure")
    torch.manual_seed(0)
    off = _pressure_layer(scale_format, "off")
    s0 = on.fmt.decode_scale(on._scale, on._tscale).clone()
    t_on, t_off = _drive(on), _drive(off)
    print(f"\n[{scale_format}] max weight: growth off {t_off[0]:.3f}->{t_off[-1]:.3f}; "
          f"pressure {t_on[0]:.3f}->{t_on[-1]:.3f}; blocks bumped {on.total_grown}")
    assert on.total_grown > 0 and off.total_grown == 0
    assert (on.fmt.decode_scale(on._scale, on._tscale) > s0).any()
    assert torch.equal(off.fmt.decode_scale(off._scale, off._tscale), s0)   # null twin: frozen
    assert t_off[-1] <= t_off[10] * (1 + 1e-6)         # saturated at the outer rung
    assert t_on[-1] > t_off[-1] * 1.05                 # kept increasing past it


@pytest.mark.parametrize("scale_format", ["e8m0", "nvfp4"])
def test_grow_scale_reencode_is_close_and_odd(scale_format):
    f = fmt_of(E2M1, scale_format)
    torch.manual_seed(0)
    w = torch.randn(16, 128) * 0.4
    codes, sb, ts = f.quantize_scaled(w, headroom=3.0 if scale_format == "e8m0" else 6.0)
    # random pending accumulators
    idx = codes.long() // f.acc_levels
    acc = torch.randint(-(f.C - 1), f.C, idx.shape)
    codes = f.encode(idx, acc).to(torch.uint8)
    mask = (torch.rand(16, 128 // f.block) < 0.5) & (sb < (126 if f.nvfp4 else 254))
    new, sb2 = f.grow_scale(codes, sb, mask, ts)
    em = mask.repeat_interleave(f.block, dim=1)
    # untouched blocks are bit-identical, bumped blocks went up exactly one step
    assert torch.equal(new[~em], codes[~em]) and torch.equal(sb2[~mask], sb[~mask])
    assert (sb2[mask].int() - sb[mask].int() == 1).all()
    # decoded value moves by at most half the local gap of the NEW ladder (+eps)
    old_v = f.dequantize(codes, sb, ts)
    new_v = f.dequantize(new, sb2, ts)
    s_new = f.expand_scale(sb2, ts)
    i_new = new.long() // f.acc_levels
    half_gap = torch.maximum(f._gaps_up[i_new], f._gaps_down[i_new]) * s_new / 2
    assert ((new_v - old_v).abs()[em] <= half_gap[em] * (1 + 1e-5) + 1e-7).all()
    # pending value is preserved within one tick of the new rung
    a_new = (new.long() % f.acc_levels) - (f.C - 1)
    gdir = lambda c, a, s: torch.where(a >= 0, f._gaps_up[c // f.acc_levels], f._gaps_down[c // f.acc_levels]) * s
    pend_old = acc.float() / f.C * gdir(codes.long(), acc, f.expand_scale(sb, ts))
    pend_new = a_new.float() / f.C * gdir(new.long(), a_new, s_new)
    tick_new = gdir(new.long(), a_new, s_new) / f.C
    clamp_lim = (f.C - 1) / f.C * gdir(new.long(), a_new, s_new)
    ok = (pend_new - pend_old).abs() <= tick_new / 2 * (1 + 1e-4) + 1e-7
    ok |= pend_old.abs() > clamp_lim                 # clamped to +-(C-1) ticks
    assert ok[em].all()
    # oddness: growing the mirrored state gives the mirrored result
    codes_m = f.encode(f.n_vals - 1 - idx, -acc).to(torch.uint8)
    new_m, sb_m = f.grow_scale(codes_m, sb, mask, ts)
    assert torch.equal(sb_m, sb2)
    assert torch.equal(f.dequantize(new_m, sb_m, ts), -f.dequantize(new, sb2, ts))
    assert torch.equal((new_m.long() % f.acc_levels) - (f.C - 1), -a_new)


def test_rebase_tscale_preserves_decoded_weights():
    """Doubling the per-tensor scale moves normal E4M3 bytes by exactly -8: values unchanged;
    only blocks that fall into the subnormals are re-rounded (within half a rung)."""
    f = fmt_of(E2M1, "nvfp4")
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    codes, sb, ts = f.quantize_scaled(w)
    idx = codes.long() // f.acc_levels
    acc = torch.randint(-(f.C - 1), f.C, idx.shape)
    codes = f.encode(idx, acc).to(torch.uint8)
    c2, sb2, ts2 = f.rebase_tscale(codes, sb, ts)
    assert float(ts2) == 2 * float(ts)
    normal = (sb >= 16)                              # e >= 2 -> exactly representable halved
    assert torch.equal(sb2[normal], sb[normal] - 8)
    em = (sb >= 16).repeat_interleave(16, dim=1)
    assert torch.equal(c2[em], codes[em])
    assert torch.equal(f.dequantize(c2, sb2, ts2)[em], f.dequantize(codes, sb, ts)[em])
    assert int(sb2.max()) <= 118


def test_nvfp4_growth_rebases_when_a_block_sits_on_448():
    torch.manual_seed(0)
    layer = _pressure_layer("nvfp4", "pressure")
    assert int(layer._scale.max()) >= 125            # tensor scale was fitted to the max block
    ts0 = float(layer._tscale)
    _drive(layer, 30)
    assert layer.total_rebased > 0 and float(layer._tscale) > ts0
    assert int(layer._scale.max()) <= 126


def test_grow_scale_saturates_at_the_top_code():
    f = fmt_of(WIDE, "nvfp4")
    codes = torch.full((1, 16), 2 * f.acc_levels + f.C - 1, dtype=torch.uint8)
    sb = torch.tensor([[0x7E]], dtype=torch.uint8)
    new, sb2 = f.grow_scale(codes, sb, torch.ones(1, 1, dtype=torch.bool), torch.tensor(1.0))
    assert int(sb2) == 0x7E and torch.equal(new, codes)


def test_growth_defaults_change_nothing():
    """Same seed, default args vs explicit opt-out args: identical state after training."""
    def run(**kw):
        torch.manual_seed(0)
        layer = MXNLinear(64, 8, fmt_of(E2M1), **kw)
        for _ in range(5):
            layer(torch.randn(8, 64)).pow(2).sum().backward()
        return layer._state.clone(), layer._scale.clone()
    a = run()
    b = run(headroom_mode="absolute", scale_growth="off", track_clip=True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
