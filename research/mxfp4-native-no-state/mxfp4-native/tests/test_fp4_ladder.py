"""Full FP4 E2M1 ladder (15 signed rungs) in the na_* counters: code storage and NVFP4 scale growth.

1. Overflow. n_codes = 15 * (2C - 1): C <= 9 fits uint8 (<= 255), C >= 10 does not (C = 16 -> 465 codes,
   9 bits). The codes used to be stored in uint8 and silently wrapped mod 256 (the positive rungs vanished).
   Now ``PackedNA.code_dtype`` is uint8 while n_codes <= 256, int16 up to 32768 codes, and the constructor
   raises ValueError beyond; the fused Triton path (uint8 only) is refused instead of guessed.
2. Scale growth with --scale-lr. NVFP4 init puts the largest block on E4M3 byte 126 (= 448), and
   ``scale_code_step`` clamps to [1, 126], so that block could never grow. ``scale_code_step_rebasing``
   doubles the per-tensor scale first (every byte -8, decoded weights preserved) and then applies the step.
3. --headroom-mode ladder --headroom h with nvfp4: block amax -> L/h on the ladder.

Every effect has a null twin (the old primitive / the knob off).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("e1_quality", ROOT / "probes" / "e1_quality.py")
e1 = importlib.util.module_from_spec(_spec)
sys.modules["e1_quality"] = e1
_spec.loader.exec_module(e1)

from mx6.layer import MXNLinear                                # noqa: E402
from mx6.packed_na import PackedNA                             # noqa: E402
from mx6.triton_kernel import packed_matmul, packed_matmul_dx  # noqa: E402

FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
VOCAB, D = 24, 64


def nv(C):
    return PackedNA(FP4, C, scale_format="nvfp4")


def _block_linears(model):
    return [lin for blk in model.blocks for lin in (blk.qkv, blk.proj, blk.fc, blk.out)]


# ── 1. overflow: code storage ─────────────────────────────────────────────────

@pytest.mark.parametrize("C,codes,bits,dtype", [
    (1, 15, 4, torch.uint8), (2, 45, 6, torch.uint8), (4, 105, 7, torch.uint8), (8, 225, 8, torch.uint8),
    (9, 255, 8, torch.uint8),                                  # last C that fits uint8
    (10, 285, 9, torch.int16), (16, 465, 9, torch.int16),     # first overflowing C, and the reported one
])
def test_code_dtype_follows_the_code_count(C, codes, bits, dtype):
    f = nv(C)
    assert (f.n_codes, f.bits, f.code_dtype) == (codes, bits, dtype)
    w = torch.randn(4, 64, generator=torch.Generator().manual_seed(0))
    q, _, _ = f.quantize_scaled(w)
    assert q.dtype == dtype and int(q.max()) < f.n_codes


def test_every_code_round_trips_encode_decode_and_bit_packing():
    f = nv(16)
    mag = torch.arange(f.n_vals).repeat_interleave(f.acc_levels)
    acc = torch.arange(-(f.C - 1), f.C).repeat(f.n_vals)
    code = f.encode(mag, acc)
    assert torch.equal(code, torch.arange(f.n_codes))                 # a bijection onto 0..464
    m2, a2 = f.decode(code)
    assert torch.equal(m2, mag) and torch.equal(a2, acc)
    v = f.value_float(code)
    assert bool((v > 0).any()) and bool((v < 0).any()) and float(v.max()) == 6.0 and float(v.min()) == -6.0
    # bit-packed state (9 bits/code, 8 codes / 9 bytes) round-trips every code, in the unpacked dtype
    in_f = 8 * math.ceil(f.n_codes / 8)
    codes = (torch.arange(2 * in_f) % f.n_codes).reshape(2, in_f).to(f.code_dtype)
    back = f.unpack_state(f.pack_state(codes), in_f)
    assert back.dtype == torch.int16 and torch.equal(back, codes)
    # null twin: the OLD storage (uint8) wraps mod 256 -- exactly the reported corruption
    wrapped = code.to(torch.uint8).long()
    assert not torch.equal(wrapped, code) and int((wrapped // f.acc_levels).max()) < f.n_vals - 1


def test_c16_keeps_both_signs_and_the_same_value_rungs_as_c8():
    """The value rung does not depend on C: C=16 (int16) must decode to the same weights as C=8 (uint8)."""
    torch.manual_seed(0)
    w = torch.randn(16, 128)
    f8, f16 = nv(8), nv(16)
    c8, sb8, ts8 = f8.quantize_scaled(w)
    c16, sb16, ts16 = f16.quantize_scaled(w)
    assert torch.equal(sb8, sb16) and torch.equal(ts8, ts16)
    d8, d16 = f8.dequantize(c8, sb8, ts8), f16.dequantize(c16, sb16, ts16)
    assert torch.equal(d8, d16)
    assert bool((d16 > 0).any()) and bool((d16 < 0).any())
    assert torch.equal(c16.long() // f16.acc_levels, c8.long() // f8.acc_levels)


@pytest.mark.parametrize("gap_mode", ["directional", "min"])
def test_steps_reencode_and_growth_stay_in_range_for_int16_codes(gap_mode):
    f = PackedNA(FP4, 16, gap_mode=gap_mode, scale_format="nvfp4")
    torch.manual_seed(0)
    codes, sb, ts = f.quantize_scaled(torch.randn(8, 64))
    gen = torch.Generator().manual_seed(1)
    seen = set()
    for _ in range(40):
        codes = f.step(codes, sb, torch.randn(8, 64, generator=gen) * 3, 0.05, gen=gen, tscale=ts)
        assert codes.dtype == torch.int16 and int(codes.min()) >= 0 and int(codes.max()) < f.n_codes
        seen.update((codes.long() // f.acc_levels).unique().tolist())
    assert {0, f.n_vals - 1} <= seen and any(r > f._mid_idx for r in seen) and any(r < f._mid_idx for r in seen)
    new, sb2 = f.grow_scale(codes, sb, torch.ones_like(sb, dtype=torch.bool) & (sb < 126), ts)
    assert new.dtype == torch.int16 and int(new.max()) < f.n_codes
    c3, sb3, ch = f.scale_code_step(codes, sb, torch.full(sb.shape, -0.3), ts, gen=gen)
    assert c3.dtype == torch.int16 and int(c3.max()) < f.n_codes and bool(ch.any())


def test_constructor_guard_for_unrepresentable_code_counts():
    # 3 rungs (-1, 0, +1) x (2C - 1): 32763 codes at C = 5461 is the last that fits int16, 32769 at 5462 is not
    ok = PackedNA([0.0, 1.0], 5461)
    assert ok.n_codes == 32763 and ok.code_dtype == torch.int16 and ok.bits == 15
    with pytest.raises(ValueError, match="32769 codes"):
        PackedNA([0.0, 1.0], 5462)
    with pytest.raises(ValueError, match="not representable"):
        nv(2000)                                                      # 15 * 3999 codes
    # the full-ladder format at the C of the failed Colab run now constructs instead of corrupting
    assert nv(16).code_dtype == torch.int16


def test_fused_kernel_is_refused_for_int16_codes_but_not_for_uint8_formats():
    with pytest.raises(ValueError, match="fused"):
        MXNLinear(64, 8, nv(16), fused="on")
    with pytest.raises(ValueError, match="fused"):
        MXNLinear(64, 8, nv(16), fused="auto")
    MXNLinear(64, 8, nv(16), fused="off")                             # unfused int16 is fine
    MXNLinear(64, 8, nv(8), fused="on")                               # null twin: uint8 format unaffected
    x = torch.zeros(2, 64)
    sb = torch.zeros(4, 4, dtype=torch.uint8)
    lut = nv(16).code_to_value
    with pytest.raises(ValueError, match="uint8"):                    # checked before Triton is needed
        packed_matmul(x, torch.zeros(4, 64, dtype=torch.int16), sb, lut,
                      scale_format="nvfp4", tensor_scale=torch.tensor(1.0))
    with pytest.raises(ValueError, match="uint8"):
        packed_matmul_dx(torch.zeros(2, 4), torch.zeros(4, 64, dtype=torch.int16), sb, lut,
                         scale_format="nvfp4", tensor_scale=torch.tensor(1.0))


def test_layer_c16_trains_with_both_signs_and_exact_bit_accounting():
    torch.manual_seed(0)
    f = nv(16)
    layer = MXNLinear(64, 16, f, lr=0.02)
    assert layer.codes.dtype == torch.int16
    for k in range(10):
        layer.apply_update_(torch.randn(16, 64, generator=torch.Generator().manual_seed(k)))
    w = layer.weight()
    assert layer.codes.dtype == torch.int16 and bool((w > 0).any()) and bool((w < 0).any())
    assert int(layer.codes.max()) < f.n_codes
    held = sum(t.numel() * t.element_size() for t in (layer._state, layer._scale, layer.v_rms, layer._tscale))
    assert layer.persistent_bytes() == held and layer._state.dtype == torch.uint8
    assert layer.bits_per_param() == pytest.approx(9 + 0.5 + 32.0 / 64 + 32.0 / (64 * 16), abs=1e-9)


@pytest.mark.parametrize("C,bits", [(1, 4), (2, 6), (4, 7), (8, 8), (16, 9)])
def test_e1_na_fp4_arms_have_the_documented_bits(C, bits, monkeypatch):
    monkeypatch.setattr(e1, "SCALE_FORMAT", "nvfp4")
    assert e1.LADDERS["fp4"] == e1.E2M1_FULL == FP4 and e1.parse_na(f"na_fp4_c{C}") == ("fp4", C)
    model = e1.CharLM(f"na_fp4_c{C}", VOCAB, d=D, blocks=1)
    lin = _block_linears(model)[0]
    assert lin.fmt.bits == bits
    meas, want = e1.measured_bits(model, f"na_fp4_c{C}"), e1.expected_bits(model, f"na_fp4_c{C}")
    assert abs(meas - want) < 0.02 and meas > bits + 0.5          # code bits + E4M3/16 scale + RMS ...


def test_e1_na_fp4_c16_run_keeps_the_positive_rungs(monkeypatch):
    """The reported failure: histogram without any positive rung and val 3.11. Short CPU run, same arm."""
    monkeypatch.setattr(e1, "SCALE_FORMAT", "nvfp4")
    ids = torch.randint(0, VOCAB, (4000,))
    r = e1.run_arm("na_fp4_c16", 0, D, 6, 4, 32, ids, VOCAB, lr=0.01, adam_lr=3e-3, sched="cosine", eval_every=3)
    values, fracs = r["hist"]
    pos = sum(f for v, f in zip(values, fracs) if v > 0)
    neg = sum(f for v, f in zip(values, fracs) if v < 0)
    assert len(values) == 15 and pos > 0.2 and neg > 0.2 and math.isfinite(r["val"])


# ── 2. scale growth past E4M3's 448 ───────────────────────────────────────────

def _scale_state(seed=0, shape=(4, 128), C=8):
    """Arbitrary consistent state: random rungs/accumulators, block bytes 40..100 and ONE block on 448."""
    torch.manual_seed(seed)
    f = nv(C)
    out, nb = shape[0], shape[1] // 16
    mag = torch.randint(0, f.n_vals, shape)
    acc = torch.randint(-(f.C - 1), f.C, shape)
    codes = f.encode(mag, acc).to(f.code_dtype)
    sb = torch.randint(40, 100, (out, nb), dtype=torch.uint8)
    sb[0, 0] = 126
    return f, codes, sb, torch.tensor(1e-3)


def test_rebasing_step_grows_a_block_on_448_and_the_old_step_cannot():
    f, codes, sb, ts = _scale_state()
    top = sb == 126
    ceiling = 448.0 * float(ts)
    dlog = torch.full(sb.shape, 0.25)                        # ~3 code steps up for every block
    # null twin: the unfixed primitive leaves the top block where it is
    c_old, sb_old, ch_old = f.scale_code_step(codes, sb, dlog, ts, gen=torch.Generator().manual_seed(1))
    assert torch.equal(sb_old[top], sb[top]) and not bool(ch_old[top].any())
    assert float(f.decode_scale(sb_old, ts).max()) == pytest.approx(ceiling)       # stuck at 448 * ts
    # fixed primitive: tensor scale doubles, bytes -8, then the step: the top block ends beyond 448 * ts
    c_new, sb_new, ts_new, ch_new, rebased = f.scale_code_step_rebasing(
        codes, sb, dlog, ts, gen=torch.Generator().manual_seed(1))
    assert rebased and float(ts_new) == 2.0 * float(ts)
    assert bool(ch_new[top].all()) and int(sb_new.max()) <= 126
    top_new = float(f.decode_scale(sb_new, ts_new)[top])
    assert top_new > 1.1 * ceiling
    assert f.ladder_max * top_new > 1.1 * f.ladder_max * ceiling                   # max representable magnitude
    # a value-preserving re-encode: every weight moves by at most half the local gap of its NEW rung
    old_v, new_v = f.dequantize(codes, sb, ts), f.dequantize(c_new, sb_new, ts_new)
    i_new = c_new.long() // f.acc_levels
    half_gap = torch.maximum(f._gaps_up[i_new], f._gaps_down[i_new]) * f.expand_scale(sb_new, ts_new) / 2
    assert bool(((new_v - old_v).abs() <= half_gap * (1 + 1e-5) + 1e-9).all())
    assert c_new.dtype == codes.dtype and sb_new.dtype == torch.uint8


def test_rebase_alone_is_value_preserving_for_blocks_the_step_does_not_move():
    f, codes, sb, ts = _scale_state(seed=1)
    dlog = torch.zeros(sb.shape)
    dlog[0, 0] = 0.25                                        # only the 448 block wants to grow
    c1, sb1, ts1, ch, rebased = f.scale_code_step_rebasing(codes, sb, dlog, ts, gen=torch.Generator().manual_seed(2))
    assert rebased and bool(ch[0, 0])
    keep = (~ch).repeat_interleave(16, dim=1)
    assert torch.equal(c1[keep], codes[keep])                                       # codes of untouched blocks
    assert torch.equal(f.dequantize(c1, sb1, ts1)[keep], f.dequantize(codes, sb, ts)[keep])   # ... and values


def test_rebasing_step_is_bit_identical_to_the_old_one_when_nothing_overshoots():
    f, codes, sb, ts = _scale_state(seed=2)
    for sign in (+1.0, -1.0):
        dlog = torch.where(sb < 100, sign * 0.05, 0.0)       # <= 1 code step, blocks far from 126; 448 block idle
        a = f.scale_code_step(codes, sb, dlog, ts, gen=torch.Generator().manual_seed(3))
        b = f.scale_code_step_rebasing(codes, sb, dlog, ts, gen=torch.Generator().manual_seed(3))
        assert bool(a[2].any()) and not b[4] and float(b[2]) == float(ts)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[3])
    z = f.scale_code_step_rebasing(codes, sb, torch.zeros(sb.shape), ts)                # null twin: dlog = 0
    assert torch.equal(z[0], codes) and torch.equal(z[1], sb) and not bool(z[3].any()) and not z[4]


def _grow_layer(monkeypatch=None, stuck=False, C=8, steps=25):
    fmt = nv(C)
    torch.manual_seed(0)
    w = torch.randn(8, 128) * 0.1
    layer = MXNLinear.from_dense(w, fmt, lr=0.0, scale_lr=0.02)       # lr = 0: only the scale moves
    ts0 = float(layer._tscale)
    assert int(layer._scale.max()) == 126
    if stuck:                                                         # the pre-fix behaviour, as a twin
        old = fmt.scale_code_step

        def clamped(codes, sb, dlog, ts, gen=None):
            c, s, ch = old(codes, sb, dlog, ts, gen)
            return c, s, ts, ch, False
        monkeypatch.setattr(fmt, "scale_code_step_rebasing", clamped)
    for k in range(steps):
        torch.manual_seed(k)
        layer.apply_update_(-layer.weight())                          # dL/dln(s) < 0: every block wants to grow
    return layer, ts0


def test_scale_lr_layer_grows_past_448_with_rebase_counted_and_stuck_without(monkeypatch):
    layer, ts0 = _grow_layer()
    s_max = float(layer.fmt.decode_scale(layer._scale, layer._tscale).max())
    assert layer.total_rebased > 0 and float(layer._tscale) == ts0 * 2 ** layer.total_rebased
    assert s_max > 1.2 * 448.0 * ts0                                  # beyond the old ceiling
    assert int(layer._scale.max()) <= 126
    f = layer.fmt                                                     # the readout follows the learned scale
    assert torch.allclose(layer.weight(), f.signed[layer.codes.long() // f.acc_levels]
                          * f.expand_scale(layer._scale, layer._tscale))
    twin, ts0b = _grow_layer(monkeypatch, stuck=True)
    s_max_twin = float(twin.fmt.decode_scale(twin._scale, twin._tscale).max())
    assert twin.total_rebased == 0 and float(twin._tscale) == ts0b
    assert s_max_twin == pytest.approx(448.0 * ts0b)                  # the old ceiling, exactly
    assert twin.total_scale_moves > 0                                 # (other blocks did move)


def test_scale_lr_zero_never_rebases():
    torch.manual_seed(0)
    layer = MXNLinear.from_dense(torch.randn(8, 128) * 0.1, nv(8), lr=0.01)
    ts0, sb0 = float(layer._tscale), layer._scale.clone()
    for k in range(10):
        layer.apply_update_(torch.randn(8, 128, generator=torch.Generator().manual_seed(k)))
    assert layer.total_rebased == 0 and float(layer._tscale) == ts0 and torch.equal(layer._scale, sb0)


def test_pressure_growth_also_passes_448_with_the_full_ladder():
    """--scale-growth pressure already rebases (layer.py); pin it for the fp4 ladder too."""
    fmt = nv(8)
    torch.manual_seed(0)
    layer = MXNLinear(32, 2, fmt, scale_growth="pressure", headroom_mode="ladder", scale_headroom=1.0, lr=0.05)
    ts0 = float(layer._tscale)
    x = torch.ones(4, 32)
    for _ in range(40):
        (-layer(x).sum()).backward()
    assert layer.total_rebased > 0 and layer.total_grown > 0
    assert float(fmt.decode_scale(layer._scale, layer._tscale).max()) > 448.0 * ts0
    torch.manual_seed(0)                                              # null twin: growth off stays at the ceiling
    off = MXNLinear(32, 2, nv(8), scale_growth="off", headroom_mode="ladder", scale_headroom=1.0, lr=0.05)
    ts_off = float(off._tscale)
    for _ in range(40):
        (-off(x).sum()).backward()
    assert off.total_rebased == 0 and float(off._tscale) == ts_off


# ── 3. --headroom-mode ladder --headroom h with nvfp4 ─────────────────────────

def _outer_frac(layer):
    idx = layer.codes.long() // layer.fmt.acc_levels
    return float(((idx == layer.fmt.n_vals - 1) | (idx == 0)).float().mean())


def test_ladder_headroom_nvfp4_maps_block_amax_to_L_over_h():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    L = FP4[-1]
    ref = MXNLinear.from_dense(w, nv(8), scale_headroom=1.0, headroom_mode="ladder")
    fr = {}
    for h in (1.0, 2.0, 4.0):
        lay = MXNLinear.from_dense(w, nv(8), scale_headroom=h, headroom_mode="ladder")
        fr[h] = _outer_frac(lay)
        # every block's amax / s sits at L/h up to the E4M3 rounding of the block scale (<= 6.25 %)
        r = (w.abs() / lay.fmt.expand_scale(lay._scale, lay._tscale)).reshape(64, -1, 16).amax(-1)
        assert float(r.max()) <= L / h * 1.07 and float(r.min()) >= L / h / 1.07
        # the whole headroom is in the per-tensor scale: block bytes are unchanged, tscale = h * tscale(h=1)
        assert torch.equal(lay._scale, ref._scale) and float(lay._tscale) == pytest.approx(h * float(ref._tscale))
    print(f"\nouter-rung mass (fp4 ladder, nvfp4): h=1 {fr[1.0]:.4f}  h=2 {fr[2.0]:.4f}  h=4 {fr[4.0]:.4f}")
    assert fr[1.0] > 0.01                                            # h = 1: block maxima sit on the outer rung
    assert fr[2.0] == 0.0 and fr[4.0] == 0.0                         # room to grow: nothing on +-6 yet


def test_e1_headroom_flags_reach_the_na_fp4_layers(monkeypatch):
    monkeypatch.setattr(e1, "SCALE_FORMAT", "nvfp4")
    monkeypatch.setattr(e1, "HEADROOM_MODE", "ladder")
    out = {}
    for h in (1.0, 4.0):
        monkeypatch.setattr(e1, "HEADROOM", h)
        torch.manual_seed(0)
        model = e1.CharLM("na_fp4_c8", VOCAB, d=D, blocks=1)
        lins = _block_linears(model)
        assert all(m.headroom_mode == "ladder" and m.scale_headroom == h and m.fmt.nvfp4 for m in lins)
        out[h] = sum(_outer_frac(m) * m.codes.numel() for m in lins) / sum(m.codes.numel() for m in lins)
    assert out[1.0] > 0.01 and out[4.0] == 0.0


def test_cli_headroom_flags_set_the_globals(monkeypatch, tmp_path):
    for n in ("RESCALE_EVERY", "SCALE_MODE", "CLIP_FRAC", "SCALE_FORMAT", "HEADROOM_MODE", "HEADROOM",
              "SCALE_GROWTH", "GROWTH_TAU", "COUNTER_WD", "SCALE_LR"):
        monkeypatch.setattr(e1, n, getattr(e1, n))
    data = tmp_path / "t.txt"
    data.write_text("abcdefgh" * 400, encoding="utf-8")
    seen = []

    def fake_run_arm(arm, seed, d, steps, batch, seq, ids, vocab, lr, adam_lr, **kw):
        seen.append((arm, e1.SCALE_FORMAT, e1.HEADROOM_MODE, e1.HEADROOM, e1.SCALE_LR, e1.SCALE_GROWTH))
        return {"arm": arm, "seed": seed, "bits": 1.0, "val": 1.0, "sec": 0.0, "hist": None,
                "clip": float("nan"), "grown": 0, "curve": [(1, 1.0, lr, 0.0, 0.0, float("nan"), 1.0)],
                "best": 1.0, "flip": (0.0, 0.0, 0.0), "edge": 0.0, "train": 1.0, "gap": 0.0,
                "wstats": {"kurt": 3.0, "tail3": 0.0027, "zero": 0.1}, "scale_moves": 0, "rebased": 2}

    monkeypatch.setattr(e1, "run_arm", fake_run_arm)
    e1.main(["--d", "32", "--steps", "2", "--seeds", "0", "--data", str(data), "--arms", "na_fp4_c8",
             "--scale-format", "nvfp4", "--headroom-mode", "ladder", "--headroom", "4", "--scale-lr", "0.003"])
    assert seen == [("na_fp4_c8", "nvfp4", "ladder", 4.0, 0.003, "off")]
