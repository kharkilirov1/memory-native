#!/usr/bin/env python3
"""Smoke tests for PackedNA5 (5-bit) and PackedNA6 (6-bit) formats.

Pytest-style (the previous version took ``fmt``/``label`` as plain arguments,
so pytest collected them as two missing fixtures and errored out; the checks
never ran under the suite).

Per format:
  1. Round-trip (quantize -> dequantize) and code count / bit width
  2. One forward + backward changes codes
  3. The accumulator lives inside the code word (no separate buffer)
  4. pack_state / unpack_state are lossless for every code
  5. persistent_bytes matches the packed width (bits + E8M0 scale)
Plus: enable_compile() actually engages torch.compile.

Also runnable as a script:  python tests/test_packed_na.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mx6.packed_na as P
from mx6.packed_na import PackedNA, PackedNA5, PackedNA6
from mx6.layer import MXNLinear

CASES = [(PackedNA5, "PackedNA5"), (PackedNA6, "PackedNA6")]


@pytest.mark.parametrize("fmt,label", CASES)
def test_format(fmt: PackedNA, label: str):
    """Round-trip + code count + bits."""
    torch.manual_seed(42)
    w = torch.randn(4, 64) * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    w_dq = fmt.dequantize(codes, sb)
    rel_err = (w - w_dq).abs().mean() / w.abs().mean()
    assert rel_err < 0.50, f"{label}: round-trip error {rel_err:.3f} too high"
    assert codes.max() < fmt.n_codes, f"{label}: code {codes.max()} >= {fmt.n_codes}"
    assert codes.min() >= 0, f"{label}: negative code {codes.min()}"
    needed = math.ceil(math.log2(fmt.n_codes))
    assert needed == fmt.bits, f"{label}: bits {fmt.bits} != ceil(log2({fmt.n_codes}))={needed}"


@pytest.mark.parametrize("fmt,label", CASES)
def test_one_step(fmt: PackedNA, label: str):
    """One forward + backward must change codes."""
    torch.manual_seed(0)
    layer = MXNLinear(64, 32, fmt, lr=0.1, scale_headroom=2.0)
    layer.train()
    codes_before = layer.codes.clone()
    y = layer(torch.randn(8, 64))
    y.square().mean().backward()
    n_changed = (layer.codes != codes_before).sum().item()
    assert n_changed > 0, f"{label}: no codes changed"


@pytest.mark.parametrize("fmt,label", CASES)
def test_accumulator_is_packed_in_the_code(fmt: PackedNA, label: str):
    """No separate accumulator tensor: the low residue of the code carries it."""
    torch.manual_seed(1)
    layer = MXNLinear(64, 32, fmt, lr=0.5, scale_headroom=2.0)
    layer.train()
    assert "_acc" not in dict(layer.named_buffers()), \
        f"{label}: separate accumulator buffer still registered"

    # every code starts with a neutral accumulator
    assert torch.all(layer.codes.to(torch.long) % fmt.acc_levels == fmt.C - 1)
    y = layer(torch.randn(8, 64))
    y.square().mean().backward()
    residues = torch.unique(layer.codes.to(torch.long) % fmt.acc_levels)
    assert residues.numel() > 1, f"{label}: accumulator never moved inside the code"


@pytest.mark.parametrize("fmt,label", CASES)
def test_pack_state_roundtrip_is_lossless(fmt: PackedNA, label: str):
    """Every valid code must survive pack_state -> unpack_state."""
    codes = (torch.arange(fmt.n_codes).repeat(64 // fmt.n_codes + 1)[:64]
             .to(torch.uint8).reshape(1, 64))
    packed = fmt.pack_state(codes)
    g, b = fmt.group()
    assert packed.shape == (1, (64 // g) * b), f"{label}: unexpected packed width"
    back = fmt.unpack_state(packed, 64)
    assert torch.equal(codes, back), f"{label}: pack/unpack is lossy"


@pytest.mark.parametrize("fmt,label", CASES)
def test_persistent_bytes_is_the_packed_width(fmt: PackedNA, label: str):
    """bits/param must approach code bits + 0.25 E8M0, not a dense uint8 plane."""
    out_f, in_f = 8, 4096
    expected = out_f * fmt.packed_width(in_f) + out_f * (in_f // P.BLOCK) + out_f * 4
    assert fmt.persistent_bytes(out_f, in_f) == expected
    bpp = expected * 8.0 / (out_f * in_f)
    assert abs(bpp - (fmt.bits + 0.25)) < 0.02, f"{label}: {bpp:.3f} bits/param"


def test_enable_compile_engages():
    """enable_compile() must actually wrap the kernels (it used to be inert)."""
    try:
        P.disable_compile()
        plain = P._get_compiled(P._decode_core)
        assert not hasattr(plain, "_torchdynamo_orig_callable")
        P.enable_compile()
        compiled = P._get_compiled(P._decode_core)
        assert hasattr(compiled, "_torchdynamo_orig_callable"), \
            "enable_compile() did not compile the decode kernel"
        assert P._get_compiled(P._decode_core) is compiled, "compiled variant not cached"
    finally:
        P.disable_compile()


if __name__ == "__main__":
    tests = [
        ("5-bit format", lambda: test_format(PackedNA5, "PackedNA5")),
        ("6-bit format", lambda: test_format(PackedNA6, "PackedNA6")),
        ("5-bit one step", lambda: test_one_step(PackedNA5, "PackedNA5")),
        ("6-bit one step", lambda: test_one_step(PackedNA6, "PackedNA6")),
        ("5-bit acc in code", lambda: test_accumulator_is_packed_in_the_code(PackedNA5, "PackedNA5")),
        ("6-bit acc in code", lambda: test_accumulator_is_packed_in_the_code(PackedNA6, "PackedNA6")),
        ("5-bit pack roundtrip", lambda: test_pack_state_roundtrip_is_lossless(PackedNA5, "PackedNA5")),
        ("6-bit pack roundtrip", lambda: test_pack_state_roundtrip_is_lossless(PackedNA6, "PackedNA6")),
        ("5-bit persistent bytes", lambda: test_persistent_bytes_is_the_packed_width(PackedNA5, "PackedNA5")),
        ("6-bit persistent bytes", lambda: test_persistent_bytes_is_the_packed_width(PackedNA6, "PackedNA6")),
        ("enable_compile", test_enable_compile_engages),
    ]
    passed = 0
    for name, fn in tests:
        print(f"=== {name} ===")
        try:
            fn()
            print("  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
        print()
    print(f"SUMMARY: {passed}/{len(tests)} passed")
