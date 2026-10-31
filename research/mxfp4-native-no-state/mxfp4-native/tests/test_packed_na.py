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
import torch.nn as nn

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


def test_charlm():
    """Tiny char-LM: 200 steps, the 5-bit format should beat SR-only."""
    from mx6.layer import MX6Linear

    torch.manual_seed(42)
    data_path = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        pytest.skip("no tinyshakespeare.txt")
    text = data_path.read_text(encoding="utf-8")[:8192]
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    V = len(chars)
    ids = torch.tensor([vocab[c] for c in text], dtype=torch.long)
    D, SEQ, BS = 64, 32, 16

    class LM(nn.Module):
        def __init__(self, fmt=None, sr=False):
            super().__init__()
            self.embed = nn.Embedding(V, D)
            if sr:
                self.fc1 = MX6Linear(D, D * 4, lr=0.01, integrator=0, scale_headroom=1.5)
                self.fc2 = MX6Linear(D * 4, D, lr=0.01, integrator=0, scale_headroom=1.5)
            else:
                self.fc1 = MXNLinear(D, D * 4, fmt, lr=0.01, scale_headroom=2.0)
                self.fc2 = MXNLinear(D * 4, D, fmt, lr=0.01, scale_headroom=2.0)
            self.ln = nn.LayerNorm(D)
            self.head = nn.Linear(D, V, bias=False)
            self.opt = torch.optim.SGD(
                list(self.embed.parameters()) + list(self.ln.parameters())
                + list(self.head.parameters()), lr=0.01)

        def forward(self, x):
            h = self.embed(x)
            h = self.fc1(h)
            h = torch.relu(h)
            h = self.fc2(h)
            h = self.ln(h)
            return self.head(h)

    def run(model, steps=200):
        model.train()
        for _ in range(steps):
            idx = torch.randint(0, len(ids) - SEQ - 1, (BS,))
            x = ids[idx.unsqueeze(1) + torch.arange(SEQ)]
            y = ids[idx.unsqueeze(1) + torch.arange(1, SEQ + 1)]
            loss = nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            model.opt.zero_grad()
            loss.backward()
            model.opt.step()
        model.eval()
        with torch.no_grad():
            idx = torch.randint(0, len(ids) - SEQ - 1, (BS,))
            x = ids[idx.unsqueeze(1) + torch.arange(SEQ)]
            y = ids[idx.unsqueeze(1) + torch.arange(1, SEQ + 1)]
            return nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()

    val_sr = run(LM(sr=True))
    val_5 = run(LM(fmt=PackedNA5))
    assert val_5 < val_sr, f"5-bit should beat SR-only at 200 steps ({val_5:.4f} vs {val_sr:.4f})"


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
        ("char-LM sanity", test_charlm),
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
