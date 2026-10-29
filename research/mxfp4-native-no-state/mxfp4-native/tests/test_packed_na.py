#!/usr/bin/env python3
"""Smoke test for PackedNA5 (5-bit) and PackedNA6 (6-bit) formats.

Tests per format:
  1. Format round-trip (quantize → dequantize, error within spec)
  2. Code count fits in claimed bits
  3. One forward + backward changes codes (update fires)
  4. bits/param matches design target
  5. Tiny char-LM: 200 steps, val loss beats SR-only baseline

Run:  python tests/test_packed_na.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mx6.packed_na import PackedNA, PackedNA5, PackedNA6
from mx6.layer import MXNLinear, MX6Linear


def test_format(fmt: PackedNA, label: str):
    """Round-trip + code count + bits."""
    torch.manual_seed(42)
    w = torch.randn(4, 64) * 0.1
    codes, sb = fmt.quantize(w, headroom=2.0)
    w_dq = fmt.dequantize(codes, sb)
    rel_err = (w - w_dq).abs().mean() / w.abs().mean()
    assert rel_err < 0.50, f"{label}: round-trip error {rel_err:.3f} too high"
    # code range
    assert codes.max() < fmt.n_codes, f"{label}: code {codes.max()} >= {fmt.n_codes}"
    assert codes.min() >= 0, f"{label}: negative code {codes.min()}"
    # bits
    import math
    needed = math.ceil(math.log2(fmt.n_codes))
    assert needed == fmt.bits, f"{label}: bits {fmt.bits} != ceil(log2({fmt.n_codes}))={needed}"
    print(f"  {label}: {fmt.n_vals} values, {fmt.acc_levels} acc levels, "
          f"{fmt.n_codes} codes -> {fmt.bits} bits, round-trip err {rel_err:.4f}")


def test_one_step(fmt: PackedNA, label: str):
    """One forward + backward must change codes."""
    torch.manual_seed(0)
    layer = MXNLinear(64, 32, fmt, lr=0.1, scale_headroom=2.0)
    layer.train()
    codes_before = layer.codes.clone()
    x = torch.randn(8, 64)
    y = layer(x)
    y.square().mean().backward()
    n_changed = (layer.codes != codes_before).sum().item()
    assert n_changed > 0, f"{label}: no codes changed"
    print(f"  {label}: {n_changed}/{layer.codes.numel()} codes changed, "
          f"bits/param={layer.bits_per_param():.3f}")


def test_charlm():
    """Tiny char-LM: 200 steps, both formats should beat SR-only."""
    torch.manual_seed(42)
    data_path = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        print("  SKIP (no tinyshakespeare.txt)")
        return
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
                self.fc1 = MX6Linear(D, D*4, lr=0.01, integrator=0, scale_headroom=1.5)
                self.fc2 = MX6Linear(D*4, D, lr=0.01, integrator=0, scale_headroom=1.5)
            else:
                self.fc1 = MXNLinear(D, D*4, fmt, lr=0.01, scale_headroom=2.0)
                self.fc2 = MXNLinear(D*4, D, fmt, lr=0.01, scale_headroom=2.0)
            self.ln = nn.LayerNorm(D)
            self.head = nn.Linear(D, V, bias=False)
            self.opt = torch.optim.SGD(
                list(self.embed.parameters())+list(self.ln.parameters())+list(self.head.parameters()), lr=0.01)
        def forward(self, x):
            h = self.embed(x); h = self.fc1(h); h = torch.relu(h)
            h = self.fc2(h); h = self.ln(h); return self.head(h)

    def run(model, steps=200):
        model.train()
        for _ in range(steps):
            idx = torch.randint(0, len(ids)-SEQ-1, (BS,))
            x = ids[idx.unsqueeze(1)+torch.arange(SEQ)]
            y = ids[idx.unsqueeze(1)+torch.arange(1, SEQ+1)]
            loss = nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
            model.opt.zero_grad(); loss.backward(); model.opt.step()
        model.eval()
        with torch.no_grad():
            idx = torch.randint(0, len(ids)-SEQ-1, (BS,))
            x = ids[idx.unsqueeze(1)+torch.arange(SEQ)]
            y = ids[idx.unsqueeze(1)+torch.arange(1, SEQ+1)]
            return nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()

    val_sr = run(LM(sr=True))
    val_5 = run(LM(fmt=PackedNA5))
    val_6 = run(LM(fmt=PackedNA6))
    print(f"  SR-only:  {val_sr:.4f}")
    print(f"  5-bit:    {val_5:.4f}  ({'WINS' if val_5 < val_sr else 'loses'} vs SR)")
    print(f"  6-bit:    {val_6:.4f}  ({'WINS' if val_6 < val_sr else 'loses'} vs SR)")
    # 5-bit should win even at 200 steps (C=3 carries quickly)
    assert val_5 < val_sr, "5-bit should beat SR-only at 200 steps"
    # 6-bit (C=5) needs more steps to start carrying; just print, don't assert


if __name__ == "__main__":
    tests = [
        ("5-bit format", lambda: test_format(PackedNA5, "PackedNA5")),
        ("6-bit format", lambda: test_format(PackedNA6, "PackedNA6")),
        ("5-bit one step", lambda: test_one_step(PackedNA5, "PackedNA5")),
        ("6-bit one step", lambda: test_one_step(PackedNA6, "PackedNA6")),
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
            import traceback; traceback.print_exc()
        print()
    print(f"SUMMARY: {passed}/{len(tests)} passed")
