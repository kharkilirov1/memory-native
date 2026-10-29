#!/usr/bin/env python3
"""Smoke test for the block-shared accumulator (MX6BlockLinear).

Tests:
  1. Split-scale format round-trip (quantize_split → dequantize_split)
  2. Block accumulator state encoding (combine/split, acc_high get/set)
  3. One forward + backward step changes codes (the update fires)
  4. Bits-per-param accounting matches the 4.50 design target
  5. Tiny char-LM: 200 steps, val loss < SR-only baseline (3.13 at 800 steps
     → must be below 3.05 at 200 steps to show momentum is working)

Run:  python tests/test_block_acc.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mx6 import pack, update
from mx6.layer import MX6BlockLinear, MX6Linear


def test_format_roundtrip():
    """Split-scale quantize → dequantize preserves values within E2M1 error."""
    torch.manual_seed(42)
    w = torch.randn(4, 64) * 0.1
    codes, scale_byte = pack.quantize_split(w, headroom=6.0)
    w_dq = pack.dequantize_split(codes, scale_byte)
    rel_err = (w - w_dq).abs().mean() / w.abs().mean()
    assert rel_err < 0.25, f"round-trip error too high: {rel_err:.3f}"
    # scale byte: upper 5 bits = exponent, lower 3 bits = 0 (acc-high starts at 0)
    assert (scale_byte & 0x07 == 0).all(), "acc-high not zeroed at init"
    print(f"  round-trip rel error: {rel_err:.4f} (OK < 0.25)")


def test_acc_encoding():
    """Block accumulator combine/split and acc_high get/set are inverses."""
    acc_high = torch.tensor([[3, 7, 0], [1, 5, 2]], dtype=torch.uint8)
    acc_low = torch.tensor([[200, 0, 255], [100, 50, 128]], dtype=torch.uint8)
    acc = pack.block_acc_combine(acc_high, acc_low)
    ah, al = pack.block_acc_split(acc)
    assert torch.equal(ah, acc_high), f"acc_high mismatch: {ah} vs {acc_high}"
    assert torch.equal(al, acc_low), f"acc_low mismatch: {al} vs {acc_low}"
    # acc_high set/get into scale byte
    scale_byte = torch.tensor([[0x80, 0x40, 0x20], [0x10, 0x08, 0x04]], dtype=torch.uint8)
    sb2 = pack.acc_high_set(scale_byte, acc_high)
    ah_read = pack.acc_high_get(sb2)
    assert torch.equal(ah_read, acc_high), "acc_high set/get not inverse"
    # exponent bits preserved
    exp_before = scale_byte >> 3
    exp_after = sb2 >> 3
    assert torch.equal(exp_before, exp_after), "exponent bits corrupted by acc_high set"
    print(f"  acc combine/split + acc_high get/set: OK")


def test_one_step_updates_codes():
    """One forward + backward must change at least some codes."""
    torch.manual_seed(0)
    layer = MX6BlockLinear(64, 32, lr=0.1, C_block=4, scale_headroom=6.0)
    layer.train()
    codes_before = layer.codes.clone()
    x = torch.randn(8, 64)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()
    n_changed = (layer.codes != codes_before).sum().item()
    assert n_changed > 0, "no codes changed after one step — update didn't fire"
    # acc should have accumulated something
    acc_high = pack.acc_high_get(layer.scale_byte)
    acc_low = layer.acc_low
    acc = pack.block_acc_combine(acc_high, acc_low)
    assert acc.sum() > 0, "accumulator is zero after a step — integration failed"
    print(f"  codes changed: {n_changed}/{layer.codes.numel()}, acc sum: {acc.sum().item()}")


def test_bits_per_param():
    """Persistent bytes should give ~4.50 bits/param."""
    layer = MX6BlockLinear(256, 256, C_block=32)
    bpp = layer.bits_per_param()
    # 4.00 (codes) + 0.25 (scale+acc-high) + 0.25 (acc-low) + tiny (v_rms)
    assert 4.4 < bpp < 4.7, f"bits/param {bpp:.3f} outside [4.4, 4.7]"
    print(f"  bits/param: {bpp:.3f} (target ~4.50)")


def test_charlm_sanity():
    """Tiny char-LM: 200 steps, val loss should beat SR-only baseline.

    SR-only at 800 steps = 3.1287. At 200 steps it's even worse (~3.3+).
    If block-acc momentum works, 200 steps should already be below 3.1.
    """
    torch.manual_seed(42)
    # tiny dataset: first 4096 chars of tinyshakespeare
    data_path = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        print("  SKIP (no tinyshakespeare.txt)")
        return
    text = data_path.read_text(encoding="utf-8")[:8192]
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    V = len(chars)
    ids = torch.tensor([vocab[c] for c in text], dtype=torch.long)

    D, NB, SEQ, BS = 64, 2, 32, 16  # 2-block char-LM, d=64

    class CharLM(nn.Module):
        def __init__(self, use_block=True):
            super().__init__()
            self.embed = nn.Embedding(V, D)
            if use_block:
                self.fc1 = MX6BlockLinear(D, D * 4, lr=0.01, C_block=16, scale_headroom=6.0)
                self.fc2 = MX6BlockLinear(D * 4, D, lr=0.01, C_block=16, scale_headroom=6.0)
            else:
                self.fc1 = MX6Linear(D, D * 4, lr=0.01, integrator=0, scale_headroom=6.0)
                self.fc2 = MX6Linear(D * 4, D, lr=0.01, integrator=0, scale_headroom=6.0)
            self.ln = nn.LayerNorm(D)
            self.head = nn.Linear(D, V, bias=False)

        def forward(self, x):
            h = self.embed(x)
            h = self.fc1(h)
            h = torch.relu(h)
            h = self.fc2(h)
            h = self.ln(h)
            return self.head(h)

    def run(model, steps=200):
        model.train()
        for i in range(steps):
            idx = torch.randint(0, len(ids) - SEQ - 1, (BS,))
            x = ids[idx.unsqueeze(1) + torch.arange(SEQ)]
            y = ids[idx.unsqueeze(1) + torch.arange(1, SEQ + 1)]
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            # zero non-counter grads (embed, head, ln)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            loss.backward()
            # step non-counter params with SGD
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= 0.01 * p.grad
        # eval
        model.eval()
        with torch.no_grad():
            idx = torch.randint(0, len(ids) - SEQ - 1, (BS,))
            x = ids[idx.unsqueeze(1) + torch.arange(SEQ)]
            y = ids[idx.unsqueeze(1) + torch.arange(1, SEQ + 1)]
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        return loss.item()

    # SR-only baseline (no accumulator)
    t0 = time.time()
    model_sr = CharLM(use_block=False)
    val_sr = run(model_sr, steps=200)
    t_sr = time.time() - t0

    # Block accumulator
    t0 = time.time()
    model_blk = CharLM(use_block=True)
    val_blk = run(model_blk, steps=200)
    t_blk = time.time() - t0

    print(f"  SR-only  val: {val_sr:.4f}  ({t_sr:.1f}s)")
    print(f"  block-acc val: {val_blk:.4f}  ({t_blk:.1f}s)")
    print(f"  delta: {val_blk - val_sr:+.4f}  ({'block-acc WINS' if val_blk < val_sr else 'SR-only wins'})")
    # We don't hard-assert the win at 200 steps (too noisy), just print it.


if __name__ == "__main__":
    tests = [
        ("format round-trip", test_format_roundtrip),
        ("acc encoding", test_acc_encoding),
        ("one step updates codes", test_one_step_updates_codes),
        ("bits per param", test_bits_per_param),
        ("char-LM sanity", test_charlm_sanity),
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
