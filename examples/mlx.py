#!/usr/bin/env python3
"""memory-native on MLX — end-to-end demo (runs on a MacBook; CPU fallback anywhere).

A tiny char-LM whose entire transformer-ish body is counter synapses inside reversible
coupling blocks: no FP master weights, no Adam moments, 0.75 byte/weight persistent state
(packed), O(1)-in-depth activation referencing. Only the embedding and the output head are
ordinary trainable parameters (mlx AdamW). The counter layers train THEMSELVES inside the
VJP of one nn.value_and_grad call per step.

    python scripts/mlx_demo.py                 # synthetic corpus, ~1 min on an M-series GPU
    DATA_PATH=tinyshakespeare.txt python scripts/mlx_demo.py

On Apple silicon the packed layers route their update through the fused Metal kernel
automatically; elsewhere (e.g. Linux mlx[cpu]) they use the identical pure-MLX math.
"""
from __future__ import annotations

import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from memory_native_mlx import PackedRMSCounterLinear, ReversibleCouplingBlock, ReversibleSequence
from memory_native_mlx.metal_update import metal_available

VOCAB = 64
DIM = 128            # reversible channel dim (block halves are DIM // 2)
DEPTH = 4            # reversible blocks; activation referencing is O(1) in this
SEQ, BATCH, STEPS = 64, 16, 300


def corpus() -> mx.array:
    path = os.environ.get("DATA_PATH", "")
    if path and os.path.exists(path):
        text = open(path, "rb").read()
        data = mx.array([b % VOCAB for b in text[:2_000_000]], dtype=mx.int32)
        print(f"corpus: {path} ({data.size} bytes)")
        return data
    # offline fallback: a synthetic Markov-ish stream with learnable structure
    key = mx.random.key(0)
    n = 200_000
    steps = mx.random.randint(1, 5, (n,), key=key)
    data = mx.cumsum(steps) % VOCAB
    print(f"corpus: synthetic ({n} tokens)")
    return data.astype(mx.int32)


class CounterCharLM(nn.Module):
    """embed (AdamW) -> reversible counter body (self-updating) -> head (AdamW)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        d = DIM // 2
        self.body = ReversibleSequence([
            ReversibleCouplingBlock(
                DIM,
                PackedRMSCounterLinear(d, d, C=11, lr=6e-3, key=mx.random.key(10 + i)),
                PackedRMSCounterLinear(d, d, C=11, lr=6e-3, key=mx.random.key(50 + i)),
            )
            for i in range(DEPTH)
        ])
        self.head = nn.Linear(DIM, VOCAB)

    def __call__(self, tokens):
        h = self.embed(tokens)
        h = self.body(h)
        return self.head(h)


def main() -> None:
    data = corpus()
    model = CounterCharLM()
    opt = optim.AdamW(learning_rate=3e-3)

    counters = [b.F for b in model.body.layers] + [b.G for b in model.body.layers]
    n_counter_weights = sum(c.in_features * c.out_features for c in counters)
    n_state_bytes = sum(c.codes.size for c in counters)
    print(f"backend: {'Metal (fused update kernel)' if metal_available() else 'CPU (pure-MLX fallback)'}")
    print(f"counter body: {n_counter_weights} weights -> {n_state_bytes} bytes persistent state "
          f"({n_state_bytes / n_counter_weights:.2f} B/weight; no master weights, no Adam moments)")

    def loss_fn(m, xb, yb):
        logits = m(xb)
        return nn.losses.cross_entropy(logits, yb).mean()

    vg = nn.value_and_grad(model, loss_fn)
    key = mx.random.key(42)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        key, sub = mx.random.split(key)
        starts = mx.random.randint(0, data.size - SEQ - 1, (BATCH,), key=sub)
        idx = starts[:, None] + mx.arange(SEQ)[None, :]
        xb, yb = data[idx], data[idx + 1]
        loss, grads = vg(model, xb, yb)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  ({(time.time() - t0):.1f}s)")

    stats = counters[0].state_statistics()
    print("first counter layer state:",
          " ".join(f"{k}={v:.3f}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
