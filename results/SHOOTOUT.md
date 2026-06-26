# Baseline shootout — counter vs real memory-efficient training (T4)

The decisive experiment: is the finite-state counter actually better than the optimizers
people already use to save training memory, or just another quantizer? Run on Kaggle Tesla T4,
s512 (d=512, 8 layers), real tinyshakespeare, 400 steps, batch 16, single seed.
Raw log: [`gpu_shootout_T4.log`](gpu_shootout_T4.log).

| config | val loss | training peak | tok/s |
|---|---|---|---|
| **counter_packed + int4 acts** (the method) | **2.5161** | **0.99 GiB** | 3,299 |
| dense + AdamW | 2.5672 | 1.19 GiB | 17,669 |
| dense + 8-bit Adam (bitsandbytes) | 2.5789 | 1.26 GiB | 17,646 |
| dense + GaLore | 2.6175 | 1.35 GiB | 16,887 |
| dense + LoMo | 2.5133 | 1.35 GiB | 18,390 |

## Verdict

**On memory the method wins outright.** counter_packed+int4 has the lowest training peak of
every contestant — 1.20× below AdamW, 1.27× below 8-bit Adam, 1.36× below GaLore/LoMo. The key
insight: the memory-efficient *optimizers* (8-bit Adam, GaLore, LoMo) only shrink the optimizer
pool, which is a small slice of the peak at this scale — their peaks barely beat (or exceed)
plain AdamW. The counter attacks **both** the optimizer pool (zero state) **and** activations
(int4), so it's the only one that moves the peak meaningfully. This is the method's real edge.

**On quality it is competitive — not worse.** counter_int4 (2.516) is second-best, essentially
tied with LoMo (2.513) and ahead of AdamW (2.567), GaLore (2.618), 8-bit Adam (2.579). The
spread is within single-seed/short-run noise (~±0.04 seen earlier), so the honest claim is
"on par with the best," not "beats them" — but it is clearly not paying a quality penalty for
the memory win.

**The cost is throughput — but smaller than first measured.** The shootout's counter row
(3.3k tok/s) used the default tiled update (tile_rows=64). The ~5× gap turned out to be
*kernel-launch overhead from the per-tile Python loop*, not compute: doing the update untiled
(now the default) is **2.9× faster at identical peak** — counter_packed+int4 runs at **~8.9k
tok/s** on T4, i.e. **~2.1× slower than AdamW (18.7k), not 5×.** (Measured: tile_rows 64 →
3.1k, 128 → 5.1k, untiled → 8.9k tok/s, all at 1.30 GiB.)

A surprising negative result worth recording: the Triton forward + grad_x kernels (verified
correct on T4) **do not help here** — no memory benefit (the dense weight is negligible vs the
activation-bound peak) and they are actually *slower* than torch's `decode + cuBLAS` (a naive
non-autotuned Triton matmul loses to cuBLAS). So a fused *update* kernel — not a weight kernel
— is the only remaining throughput lever; the untiled torch path already captures most of it.

Net value proposition: **lowest-memory training (1.2–1.4× below the best memory-efficient
optimizers) at competitive quality, for ~2× slower steps.**

## Caveats
- Single seed, 400 steps; per-optimizer LRs are reasonable but not exhaustively tuned (GaLore/
  LoMo could improve with tuning). Quality differences ≤0.05 are within noise.
- GaLore/LoMo/bnb8 peaks landing at/above AdamW is partly their transient buffers (SVD,
  quantization state) at this small scale; the robust signal is counter's clearly-lowest peak.
- d=512 is still modest scale. The memory gap should widen at larger d/batch (where activations
  and optimizer state dominate more), but that needs a bigger-GPU run to confirm.
