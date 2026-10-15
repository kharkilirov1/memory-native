# Baseline shootout — counter vs real memory-efficient training (T4)

An exploratory experiment comparing the finite-state counter with optimizers
used to save training memory. Run on Kaggle Tesla T4,
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
plain AdamW. The counter reduces **both** separate coefficient optimizer state and activations
(int4), so it's the only one that moves the peak meaningfully. This is the method's real edge.

**On quality it is competitive — not worse.** counter_int4 (2.516) is second-best, essentially
tied with LoMo (2.513) and ahead of AdamW (2.567), GaLore (2.618), 8-bit Adam (2.579). The
spread is within single-seed/short-run noise (~±0.04 seen earlier), so the honest claim is
"similar short-run loss on these settings," not a proof of converged parity or an
absence of a quality penalty at other budgets.

**The cost is throughput — but smaller than first measured.** The shootout's counter row
(3.3k tok/s) used the default tiled update (tile_rows=64). The ~5× gap turned out to be
*kernel-launch overhead from the per-tile Python loop*, not compute: doing the update untiled
(now the default) is **2.9× faster at identical peak** — the separate counter_packed timing runs at **~8.9k
tok/s** on T4, i.e. **~2.1× slower than AdamW (18.7k)** in that separate timing comparison. (Measured: tile_rows 64 →
3.1k, 128 → 5.1k, untiled → 8.9k tok/s, all at 1.30 GiB.)

A surprising negative result worth recording: the Triton forward + grad_x kernels (verified
correct on T4) **do not help here** — no memory benefit (the dense weight is negligible vs the
activation-bound peak) and they are actually *slower* than torch's `decode + cuBLAS` (a naive
non-autotuned Triton matmul loses to cuBLAS). So a fused *update* kernel — not a weight kernel
— is the only remaining throughput lever; the untiled torch path already captures most of it.

These runs show lower training peak for the tested counter configuration, similar
short-run loss, and a throughput cost. The first shootout's 0.99 GiB peak and the
later timing run's 8.9k tok/s are different measurements, not one jointly measured point.

## Two-width check (d=512 → d=768)

Re-run at two sizes (200 steps, batch 16, untiled counter). The measured memory gap
widens over these two configurations; extrapolation beyond them remains untested:

| metric | d=512 (8L) | d=768 (12L) |
|---|---|---|
| counter peak vs AdamW | 1.19× less | **1.32× less** |
| counter peak vs 8-bit Adam | 1.36× less | **1.62× less** |
| counter val vs AdamW | +0.02 | **−0.146 (−4.7%, one short run)** |
| counter speed vs AdamW | ~2.0× slower | **~1.7× slower** |

Raw (d=768): counter+int4 **2.06 GiB / val 2.93 / 3.5k tok/s**; dense+AdamW 2.73 GiB / 3.08 /
6.0k; dense+8-bit-Adam 3.34 GiB / 3.12 / 6.0k. At d=768 counter beats both AdamW and 8-bit Adam
on **peak memory and short-run validation loss**, and the speed gap narrows (counter's per-step overhead
amortizes as the GEMMs grow). Note 8-bit Adam uses *more* peak than plain AdamW at these scales
(bitsandbytes block/overhead outweighs its moment savings until much larger models). The
favorable trajectory is a result on two modest widths, not evidence that converged
quality or the speed/memory advantage will persist at billion-parameter scale.

## Caveats
- Single seed, 400 steps in the first shootout and 200 in the width sweep;
  per-optimizer LRs are not exhaustively tuned (GaLore/LoMo could improve).
  There are no confidence intervals for the width sweep's quality differences.
- GaLore/LoMo/bnb8 peaks landing at/above AdamW is partly their transient buffers (SVD,
  quantization state) at this small scale; the robust signal is counter's clearly-lowest peak.
- d=512 and d=768 are modest scale. A wider memory gap at larger d/batch remains a
  hypothesis; compare complete peaks and matched quality on a named architecture.
