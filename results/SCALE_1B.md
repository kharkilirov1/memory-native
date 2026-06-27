# Scale run — 1.21B-parameter full method, single T4 (enwik8, 2000 steps)

The same full method (`ReversibleGPT`: counter linears + O(1) `ReversibleSequence`) at **1.21B
counter coefficients**, trained end-to-end on **one 14.6 GiB Tesla T4**. The point of this run
is the scale claim: a billion-parameter model that *does not fit* under dense+AdamW trains
comfortably here, and its loss keeps falling on a corpus big enough that 2000 steps is < 1 epoch.
Log: [`gpu_scale_1B_T4.log`](gpu_scale_1B_T4.log) (Kaggle `memory-native-gpu-validate`).

## Memory

| | full method | dense + fp32 + Adam |
|---|---|---|
| persistent state | **871.7 MiB** | 18.00 GiB |
| one-step training peak | **2.25 GiB** | OOM on a 14.6 GiB T4 |

~21× on the weight+optimizer pools (0.72 byte/coeff vs 16), and the reversible stack keeps the
activation pool flat, so the **whole** training step fits in 2.25 GiB — a model that dense+AdamW
can't even allocate (18 GiB of state alone) trains with 12 GiB of headroom to spare.

## Training (enwik8, 38.0M train chars, vocab 203, 2000 steps)

| step | train | val |
|---|---|---|
| 0 | 9.16 | — |
| 250 | 2.99 | 2.95 |
| 500 | 2.70 | 2.76 |
| 1000 | 2.63 | 2.58 |
| 1500 | 2.33 | 2.32 |
| 2000 | **2.08** | **2.05** |
| final eval | 2.08 | **2.10** |

**No overfitting** — val tracks train the whole way; the final held-out eval (2.10) sits right on
top of train (2.08), no gap, exactly as expected when the corpus is large enough that 2000 steps
is a fraction of one epoch (contrast the d=1024 tinyshakespeare run, where ~8 epochs gave a mild
train/val gap). Loss is still descending at step 2000 — the run was step-capped, not converged.

## Cost

177 tok/s, 23,169 s (~6.4 h) for 2000 steps. The throughput tax is the reversible recompute plus
the per-element counter update — the latter is now a fused Triton kernel (see
[`KERNEL.md`](KERNEL.md), ×45.9 on the update / ×1.26 on the step); this 1B run predates wiring
that kernel into the path, so its tok/s is the *pre-kernel* figure.
