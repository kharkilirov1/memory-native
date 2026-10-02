# Historical scale report — 1.21B counter coefficients, single T4 (enwik8, 2000 steps)

**Artifact limitation:** the originally named primary log `gpu_scale_1B_T4.log`
(Kaggle `memory-native-gpu-validate`) is not present in the public repository. The
tables below preserve the historical reported measurements; they cannot currently be
independently audited against the raw trace. No replacement log has been fabricated.
The exact original enwik8 runner/environment are not pinned by this report either.
`scripts/fineweb_1b_2xt4.py` uses the same recorded body dimensions but different
hardware/data; running it is a separate witness. Any new scale execution must record
its own commit, environment, corpus and complete raw log.

The same full method (`ReversibleGPT`: counter linears + O(1) `ReversibleSequence`) at **1.21B
counter coefficients**, reported trained end-to-end on **one 14.6 GiB Tesla T4**. This is
a fit/execution and short learning-curve report. It does not demonstrate converged
language-model quality, parity with a trained dense model, or equivalent behavior when
converting an arbitrary pretrained donor.

## Memory

| | full method, historically reported | dense + FP32 + Adam |
|---|---|---|
| persistent state | **871.7 MiB** | Not separately recorded in this report |
| weights + gradients + Adam moments | Included in peak below | 18.00 GiB allocation requirement reported |
| one-step training peak | **2.25 GiB** | OOM on a 14.6 GiB T4 |

The original ~21× ratio compared packed counter codes with **16 B/coefficient**
for FP32 weights + gradients + Adam moments. FP32 weights + moments alone are
**12 B/coefficient**, giving a **16×** coefficient-state comparison against 0.75 B,
before either method's additional state. Six-bit packing itself costs
**0.75 byte/coefficient**, not 0.72; scale/RMS
buffers and the FP tail are additional. The reported **whole-step** allocation peak is
2.25 GiB. This is not a 21× whole-VRAM comparison: dense OOM gives no matched training
peak, and other baselines (BF16, quantized optimizers, QLoRA) use different storage.

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

The reported train/validation losses remain close over the recorded 2000 steps. That
does not establish an absence of overfitting on longer runs or broader tasks. The run
was step-capped, not converged, and has no completed matched-quality dense baseline.

## Cost

177 tok/s, 23,169 s (~6.4 h) for 2000 steps. The throughput tax is the reversible recompute plus
the per-element counter update — the latter is now a fused Triton kernel (see
[`KERNEL.md`](KERNEL.md), ×45.9 on the update / ×1.26 on the step); this 1B run predates wiring
that kernel into the path, so its tok/s is the *pre-kernel* figure.
