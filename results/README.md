# Results — what's been validated (all on a real Tesla T4)

Every number here is from an actual run, not a projection. Logs are the `*.log` files in this
folder; the narrative docs are linked below.

## The headline

**Lowest-memory training of every contestant, at competitive-to-better quality, ~2× slower —
and the advantage grows with scale.** See [`SHOOTOUT.md`](SHOOTOUT.md).

| | counter+int4 | dense+AdamW | dense+8-bit Adam | dense+GaLore | dense+LoMo |
|---|---|---|---|---|---|
| peak @ d=512 | **0.99 GiB** | 1.19 | 1.26 | 1.35 | 1.35 |
| peak @ d=768 | **2.06 GiB** | 2.73 | 3.34 | — | — |
| val @ d=768 | **2.93** | 3.08 | 3.12 | — | — |
| tok/s @ d=512 | 8.9k | 18.7k | 17.6k | 16.9k | 18.4k |

Trajectory d=512 → d=768: peak gap vs AdamW **1.19× → 1.32×**, vs 8-bit Adam **1.36× → 1.62×**;
quality goes from tied to **−4.7% (better)**; speed gap **2.0× → 1.7×**. Everything improves
with scale.

## What was verified, step by step

| topic | finding | log |
|---|---|---|
| Triton forward kernel | decode-in-GEMM matches dense ref (err ≤ 3e-6) | `gpu_validate_T4.log` |
| Triton grad_x kernel | grad_x from packed state matches ref (err ≤ 1e-5); full in-kernel backward trains | `gpu_validate_T4_grad_x.log` |
| parity at scale | counter+RMS ≈/> dense AdamW at d=512 | `gpu_validate_T4.log` |
| activation lever | int8 saved acts: −14% peak for ~1% val (3 seeds) | `gpu_validate_T4_actbits.log` |
| true 4-bit packing | int4-packed acts: −19% peak, lossless | `gpu_validate_T4_int4packed.log` |
| baseline shootout | lowest peak of all; quality competitive | `gpu_shootout_T4.log` |
| scale check | advantage grows d=512 → d=768 | `gpu_shootout_scale_T4.log` |
| throughput | untiled update = 2.9× faster, same peak | `gpu_throughput_T4_v11.log` |

## Honest negatives (recorded, not hidden)

- **Triton forward/grad_x kernels don't help in practice**: no memory benefit (the dense weight
  is negligible vs the activation-bound peak) and they're *slower* than torch `decode + cuBLAS`
  (a naive non-autotuned Triton matmul loses to cuBLAS). They proved the decode-in-GEMM is
  correct, but the real speed lever was the update path, not the weight path.
- **The weight/optimizer side barely moves the peak** at these scales — the peak is
  activation-bound. Quantizing counter-layer *inputs* (int4) is what helps; the remaining
  fp activations (LayerNorm/gelu/attention) are the reversible-block frontier.
- **bit-budget MSE numbers are step-count sensitive** — pin steps/seeds when quoting
  (`DEEP_V2_VERIFICATION.md`).

## Remaining levers (by ROI, not yet done)

1. **Fused update Triton kernel** — the untiled torch update already captures most of the speed;
   a kernel might add ~1.5–2× more but is high-risk (stochastic rounding + packed RMW, hard to
   verify) and naive Triton already lost to torch once. Lower ROI than it looks.
2. **Bigger scale (d≥1024, larger batch)** — needs a bigger GPU than a free T4; the trajectory
   predicts the gap keeps widening.
3. **Reversible blocks** — removes the remaining fp activations (the bulk), but it's an
   architectural change (coupling structure, no-update-during-recompute, depth-sweep for the
   float-reconstruction error). The biggest remaining memory lever, the biggest effort.

> The `*.log` files here are captured on Kaggle Tesla T4. Re-run on your own GPU with
> `scripts/run_scale_validation.sh` or the per-experiment commands in the docs.
