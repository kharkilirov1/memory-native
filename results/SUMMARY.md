# GPU validation — verified results

Real-hardware run of the package on **Kaggle Tesla T4** (torch 2.10.0+cu128, triton 3.6.0).
Raw log: [`gpu_validate_T4.log`](gpu_validate_T4.log). Reproduce with
`scripts/run_scale_validation.sh` on any CUDA box, or the Kaggle kernel that produced this.

## What was verified on the GPU

**1. Triton forward kernel is correct.** The in-GEMM packed-state decode matches the dense
reference within f32 tolerance — this kernel had never been run on hardware before:

| shape (M×K @ N) | max abs err |
|---|---|
| 32×64 @ 48 | 7.2e-07 |
| 64×512 @ 512 | 3.3e-06 |
| 128×256 @ 1024 | 0.0 |

**2. Parity holds — and slightly wins — at d=512.** s512 config (8 layers, d=512), real
tinyshakespeare, 300 steps, AdamW on the non-counter params:

| kind | val loss | gap vs dense | tok/s | peak |
|---|---|---|---|---|
| dense | 2.6393 | +0.0% | 20,338 | 1.64 GiB |
| ternary-QAT | 2.6256 | −0.5% | 18,293 | 1.74 GiB |
| **counter_rms** | **2.5935** | **−1.7%** | 5,534 | 1.35 GiB |
| **counter_packed** | **2.6020** | **−1.4%** | 3,847 | 1.34 GiB |

counter+RMS not only tracks AdamW at this scale, it edges it out — the strongest scale
evidence so far (previous evidence was micro/tiny only). It is slower (per-element decode/
update in PyTorch); the Triton forward addresses part of this, the fused backward the rest.

## Update: Triton grad_x kernel verified, and the peak is activation-bound

A second T4 run ([`gpu_validate_T4_grad_x.log`](gpu_validate_T4_grad_x.log)) added the new
backward kernel `triton_grad_x` (grad_x decoded from packed state, no dense weight):

- **Both kernels correct on hardware**: forward err ≤ 3.6e-6, grad_x err ≤ 1e-5 vs the dense
  reference; full in-kernel forward+grad_x backward recovers a ternary teacher to MSE 0.0.
- **But removing the dense weight does NOT lower the training peak** at s512/batch16:

  | s512, batch 16 | training peak |
  |---|---|
  | counter_packed (torch backward) | 1.57 GiB |
  | counter_packed (Triton fwd + grad_x) | 1.57 GiB |
  | dense + AdamW | 1.64 GiB |

  The Triton path is byte-for-byte the same peak as the torch path. The per-layer weight
  (512×512×4 = 1 MiB) is negligible next to the activation tensors, so eliminating it doesn't
  move a 1.57 GiB peak. **Empirically: the weight/optimizer side is solved; the training peak
  is activation-dominated.** This is exactly the deep-v2 conclusion ("activation is the next
  wall"). The lever that moves the GiB-scale peak is `act_save_bits` / reversible blocks, not
  more weight-side kernels. The in-kernel grad_x still matters for bandwidth and for very large
  d / large-batch regimes where weights dominate — just not for this peak.

## Honest finding on memory

**Training peak is only ~1.05× smaller than dense+AdamW right now** (counter 1.57 GiB vs
dense 1.64 GiB at s512, batch 16), across adamw/galore/lomo baselines. The optimizer-state
saving barely moves the *peak* because at this scale the peak is dominated by activations and
the transient dense weight the pure-PyTorch path still materializes. The real, measured win
is in **persistent state**: counter_packed 34.85 MiB vs counter_rms 40.85 MiB (the packed
0.75 B/weight), and both far below a dense fp32 weights + Adam-moments footprint.

Takeaway: the sub-byte *persistent* claim is real and verified; the sub-byte *training-peak*
claim still needs the fused backward kernel (forming grad_w in registers, updating state in
place) so no dense weight or grad is materialized. That is the one remaining milestone.

(bnb8 / 8-bit Adam was skipped: bitsandbytes not installed in the run image.)
