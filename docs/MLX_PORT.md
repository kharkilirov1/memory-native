# memory-native on MLX / Metal — the macOS port

The MLX port lives in [`src/memory_native_mlx/`](../src/memory_native_mlx/).
Code encoding/packing and covered state round-trips match PyTorch bit-for-bit on the
Linux `mlx[cpu]` backend. Numerical update parity is tested with tolerances for FP
reduction order and stochastic-rounding boundary flips; it is not a promise of
bit-identical training on all shapes/backends. Packed codes cost 0.75 B/coefficient,
with scales/statistics and the model tail additional. The fused Metal kernel source
is present, but no committed Apple-silicon execution/performance witness exists.
Tests are [`test_mlx_port.py`](../tests/test_mlx_port.py) and
[`test_mlx_group_scale.py`](../tests/test_mlx_group_scale.py).

## Why a MacBook is a natural home for this method

The proposed benefit is lower coefficient-training state: no separate FP master
weights or Adam moments for the counter body, plus reversible activations. The
historical CUDA 1.21B fit report is limited by its missing primary log and does not
validate a Mac memory budget. On Apple silicon, **fine-tuning on unified memory**
is a use case to investigate. An M-series
MacBook has 16–128 GB of memory shared between CPU and GPU, no PCIe transfer, and a mature
local-ML culture around MLX — but full fine-tuning there is normally killed by exactly the
pools this method attacks (a dense-7B + Adam wants ~84 GB of weights+moments+grads in fp32
before activations). Six-bit codes for 7B coefficients alone would use 5.25 GB decimal;
group scales/statistics, metadata, FP tail and its optimizer, temporary weights/correlation,
and activations add memory. Complete donor recovery on a laptop remains unvalidated.

The PyTorch fine-tuning entry path is public in this checkout: the PTQ warm-start
(GPTQ-ternary import) and behavior-recovery pipeline on
the recovery/solver work in this repo produces
counter-format models from pretrained checkpoints. Because the MLX port shares the exact
state encoding and packing, covered row-scale layers cross over losslessly (see *Interop*
below). A full-model bridge for group-scale solver checkpoints and recovery on a Mac
still needs implementation and hardware validation.

## Design mapping (torch -> MLX)

| PyTorch reference | MLX port | Notes |
|---|---|---|
| `torch.autograd.Function` with in-backward self-update | `mx.custom_function` with a custom VJP that applies the counter transition | grad_w is a transient dense VJP tensor; no retained coefficient `.grad` buffer, but scratch contributes to peak memory |
| `tap` scalar (forces backward to run) | `tap` — a real trainable 0-scalar parameter | `nn.value_and_grad` only differentiates paths reaching a trainable parameter; tap threads every counter layer into the diff set. Gets zero grad; 0 is an AdamW weight-decay fixpoint |
| `register_buffer` (state/scale/v) | public arrays, `freeze()`-d | saved by `save_weights`, evaluated by `mx.eval(model.parameters())`, invisible to optimizers. (Named `codes`, not `state` — `mlx.nn.Module` owns `.state`) |
| `torch.rand` stochastic rounding | **hash-SR always** (the Triton/OpenCL deterministic scheme) | explicit hash seeds make the rounding variates reproducible without depending on backend RNG consumption; floating reductions may still change update thresholds |
| Triton `_counter_update_kernel` (packed, one launch/row) | `mx.fast.metal_kernel` in [`metal_update.py`](../src/memory_native_mlx/metal_update.py) | same two-pass structure, same hash; functional outputs (MLX kernels don't mutate) |
| `_ReversibleSequenceFn` (whole-chain, stores only output) | whole-chain `mx.custom_function`; VJP inverts block-by-block and recomputes locally via `mx.vjp` | inner counter layers self-update exactly once per block during the walk (tested); `anchor_every=A` checkpoint mode ported too |
| in-backward DDP all-reduce | not ported (v1) | `mlx.distributed` exists; single-Mac training doesn't need it |
| `PackedRMSCounterLinear` 4-codes/3-bytes | identical bit layout | packed bytes verified equal to the torch layer's buffer in tests |

Contract carries over unchanged: eager-only (one forward → one VJP per step), train through
`nn.value_and_grad` (a plain call runs no VJP and therefore never updates — that *is* the
inference path), no `mx.compile` over the update path (the SR seed advances in Python).

## What is validated, where

- **Linux, `mlx[cpu]` backend + torch reference side by side** (this is how the port was
  developed; runs in CI without any Apple hardware):
  encode/decode, pack/unpack and `hash_u32` match torch **bit-for-bit**; the full RMS+SR
  update is compared with `memory_native.fused_update.counter_update_hashsr` on the
  covered shapes with explicit scale/RMS and code-mismatch tolerances (floating
  reduction order can cause stochastic-rounding boundary flips);
  packed and unpacked layers stay **bit-identical through training**; teacher recovery and
  loss-decrease gates pass with the same architecture/lr/thresholds as the torch tests;
  reversible chain (pure and anchored) matches the plainly-differentiated stack's grads;
  mixed model (AdamW head + self-updating counter body) trains in one `value_and_grad` loop;
  torch→MLX→torch round-trip preserves state exactly and forward outputs to 1e-5.
- **On an Apple-silicon Mac (next gate — needs real hardware):** run the same
  `pytest tests/test_mlx_port.py`; `test_metal_fused_update_matches_reference` stops
  skipping and gates the fused Metal kernel against the pure-MLX reference. Then
  `python scripts/mlx_demo.py` for the end-to-end smoke (on Linux CPU it reaches
  loss 188→1.5 in ~50 s at 0.75 B/weight state).

## What is NOT ported yet (deliberate v1 cuts)

- `act_save_bits` (int8/int4 saved activations) and the int8/int4/fp8 `update_compute`
  estimators — MLX-side quantized correlation is a follow-up; the RMS+SR core doesn't
  depend on it.
- `cache_mode` (derived T-cache) and update decimation — pure-MLX forward currently
  decodes the dense weight around the GEMM (same as the torch base path). A
  decode-in-GEMM Metal kernel (MLX's own quantized-matmul style) is the natural next
  kernel after the fused update.
- MLX-native solver-v3 recovery runner / GLM-MoE harnesses — PyTorch versions are public
  in this checkout, but the row-scale interop helper is not a whole-model/group-checkpoint
  bridge.
- `scale_rebase="lazy"`, proxy RMS mode, DDP.

(The group-scale layer itself IS ported — see the next section.)

## Bonsai import: a released ternary checkpoint becomes a trainable model

PrismML's Bonsai releases (e.g. Ternary-Bonsai-27B, Apache 2.0 — ternary weights, one FP16
scale per group of 128) are exactly the visible half of a group-scale counter state. Two
pieces close the loop:

- **`GroupScaleCounterLinear`** ([`group_scale.py`](../src/memory_native_mlx/group_scale.py))
  — the MLX port of the public group-scale layer: per-(row, group) FP scales,
  act-order `perm` support, optional residual homotopy `t + alpha*c/C`, hash-SR update in
  the custom VJP. With one group per row it degenerates to `RMSCounterLinear` **bit-for-bit**
  (tested) — a strict generalization.
- **[`bonsai.py`](../src/memory_native_mlx/bonsai.py)** — format converters:
  `group_counter_from_dense` (an "unpacked" fp checkpoint that is exactly group-ternary;
  verifies ternarity, rejects anything else), `group_counter_from_quantized` (MLX affine
  2-bit tensors, the `-mlx-2bit` builds), and `to_mlx_quantized` /
  `ternary_to_mlx_quant` — the visible weight back into MLX's native grouped quantization
  for `mx.quantized_matmul` inference on the stock optimized kernels.

  Pitfall codified in the API: `mx.quantize` must NOT be used to produce ternary quant
  tensors — it fits the affine grid to each group's min/max, and that 2-bit grid
  {-s, -s/3, +s/3, +s} cannot represent 0, silently corrupting every zero weight. The
  manual construction (q = t+1, scale = s, bias = -s) is exact; `mx.dequantize` and
  `mx.quantized_matmul` reproduce s*t to fp precision (tested to 5e-8).

The layer-level conversion loop is public code: load compatible Bonsai tensors -> counter layer
(c = 0) -> fine-tune as ternary through `nn.value_and_grad` -> export back to the native
2-bit format for native MLX inference. Serving a complete model with mlx-lm still requires
model-level wiring and validation. What remains for a model-level script is walking a
real checkpoint's layer names and wiring the non-linear parts (embeddings, norms) — per
release, deliberately out of the library.

## Interop: covered row-scale layers

`mlx_counter_from_torch` and `export_counter_to_torch` cover row-scale layers and their
packed/unpacked layouts. Do not pass a group-scale solver checkpoint through these
helpers and assume its permutations, salient channel, optimizer semantics or FP tail
have been transferred. The separate Bonsai helpers are layer format converters.

```python
# given an existing PyTorch RMSCounterLinear or PackedRMSCounterLinear row-scale layer

# on the Mac:
from memory_native_mlx.interop import mlx_counter_from_torch
mlx_layer = mlx_counter_from_torch(torch_layer)   # exact codes/scale/v, same packing
```

`mlx_counter_from_torch` / `export_counter_to_torch` go through each side's storage hooks,
so either storage layout (packed/unpacked) on either side works, and the SR stream position
carries over for covered row-scale configurations. Matching codes does not guarantee
identical numerical learning trajectories across hardware or unsupported update options.

## Performance expectations (honest framing)

- The **fused Metal update** is a performance hypothesis: the Triton analogue measured
  ×45.9 on the update / ×1.26 on the step on a T4, and the Metal kernel has the same
  structure. Unverified on Apple GPUs until the Mac gate runs.
- The pure-MLX forward pays the decode-around-GEMM tax (dense fp weight materialized per
  forward), exactly like the torch base path. Metal decode-in-GEMM may fare better than it
  did on the T4 (MLX's quantized matmuls use the same pattern successfully), but that is a
  hypothesis to measure, not a claim.
- Activation memory: the reversible chain references only the chain input, final output
  and parameters in the step graph; with MLX's lazy evaluator the O(1)-in-depth *peak*
  should follow, but peak-memory profiling on a real Mac (`mx.get_peak_memory()`) is an
  open gate, not a result.

## Gates to run on a real Mac (in order)

1. `pip install -e . mlx pytest && python -m pytest tests/test_mlx_port.py -v` — the Metal
   kernel parity test engages; everything else must stay green on the Metal backend.
2. `python scripts/mlx_demo.py` — end-to-end smoke on the GPU.
3. Microbench fused-Metal update vs pure-MLX fallback across layer widths (mirror
   `results/KERNEL.md` methodology).
4. `mx.get_peak_memory()` sweep over reversible depth (mirror `results/POOLS.md`).
5. Build and validate a full-model/group-scale checkpoint bridge, then run recovery on
   a named donor/Mac configuration with strict quality, throughput and peak-memory logs.
