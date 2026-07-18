# memory-native on MLX / Metal — the macOS port

**TL;DR:** the counter-synapse method ports to MLX cleanly and the port lives in
[`src/memory_native_mlx/`](../src/memory_native_mlx/). Same 6-bit state, same packed
0.75 byte/weight layout (bit-identical to the CUDA/engine format), same deterministic
hash-SR update — validated **100% bit-for-bit** against the torch reference on the update
step, with the full pytest gate green on the Linux `mlx[cpu]` backend
([`tests/test_mlx_port.py`](../tests/test_mlx_port.py): 15 passed). The fused Metal kernel
is written (a line-for-line mirror of the T4-verified Triton kernel) and gated behind
`metal_available()`; its parity test engages automatically on an Apple-silicon Mac.

## Why a MacBook is a natural home for this method

The method's whole value proposition is *slashing training-state memory*: no FP master
weights, no Adam moments, 0.75 B/weight persistent state, O(1)-in-depth activations. On
CUDA that buys "1.21B params on a 14.6 GiB T4 at 2.25 GiB peak". On Apple silicon it buys
something arguably more interesting: **fine-tuning on unified memory**. An M-series
MacBook has 16–128 GB of memory shared between CPU and GPU, no PCIe transfer, and a mature
local-ML culture around MLX — but full fine-tuning there is normally killed by exactly the
pools this method attacks (a dense-7B + Adam wants ~84 GB of weights+moments+grads in fp32
before activations). With counter synapses the same 7B body needs ~5.3 GB of persistent
training state. That is "fine-tune on the laptop you own" territory, which is the point.

The fine-tuning entry path already exists on the other branches: the PTQ warm-start
(GPTQ-ternary import) and behavior-recovery pipeline on
the recovery/solver work in this repo produces
counter-format models from pretrained checkpoints. Because the MLX port shares the exact
state encoding and packing, those models cross over losslessly (see *Interop* below): PTQ
on any box with a GPU, recover/fine-tune on the MacBook.

## Design mapping (torch -> MLX)

| PyTorch reference | MLX port | Notes |
|---|---|---|
| `torch.autograd.Function` with in-backward self-update | `mx.custom_function` with a custom VJP that applies the counter transition | grad_w exists only inside the VJP, transiently — same "no full gradient buffer" property |
| `tap` scalar (forces backward to run) | `tap` — a real trainable 0-scalar parameter | `nn.value_and_grad` only differentiates paths reaching a trainable parameter; tap threads every counter layer into the diff set. Gets zero grad; 0 is an AdamW weight-decay fixpoint |
| `register_buffer` (state/scale/v) | public arrays, `freeze()`-d | saved by `save_weights`, evaluated by `mx.eval(model.parameters())`, invisible to optimizers. (Named `codes`, not `state` — `mlx.nn.Module` owns `.state`) |
| `torch.rand` stochastic rounding | **hash-SR always** (the Triton/OpenCL deterministic scheme) | MLX has no global RNG stream; hash-SR makes training reproducible across backends and lets Linux CI validate the port bit-for-bit against torch |
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
  update matches `memory_native.fused_update.counter_update_hashsr` **100% of codes** at
  128x256 (fp-reduction-order SR-boundary flips are permitted by the test but did not occur);
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
- group-scale layers / solver-v3 recovery runner / GLM-MoE harnesses — those live on
  their branches; the interop bridge is how their outputs reach MLX today.
- `scale_rebase="lazy"`, proxy RMS mode, DDP.

## Interop: PTQ/train on CUDA, fine-tune on the MacBook

```python
# on the CUDA box: any memory_native counter layer (incl. PTQ-warm-started)
torch.save(model.state_dict(), "counter_ckpt.pt")

# on the Mac:
from memory_native_mlx.interop import mlx_counter_from_torch
mlx_layer = mlx_counter_from_torch(torch_layer)   # exact codes/scale/v, same packing
```

`mlx_counter_from_torch` / `export_counter_to_torch` go through each side's storage hooks,
so either storage layout (packed/unpacked) on either side works, and the SR stream position
carries over — a handoff continues the same deterministic rounding stream it left.

## Performance expectations (honest framing)

- The **fused Metal update** is where the known win lives: the Triton analogue measured
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
5. The headline: import a PTQ-warm-started 1.5B model (finetune branch) via interop and
   run recovery fine-tuning on a 16 GB MacBook, logging peak unified memory.
