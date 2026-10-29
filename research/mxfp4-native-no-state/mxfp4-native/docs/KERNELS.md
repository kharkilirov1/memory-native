# Kernels: what the ecosystem already gives us, measured on the target GPU

Four kernel-shaped needs:

| need | frequency | who ships it |
|---|---|---|
| MXFP4 matmul (fwd / grad_x) | every step, dominant FLOPs | torch `scaled_mm`, Triton `tl.dot_scaled`, torchao, CUTLASS |
| pack / unpack E2M1 + E8M0 | load, save, reference path | torchao `MXFP4Tensor`, trivial bit ops |
| **the update** (SR on the ladder + 2-bit feedback) | every step, elementwise | **nobody — this is ours** |
| deployment matmul | inference | every MXFP4 runtime |

So the only genuinely new kernel is the updater. Everything around it should be borrowed.

## Measured on the target box

Hardware/toolchain: RTX 5060 Ti, **sm_120**, torch 2.11.0+cu128, Triton 3.9.0.
Probe: [`probes/mx_probe.py`](../probes/mx_probe.py) — 64x64x256 MXFP4 x MXFP4 matmul
through `tl.dot_scaled`, checked against a dequantised reference.

```
gpu: NVIDIA GeForce RTX 5060 Ti (12, 0)
ran: yes | max rel err vs dequantised reference: 0.00e+00
tcgen05 in PTX : False
mma.sync count : 32
e2m1 ops       : 64
```

Read this carefully:

1. **It works and is exact.** `tl.dot_scaled` accepts our layout (`a` packed along K,
   scales `(M, K/32)`, rhs packed and logically transposed) and reproduces the reference
   bit-for-bit. Packing conventions are therefore settled.
2. **It is not native.** No `tcgen05` in the PTX; the compiler emitted plain `mma.sync`
   plus `e2m1` conversions. Triton's own gate is explicit:

   ```python
   def supports_block_scaling():
       return (is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]) or is_hip_cdna4()
   ```

   `sm_120` is compute capability **12.0**, so consumer Blackwell lands on the emulation
   path. Fifth-generation tensor cores with block scaling are `sm_100`/`sm_110`
   (datacenter Blackwell) and CDNA4.

**Consequence for the project:** on this GPU, MXFP4 buys **memory, not throughput**. Any
speed claim must be measured, not assumed, and the honest current statement is "same
speed, less memory" until a native path is proven.

## Reuse verdicts

| component | verdict | notes |
|---|---|---|
| `tl.dot_scaled` | **reuse, with a per-GPU branch** | correct everywhere; native only on cc 10/11. Keep a bf16 fallback for CPU/older GPUs. |
| torchao `mx_formats` | **reuse for packing + reference** | 0.18 installed locally; exposes MX configs and tensor subclasses. Verify its packing matches FORMAT.md byte-for-byte before trusting it. |
| `torch.float4_e2m1fn_x2`, `float8_e8m0fnu` | **reuse for storage** | allocating works; **casting to fp32 asserts** without torchao/triton — do not rely on `.to(torch.float32)` in the reference path. |
| `torch.nn.functional.scaled_mm` | **investigate** | present; block-scaled MX support on sm_120 unverified. First thing to benchmark in Phase 3. |
| cuBLASLt block-scaled GEMM | **investigate** | Triton's tutorial uses `cublas.block_scaled_matmul_nvfp4` only for cc 10/11, but cuBLASLt may expose `sm_120`; a 20-line test settles it. |
| CUTLASS block-scaled kernels | **reference only** | the canonical layout/preshuffle semantics; not worth vendoring. |
| gpt-oss / TensorRT-LLM MXFP4 kernels | **skip** | inference-only, MoE-shaped, CUDA-specific; nothing for the updater. |
| TransformerEngine NVFP4 recipe | **skip for now** | NVFP4 uses a per-16 fp8 scale; we are committed to OCP MX (per-32 E8M0). |

## What we must write

1. **Updater kernel** — one pass over the tile: read the 4-bit value plane, the 2-bit
   residual plane and the scale byte; compute `u`; SR on the ladder; re-quantise the
   residual to 2 bits; write both planes. No dense `grad_w`, no fp32 intermediate tile.
   This is the kernel that makes the memory claim survive the step peak.
2. **Pack/unpack + reference layer** — pure torch, slow, used as the oracle for the kernels.
3. **Accounting + invariant tests** — `persistent_bytes()`, byte-exact layout, and the four
   invariants listed at the end of FORMAT.md.

## What we will not write

- No LUT-GEMM: it exploited ±1 ternary weights (2.7x on single-token decode in the old
  project) and has no meaning on an E2M1 grid.
- No separate decode-to-dense kernel: that is the 4 B/param materialisation that dominates
  the measured peak (3.93 vs 1.21 B/param).
- No int8 path: MXFP4 replaces it.

## Open questions that only measurement answers

| question | test | decision it drives |
|---|---|---|
| native MXFP4 anywhere on sm_120? | `scaled_mm` + cuBLASLt microbenchmark | whether speed is a feature or a datacenter-only note |
| emulated `dot_scaled` vs bf16 dense | 4096x4096x512 timing | whether to use MXFP4 for compute at all on consumer HW |
| does the updater fit in one launch? | Triton implementation vs eager chain | the 3.93 → ~1.2 B/param peak improvement |
| torchao packing compatibility | byte-diff against FORMAT.md | reuse vs own implementation |
