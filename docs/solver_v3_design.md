# mn-solver v3 — group PTQ connected to packed counter recovery

## Root cause fixed

Solver v2 produced a strong group-128 reconstruction, but Stage B converted the donor again with
per-row GPTQ. The best PTQ state therefore never entered the trainable counter model; recovery
actually started from the older ~14k-PPL state.

v3 keeps the exact solver state:

```text
W[i,j] = S[i, g(j)] * (t[i,j] + alpha * c[i,j] / C)
```

`S` is one scale per output-row/group, `(t,c)` is the finite-state synapse, and `alpha -> 0`
anneals the residual homotopy to strict ternary inference without an FP master weight.

## Solver changes

1. **True alternation.** Fixed-scale GPTQ sweep -> scale refit against the achieved ternary support
   -> fresh GPTQ sweep. A refinement is accepted only when the calibration-Hessian objective does
   not increase.
2. **Hessian-weighted refit.** `hdiag` is the stable default. `hessian_cd` adds exact-H coordinate
   updates for smaller/research runs.
3. **Direct trainable import.** `ptq_warm_start(mode="gptq_group", kind="counter_packed")` imports
   `(S,t,c,perm)` directly instead of collapsing group scales back to one row scale.
4. **Recovery recipe.** `C=11`, counter cosine `0.002 -> 1e-4`, residual homotopy, optional feature
   KD, and best-checkpoint selection by geometric-mean domain PPL.

## Packed kernel architecture

`PackedGroupScaleCounterLinear` stores 4 finite-state codes in 3 bytes. Unlike the torch reference,
its state is stored in **act-order**:

```text
state position p  -> original input column perm[p]
scale group       -> p // group_size
```

This makes every group a contiguous range in the packed state. It is the key that permits parallel
state writes without two groups racing on different lanes of the same 3-byte pack.

### Forward

The Triton forward gathers `x[:, perm[p]]`, decodes `(t,c)` from packed state in registers, loads
`S[row, p//group]`, and accumulates `x @ W^T`. No dense W is materialized. FP32, FP16 and BF16
inputs are supported.

### grad_x

The backward input-gradient kernel decodes the same packed group weight and computes
`grad_out @ W`, scattering the result back to original columns through `perm`.

### Strict update-from-IO

The update never creates `[out,in] grad_w`. It uses three bounded launches:

1. `(row, group)` correlation/statistics: form one group of `grad_w` in registers, emit only
   group-scale gradient and group `sum(g²)`;
2. one row program: reduce group statistics, update RMS and all group scales;
3. `(row, group)` state transition: recompute that group's correlation, apply deterministic hash-SR,
   carry/remainder, and repack the 6-bit state.

The correlation is intentionally recomputed in stage 3. This spends FLOPs to keep scratch at

```text
3 * out * n_groups + out   float32 values
```

instead of `out * in`.

For the two dominant Qwen2.5-1.5B FFN orientations at group 128:

| weight shape `[out,in]` | strict scratch | dense fp32 `grad_w` |
|---|---:|---:|
| `[8960,1536]` | ~1.27 MiB | ~52.5 MiB |
| `[1536,8960]` | ~1.24 MiB | ~52.5 MiB |

Training scales remain FP32: packed state is 6 bits/weight plus 32/group bits/weight (0.25 bit at
group 128), row RMS, and negligible permutation metadata. Checkpoint-only FP16 scale compression is
possible later but is not claimed here.

## Honest status

- CPU/torch reference: exact act-order packing, homotopy, deterministic update, partial last group,
  no master parameters, and PTQ import are tested.
- CUDA tests are included for forward, grad_x and bit-quantified strict-update parity.
- This execution environment has no CUDA/Triton, so the new GPU kernels are **implemented but not
  yet hardware-validated**. Run the benchmark/gate below before merging the PR as production-ready.
- Strict update is currently single-rank. It raises under initialized multi-rank distributed mode
  rather than silently diverging; a groupwise correlation all-reduce is a separate systems task.
- Decode-in-GEMM and strict update are memory-first kernels. They may lose to cuBLAS on throughput.

## Gates

```bash
pytest -q tests/test_group_scale_packed.py tests/test_ptq_packed_integration.py
pytest -q tests/test_group_scale_kernels_cuda.py   # CUDA + Triton

python scripts/benchmark_group_kernels.py \
  --in-features 1536 --out-features 8960 --tokens 512 --group 128 --dtype bf16
python scripts/benchmark_group_kernels.py \
  --in-features 8960 --out-features 1536 --tokens 512 --group 128 --dtype bf16
```

The 1.5B engagement gate remains: `[v3 warm]` must match the dense group-v3 reconstruction rather
than the old per-row GPTQ warm start. The final 90% claim must use benchmark retention, not PPL alone.

## Recovery run

```bash
PYTHONPATH=src \
PTQ_MODE=gptq_group COUNTER_KIND=counter_packed \
GROUP_KERNEL_MODE=auto STRICT_UPDATE=1 \
GROUP=128 C=11 \
COUNTER_LR_START=0.002 COUNTER_LR_END=0.0001 \
FEATURE_KD_ALPHA=0.05 STEPS=6000 \
python scripts/run_ptq_recovery.py
```
