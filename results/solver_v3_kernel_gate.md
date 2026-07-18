# Solver v3 packed group-kernel gate

Date: 2026-07-15


## Scope

This gate covers the solver-v3 systems bridge:

- act-ordered 6-bit packed group state;
- per-row/per-group trainable scales;
- residual homotopy without an FP master weight;
- group-aware packed forward and `grad_x` kernels;
- strict update-from-IO with no materialized `[out,in] grad_w`;
- direct `gptq_group` PTQ import into the packed trainable layer;
- recovery-runner selection of the packed strict path.

## CPU/reference validation

Environment used for this gate:

```text
PyTorch 2.10.0+cpu
CUDA unavailable
Triton unavailable
```

Command:

```bash
PYTHONPATH=src pytest -q \
  tests/test_group_scale_packed.py \
  tests/test_ptq_packed_integration.py \
  tests/test_group_scale_kernels_cuda.py
```

Result:

```text
..........sss
10 passed, 3 skipped
```

The skipped tests are the CUDA/Triton parity gates. Python bytecode compilation also passed for
`src/`, the recovery runner, and the group-kernel benchmark.

The CPU gates pin:

- exact act-order pack/unpack and reconstruction;
- homotopy endpoints without changing packed truth;
- deterministic from-IO update equality to the explicit group-gradient path;
- `grad_x` computed from the pre-update weight;
- no trainable/master `nn.Parameter` in the packed counter layer;
- finite-state learning on a synthetic teacher;
- partial final groups;
- direct PTQ import preserving the solver reconstruction;
- sub-byte packed state plus only group/row metadata.

## Strict-update memory witness

The strict GPU wrapper allocates only

```text
3 * out * ceil(in/group) + out
```

float32 scratch values: old group scales, group-scale gradients, group `sum(g^2)` partials, and one
RMS denominator per output row.

At group 128:

| Qwen2.5-1.5B FFN weight `[out,in]` | strict scratch | dense FP32 `grad_w` | reduction |
|---|---:|---:|---:|
| `[8960,1536]` | 1.265 MiB | 52.500 MiB | 41.5x |
| `[1536,8960]` | 1.236 MiB | 52.500 MiB | 42.5x |

This is a symbolic/allocation witness. A CUDA peak-memory measurement is included in
`scripts/benchmark_group_kernels.py` but was not executable in this CPU-only environment.

## CUDA gates still required

```bash
pytest -q tests/test_group_scale_kernels_cuda.py

python scripts/benchmark_group_kernels.py \
  --in-features 1536 --out-features 8960 --tokens 512 --group 128 --dtype bf16
python scripts/benchmark_group_kernels.py \
  --in-features 8960 --out-features 1536 --tokens 512 --group 128 --dtype bf16
```

The CUDA tests check FP32/BF16 forward and `grad_x` parity plus bit-quantified equality of the
strict update against the deterministic reference.

## Verdict

The representation and CPU/reference path are closed and tested. The Triton kernels are implemented
and wired into the real recovery runner, but are not claimed hardware-valid until the CUDA gates run.
The PR remains draft for that reason. Strict distributed recovery is also intentionally rejected for
now rather than silently allowing counter replicas to diverge.
