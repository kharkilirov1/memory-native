# Single-pass reversible backward

[Evidence index](README.md) / Single-pass reversible backward

The default inverse path now reuses each branch evaluation for reconstruction and its
local vector-Jacobian product. This removes one recomputation of each branch per backward.
The forward mapping and public constructor signatures are unchanged. Floating-point
reconstruction remains approximate; the change does not introduce quantized activations.

## Correctness

```bash
python -m pytest -q tests/test_reversible.py tests/test_reversible_single_pass.py \
  tests/test_anchors.py tests/test_reversible_gpt.py
```

The focused gate covers:

- Input and parameter gradients against ordinary stored-activation autograd, including
  nonlinear branches, depth greater than one, and frozen parameters.
- Double-precision finite-difference gradcheck for both block and whole-chain paths.
- Packed-counter states, RMS values and scales against a stored-activation update reference.
- One recomputation per branch during backward, including the anchored path.
- Existing model learning and one-final-output storage checks.

Quantized saved activations can consume random numbers in a different order from the
historical implementation. Bit-identical stochastic training trajectories are not claimed.
Higher-order differentiation is explicitly unsupported.

## Validation environment

The full CPU suite on this host reported **504 passed, 22 skipped, 2 failed**. Both
failures were the distributed tests in `test_ddp_decimation.py`: Gloo failed to open a
TCP device with `Operation not permitted`, before the tested update logic executed.
This environment therefore did not validate those distributed cases. The tests were
not weakened or removed. CUDA and MLX hardware/runtime gates were not exercised.

After adding the constant-branch regression cases, the focused command above passed
**27 tests**. Editable installation, wheel construction and the offline quickstart
also completed successfully. These are local checks; GitHub CI is a separate gate.

## CPU timing witness

```bash
PYTHONPATH=src python scripts/benchmark_reversible.py --baseline-ref 851c91e
```

The baseline implementation is loaded from local git history. The candidate is the
installed/available source tree. [Raw JSON](reversible_single_pass_cpu.json) contains its
source SHA-256, baseline commit, runtime, configuration and all timing samples.

| Setting | Value |
|---|---|
| Runtime | Python 3.12.14, PyTorch 2.14.1+cpu, Linux x86_64 |
| Workload | 8 nonlinear coupling blocks, width 128, batch 32, float32 |
| Measurement | Forward + backward; one CPU thread; 5 warmups; 50 samples per implementation |
| Weights and data | Fixed seeded random weights/data; ordinary FP parameters; no optimizer |

| Measurement | Historical inverse path | Single-pass inverse path |
|---|---:|---:|
| Branch evaluations per whole step | 48 | 32 |
| Median step time | 4.131 ms | 3.981 ms |

The observed median-time ratio is **1.04×**. The maximum absolute difference between
the two implementations' collected gradients was approximately `5.25e-06` on this float32
workload. Separate tests compare against stored-activation autograd, rather than treating
agreement with the historical inverse implementation as sufficient proof.

This is one local CPU microbenchmark, not a GPU, donor-recovery or end-to-end LLM result.
Timing is sensitive to hardware and runtime; the raw sample distribution is preserved.
The reduced branch-call count is structural, but wall-time gains are workload-dependent.
Historical CUDA reports have not been rerun for the new backward.

## Prototype ideas intentionally not adopted

- **Factored RMS as a memory saving:** the existing row-only RMS already uses 16 KiB
  for a `4096 × 4096` layer, versus 32 KiB for row + column factors. A quality-focused
  ablation could justify a separate mode later.
- **Universal bitwise inversion by rounding deltas:** finite float precision still imposes
  range limits. No arbitrary-depth exactness claim or grid quantization was imported.
- **Confidence-gated token updates / new hardware kernels:** no matched quality/throughput
  witness justifies replacing the current paths with the disconnected prototype kernels.

See [architecture](../../docs/ARCHITECTURE.md) for the update order and memory boundaries.
