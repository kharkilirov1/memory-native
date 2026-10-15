# Counter stability: CPU mechanism checks and ablations

[Results index](README.md) / [Usage and compatibility](../../docs/counter_stability.md)

Three opt-in controls are implemented: startup bias correction for row RMS,
cumulative threshold rebasing, and stable named stochastic-rounding streams.
Default training behavior is unchanged. These results do not justify enabling
the new dynamics by default.

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_counter_stability.py \
  --steps 300 --seeds 5 --output research/results/counter_stability_cpu.json
python -m pytest -q tests/test_counter_stability.py tests/test_rounding_streams.py \
  tests/test_lagged_rms.py tests/test_packed.py tests/test_latent_dynamics.py
```

[Raw JSON](counter_stability_cpu.json) contains all 40 runs, settings, runtime and
source hashes. Python 3.12.14, PyTorch 2.14.1+cpu, one CPU thread. No CUDA execution,
peak-memory measurement or wall-time speedup is claimed.

## Synthetic dense-donor recovery

Each seed generates a dense 32x32 Gaussian teacher, 512 training inputs and 256
held-out inputs. The student is initialized by the existing ternary donor conversion
with residual-seeded counters. Every variant uses the same named rounding stream,
learning rates (`lr=0.002`, `lr_scale=0.0002`), data and 300-step budget. No learning
rate tuning was performed. Evaluation uses the strict ternary forward.

Mean initial held-out MSE is **0.186455**. All configurations finish worse than
that initial conversion. This witness exposes a failure of this dense-donor recovery
recipe; it is not a successful model recovery result.

| RMS beta | Correction | Rebase | Final MSE mean | Seed range |
|---|---|---|---:|---:|
| 0.9 | Off | Eager | 0.284100 | 0.251875–0.306736 |
| 0.9 | On | Eager | 0.284112 | 0.253247–0.302877 |
| 0.9 | Off | Threshold | 0.284100 | 0.251875–0.306736 |
| 0.9 | On | Threshold | 0.284112 | 0.253247–0.302877 |
| 0.99 | Off | Eager | 0.291085 | 0.260074–0.311174 |
| 0.99 | On | Eager | 0.275232 | 0.237161–0.305760 |
| 0.99 | Off | Threshold | 0.291085 | 0.260074–0.311174 |
| 0.99 | On | Threshold | 0.275232 | 0.237161–0.305760 |

At beta=0.99, correction improves final MSE in all five paired seeds, by about
5.45% in the mean relative to uncorrected training. At the default beta=0.9 the
mean effect is essentially zero, with mixed paired outcomes. Neither establishes
an advantage over the initial donor quantization. First-step visible MSE is also
unchanged in this setup: a smaller counter tick need not immediately change a
ternary forward weight. The threshold variants give the same final MSE as eager
here; no end-to-end quality advantage was observed.

## Isolated rebase and stream checks

The scale-only probe uses 64 rows of 64 counters. One driver synapse per row moves
the shared scale; the other 63 receive zero gradient. Counter learning rate is zero
to isolate rebase noise. Residuals start at 5, C=11, and the scale alternates between
0.998 and 1.002 for 1,000 updates.

| Mechanism | Observation |
|---|---|
| Eager rebase | 100% of dormant counters change at least once in this deliberately repetitive probe |
| Threshold rebase | 0% change; no rebases; maximum scale-basis mismatch about 0.2004% |
| Cumulative drift, 0.2% per step | Threshold mode rebases 41 times; final scale 7.374312; maximum relative basis mismatch 4.4914%, below 1/(2C)=4.5455% |
| Two named streams, 65,536 half-integer samples | Round-up fractions 49.9878% and 50.1450%; draws differ at 49.7360% of positions |

These probes verify mechanisms, not a universal zero-noise guarantee or statistical
independence. Carry pressure also triggers rebasing. A deliberately cumulative scale
change should move residual counters; suppressing all those movements would be wrong.

## Regression checks

- The focused command above passed **56 tests**, including EMA/history equations
  for beta 0, 0.9, 0.99 and 0.999999; raw-gradient scale learning; actual-update counts
  under tiling/decimation; import reset; checkpoint resume; cumulative drift and
  carry handling; group-local decimation and independent replica-local global RNG use.
- The full local CPU suite passed **533 tests**, skipped **22**, and failed **2**.
  Both failures were existing Gloo DDP tests, blocked at TCP initialization by
  `Operation not permitted` before update logic executed. They were not weakened.
- A subsequently added real two-process DDP test checks named streams together with
  corrected proxy RMS, tiling and threshold rebase while global RNGs differ by rank.
  Its local run is blocked by the same Gloo socket restriction. CI must validate it.
- Packed-kernel seed scheduling is exercised with the CPU hash-SR kernel reference.
  This validates the wrapper and checkpoint logic, not generated CUDA kernel code.

## Costs and boundaries

Bias correction adds 8 bytes per output row for update counts (32 KiB at 4096 rows).
Threshold mode reuses the existing calibration-scale buffer. Named streams add two
Python integer fields and 16 bytes of tensor payload per configured layer in a
checkpoint. Neither feature creates a dense second-moment state.

The new row dynamics use PyTorch updates instead of the old fused/compiled fast path.
Their transient tensors are tile-bounded, but they are not allocation-free. Named
streams can feed existing packed/group kernels without changing their math. Exact
replay still requires the same backend, tiling, update schedule and other stochastic
inputs. See the usage guide for checkpoint and constructor requirements.

Rank1 RMS, leaky readout, int64 reversible activations and token compaction were not
added: this change does not provide quality/throughput evidence for those proposals.
