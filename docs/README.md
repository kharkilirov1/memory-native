# Documentation

[Home](../README.md) / Documentation

Start with a runnable example, then follow the implementation or an empirical result.
Historical designs describe the options explored at the time; use the current status
and executable code to decide what is supported today.

## Getting started

| Guide | What you will learn |
|---|---|
| [Quickstart](../examples/quickstart.py) | Run an offline synthetic learning example |
| [Architecture](ARCHITECTURE.md) | State layout, self-updates, reversible backward and memory boundaries |
| [Project status](../PROJECT_STATUS.md) | Verified, experimental and open capabilities |
| [Reproducibility](../REPRODUCIBILITY.md) | Environments, commands and evidence requirements |
| [Results index](../results/README.md) | Measurements, limitations and negative results |
| [Artifact manifest](../results/ARTIFACT_MANIFEST.md) | Public artifacts and missing historical evidence |

## Implementation paths

| Path | Read next |
|---|---|
| Counter layers | [`counter.py`](../src/memory_native/counter.py), [`packed.py`](../src/memory_native/packed.py) |
| Experimental counter stability | [RMS startup, cumulative rebase and named rounding streams](counter_stability.md) |
| Reversible execution | [`reversible.py`](../src/memory_native/reversible.py), [single-pass witness](../results/REVERSIBLE_SINGLE_PASS.md) |
| Donor conversion and calibration | [Consolidated solver guide](solver_v3_consolidated.md) |
| Conversion/cache/recovery workflow | [Public notebook](../notebooks/cached_kd_public.ipynb) |
| MLX and interchange | [MLX port](MLX_PORT.md) |
| Campaign preflight | [Evidence contract](../production/PREFLIGHT_CONTRACT.md) |

## Research directions

- [Preprint](../paper/MEMORY_NATIVE_PREPRINT.md): method, assumptions and open questions.
- [Project article](ARTICLE.md): research narrative and saturation hypothesis; historical
  claims should be read alongside the current evidence index.
- [Acceleration investigations](MEMORY_NATIVE_ACCELERATION_DEEP_ITERATIONS_V2.md): design
  space and historical proposals, not a list of validated speedups.
- [Forward/backward pilots](../results/new_math/INTERPRETATION.md): small CPU experiments
  with structured factors and alternative arithmetic.
- [Qwen recovery design](qwen_counter_recovery_design.md), [solver v2](solver_v2_design.md)
  and [solver v3](solver_v3_design.md): background for the current solver.

## Contribute

[Development workflow](../CONTRIBUTING.md) · [Security](../SECURITY.md) ·
[Citation](../CITATION.cff) · [License](../LICENSE)
