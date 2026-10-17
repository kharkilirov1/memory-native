<p align="center">
  <img src="docs/images/memory-native-banner.svg" alt="Memory Native — compact state, reversible computation" width="100%">
</p>

<p align="center">
  <a href="https://github.com/kharkilirov1/memory-native/actions/workflows/ci.yml"><img src="https://github.com/kharkilirov1/memory-native/actions/workflows/ci.yml/badge.svg" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-8cd6bf" alt="MIT license"></a>
  <a href="docs/status.md"><img src="https://img.shields.io/badge/status-experimental-8ba9e8" alt="Experimental"></a>
</p>

<p align="center">
  <a href="docs/quickstart.md">Get started</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="docs/evidence.md">Benchmarks & status</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

**Memory Native is a PyTorch library for training with compact, finite-state weights.**
Each counter coefficient stores a ternary value and an update accumulator together.
Reversible blocks reconstruct intermediate activations during backward.

| Compact weights | Updates in backward | Reversible blocks |
|---|---|---|
| Four coefficient codes in three bytes | No separate master weights or Adam moments for counter coefficients | Reconstruct branch inputs instead of retaining every intermediate activation |

Six bits describe the coefficient code; scales, statistics, activations and working memory
are additional. The library is experimental. See [measured results and limits](docs/evidence.md).

## Get started

```bash
git clone https://github.com/kharkilirov1/memory-native.git
cd memory-native
python -m pip install -e .
python examples/quickstart.py
```

The offline CPU example learns a synthetic teacher and reports error and storage separately.
For environment setup, CUDA and optional dependencies, follow the [quickstart](docs/quickstart.md).

```python
import torch
from memory_native import PackedRMSCounterLinear

layer = PackedRMSCounterLinear(32, 32, C=11, lr=3e-3)
x, target = torch.randn(64, 32), torch.randn(64, 32)

loss = (layer(x) - target).square().mean()
loss.backward()  # counter coefficients update here

with torch.no_grad():
    prediction = layer(x)
```

Use one forward followed by one backward per training step. Ordinary parameters in a mixed
model still need their own optimizer. Read the [execution contract](docs/ARCHITECTURE.md)
before integrating counters into another model.

## Explore

| Goal | Guide |
|---|---|
| Understand the state and update lifecycle | [Architecture](docs/ARCHITECTURE.md) |
| Run examples | [Examples](examples/README.md) |
| Convert and evaluate a donor | [Workflow tools](scripts/README.md) · [Public notebook](notebooks/README.md) |
| Try experimental update controls | [Counter stability](docs/counter_stability.md) |
| Use the MLX port | [MLX guide](docs/MLX_PORT.md) |
| Reproduce measurements | [Reproducibility](docs/reproducibility.md) |

## Project layout

| Directory | Contents |
|---|---|
| [`src/`](src/) | PyTorch library, CUDA kernels and experimental MLX port |
| [`tests/`](tests/) | Unit and integration tests |
| [`examples/`](examples/) | Small runnable examples |
| [`docs/`](docs/) | User and contributor guides |
| [`scripts/`](scripts/) · [`notebooks/`](notebooks/) | Conversion, evaluation and recovery workflows |
| [`research/`](research/) | Experiments, measurements, papers and historical campaigns |

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and reproducible counterexamples
are welcome. Report vulnerabilities through [SECURITY.md](SECURITY.md).

Created under Kharki’s direction with AI coding tools. See [citation and attribution](CITATION.cff).
Licensed under [MIT](LICENSE).
