# Quickstart

[Documentation](README.md) / Quickstart

## Install

Use Python 3.10 or 3.12 for the environments exercised by CI.

```bash
git clone https://github.com/kharkilirov1/memory-native.git
cd memory-native
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Run the offline example

```bash
python examples/quickstart.py
```

This small CPU demo learns a synthetic ternary teacher. Its JSON output separates
initial/final error, packed coefficient bytes and total layer-buffer bytes. It requires
no account, dataset download or model checkpoint. It is not an LLM benchmark.

With a working CUDA installation:

```bash
python examples/quickstart.py --device cuda
```

## Train a layer

```python
import torch
from memory_native import PackedRMSCounterLinear

layer = PackedRMSCounterLinear(32, 32, C=11, lr=3e-3).train()
x = torch.randn(64, 32)
target = torch.randn(64, 32)

for _ in range(100):
    loss = (layer(x) - target).square().mean()
    loss.backward()

with torch.no_grad():
    prediction = layer(x)
```

Counter coefficients update during `backward()` and need no separate optimizer.
Embeddings, norms and other ordinary parameters in a mixed model still need one.
Packed layers require an input width divisible by four.

Use one forward followed by one backward. Shared counter weights and ordinary gradient
accumulation need an explicit scheduler. Reversible branches must be deterministic;
see the full [execution contract](ARCHITECTURE.md).

## Optional environments

| Task | Install |
|---|---|
| Core tests | `python -m pip install -e ".[dev]"` |
| Donor conversion and full CPU tests | `python -m pip install -e ".[dev,donor,campaign]"` |
| Research figures | `python -m pip install -e ".[research]"` |
| Apple-silicon MLX examples | `python -m pip install -e ".[mlx]"` |

Run `python -m pytest -q` after installing the full test dependencies. Hardware-specific
checks may skip. See [reproducibility](reproducibility.md) for full environment details.

Continue with [architecture](ARCHITECTURE.md), [workflow tools](../scripts/README.md),
or the [evidence overview](evidence.md).
