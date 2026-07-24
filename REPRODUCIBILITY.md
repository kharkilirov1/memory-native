# Reproducibility

This document is the shortest path from a fresh checkout to a trustworthy witness.

## 1. CPU correctness gate

```bash
git clone https://github.com/kharkilirov1/memory-native.git
cd memory-native
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

CUDA-, Triton-, distributed-, and MLX-specific tests may skip when their runtime or hardware is
not present. A skip is not a hardware witness.

## 2. Focused solver/export regression gate

The donor recovery work changes faster than the core counter package. Run its focused gate:

```bash
python -m pytest -q \
  tests/test_solver_v3_consolidated.py \
  tests/test_asym_calibration.py \
  tests/test_guided_hessians.py \
  tests/test_export_motifcl.py
```

## 3. Small training smoke

```bash
memory-native-charlm --config micro --steps 60 --device cpu
```

This checks execution and basic learning behavior. It does not reproduce the CUDA memory or
throughput claims.

## 4. CUDA comparison and memory gate

```bash
memory-native-charlm --config tiny --steps 600 --device cuda
memory-native-memgate --config s512 --optimizers adamw,galore,lomo,bnb8 --device cuda
```

Record:

- `git rev-parse HEAD`;
- GPU model and VRAM;
- driver, CUDA, PyTorch, Triton, and Python versions;
- exact command and environment variables;
- seed, dataset identity, and dataset revision;
- stdout/stderr and generated JSON/Markdown artifacts.

Do not compare peak-memory values collected under different process lifecycles or allocator
states without saying so.

## 5. Scale witnesses

The committed scale reports are historical evidence, not a guarantee for another checkout:

- `results/SCALE_1B.md`
- `results/SHOOTOUT.md`
- `results/POOLS.md`
- `results/KERNEL.md`

Use the scripts referenced inside each report. Preserve raw logs and add a short Markdown report
that distinguishes:

1. directly measured values;
2. derived values;
3. modeled estimates;
4. forecasts;
5. skipped or failed gates.

## 6. Reporting a reproduction

Open a reproduction-report issue and include:

- commit SHA;
- clean or dirty working-tree state;
- hardware/software matrix;
- exact command;
- expected and observed values;
- raw log or a minimally processed artifact;
- whether the result confirms, partially confirms, or contradicts the original claim.

Negative results are first-class evidence.
