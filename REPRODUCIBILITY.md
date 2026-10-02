# Reproducibility

This document is the shortest path from a fresh checkout to a trustworthy witness.

The executable CPU gates below are public. Historical run reports have varying artifact
completeness; a command or seeded pipeline alone does not prove an exact run is reproducible.
The missing artifacts listed below must be recovered from their original source or replaced
by a clearly labeled new run. Never reconstruct a raw log from its Markdown summary.
The [artifact manifest](results/ARTIFACT_MANIFEST.md) lists public paths and external
or lost evidence without treating an external version name as a verified download.

The [public cached-KD notebook](notebooks/cached_kd_public.ipynb) exercises conversion,
teacher caching, recovery, selection and selected-artifact reload without an external
ZIP. Its default synthetic CPU smoke tests plumbing, not model quality. New conversion
and cache manifests use schema v2 and verify content hashes. Historical streamed states
can still be loaded for inference with a warning; verified resume requires a fresh
conversion. Legacy teacher caches must be rebuilt. The corpus builder now records
tokenizer semantics instead of relying on directory names.

`scripts/kd_cached_recovery.py` exports `restore_selected_artifact(path, device="cpu")`.
Pass `selected_artifact.json`; the helper verifies the donor, warm conversion and selected
checkpoint, then overlays a slim KD checkpoint when one was accepted. Keep those base
artifacts: `best.pt` alone is not a full model or a training-resume checkpoint. The
notebook checks that the reloaded model reproduces its selected validation metric.

The archived external campaign's fail-closed checker and added evidence contract are
documented in [production/PREFLIGHT_CONTRACT.md](production/PREFLIGHT_CONTRACT.md).

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
python -m pip install -e ".[dev,donor,campaign]"
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

### Known evidence and workflow gaps

| Item | Public evidence | Missing or external requirement |
|---|---|---|
| 1.21B T4 scale run | `results/SCALE_1B.md` historical summary; later scripts use similar body dimensions | Named primary `gpu_scale_1B_T4.log` is absent; exact original runner/environment are not pinned by the report |
| Qwen v3f/v3f2 recovery | `results/recovery_15b_v3_final.md` tables and incident ledger | Checkpoints/corpus/full log were reported on the author's Drive, not committed public artifacts |
| Later s2i2 6000-step recovery | `results/solver_v3_salient_scope_asym_gate_colab.md` recorded EN/RU and aggregate | Checkpoint was not persisted; complete per-domain digits/raw metrics and notebook output are not public |
| Historical 27B notebook family | Archived notebook recipes | External runner ZIP is absent from the repository; those notebooks are not a self-contained public runner |
| MLX/Metal on a Mac | Linux CPU tests and Metal kernel source | Apple-silicon kernel parity, throughput and peak-memory run still needed |
| 12B cached KD | `results/GEMMA12B_CACHED_KD.md` negative-result tables | Original Kaggle dataset/kernel outputs are external artifacts; pin versions and verify access before claiming reproduction |

For a public conversion/cache/recovery workflow, use
[`notebooks/cached_kd_public.ipynb`](notebooks/cached_kd_public.ipynb) and the scripts it
calls, supplying the donor and corpus explicitly. This removes the private runner-bundle
dependency; it does not reproduce historical 27B metrics or validate large-model quality.

New streaming conversions use manifest schema 2: donor/config/shard/tokenizer hashes,
exact calibration token IDs and batch boundaries, solver/runtime options and per-block
hashes. Resume rejects changed/corrupt/legacy manifests. Complete legacy outputs may
still be loaded for inference with a warning, but need a new conversion to establish
verified provenance for resume/recovery. Preserve the manifest with each conversion.

Compare quality only on identical tokenizer, corpus revision, held-out token slice, context,
stride and metric. Inference bit rates are not trainable-state bit rates, and update-kernel
speedups are not whole-step speedups. Record actual peak memory for the complete training path.

### Reporting format

Open a reproduction-report issue and include:

- commit SHA;
- clean or dirty working-tree state;
- hardware/software matrix;
- exact command;
- expected and observed values;
- raw log or a minimally processed artifact;
- whether the result confirms, partially confirms, or contradicts the original claim.

Negative results are first-class evidence.
