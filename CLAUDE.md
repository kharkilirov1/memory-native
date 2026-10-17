# Memory Native — contributor context

Read [architecture](docs/ARCHITECTURE.md) and [project status](docs/status.md) before changing training behavior.
Current setup and validation commands are in [CONTRIBUTING.md](CONTRIBUTING.md).
The historical session ledger is preserved in [research/notes/DEVELOPMENT_LOG.md](research/notes/DEVELOPMENT_LOG.md); its chronological plans are not the current roadmap.

## Current evidence boundaries

- The public checkout contains the PyTorch CUDA/group-local recovery work and MLX
  group-scale/Bonsai helpers. Branch-era campaign references in the historical log are
  historical; use [research/layout-map.json](research/layout-map.json) for old paths.
- Six bits encode a counter coefficient and accumulator (`C=1..11` for packed layers),
  not all training state. Count FP scales/RMS statistics, salient/perm metadata,
  caches, ordinary FP tail + its optimizer, activations and transient `grad_w` too.
- Qwen 1.5B **v3f2** is EN 34.41/RU 30.34 with mean accuracy retention 70.4%;
  EN 47.4/RU 65.9 were **v3f FP-tail-only**, not v3f2. Later s2i2 EN 30.06/RU
  22.39 has no persisted checkpoint/full public trace and different corpus slices;
  do not rank it against v3f2 or the PTQ sweep without same-slice evaluation.
- Group+dec4's **1.7–3.0×** is an isolated update-kernel speedup, not end-to-end
  training speed. The 12B cached-KD recipe failed; warm state remains better.
- Linux MLX tests are CPU witnesses. Metal parity/performance, Mac peak memory and
  full-model recovery remain untested. Row-scale interop is not a complete solver
  group-checkpoint bridge.
- The 1.21B report lacks its named primary raw log and pinned original runner.
  Older 27B notebooks depend on an external runner ZIP. Use the public workflow in
  `notebooks/cached_kd_public.ipynb` for new runs. See `docs/reproducibility.md` and
  `research/results/ARTIFACT_MANIFEST.md` before claiming exact reproduction.

## Gotchas (hard-won, keep)

- Counter layers are **eager-only**: exactly one forward per backward; wrap measurement
  forwards in `torch.no_grad()` or the reuse guard fires.
- `to_empty()` allocates UNINITIALIZED memory. Rotary `inv_freq` is a COMPUTED buffer
  that no checkpoint stores, so materializing a skeleton block-by-block left RoPE running
  on garbage — finite numbers, silently corrupted calibration. Streaming rebuilds rotary
  from config and `_materialize` raises on any tensor the checkpoint lacks. Caught only
  because the streaming tests demand bit-exact agreement with the in-memory path.
- Confirm which install you are measuring: `python -c "import memory_native; print(memory_native.__file__)"`.
  On 2026-10-17 this box resolved to `<repo>/src/memory_native/__init__.py` and no
  `Desktop\memory-native` checkout existed, so the old "duplicate checkout" warning is
  satisfied here — keep the one-command check when switching checkouts.
- The workspace `.venv` is uv-managed and contains **no pip**: `python -m pip …` fails with
  "No module named pip". Install with `uv pip install --python .venv\Scripts\python.exe …`
  (or run the venv's own console scripts, e.g. `memory-native-charlm.exe`).
- Per-parameter bytes: 6 bits is the *code body* (0.750 B/param). Measured on a deployed
  packed layer: 0.906 B/param persistent, 1.082 B/param resident (int64 salient mirror),
  1.012 B/param for the unpacked reference layer, and ~23 B/param for one pure-PyTorch
  fwd+bwd step. Re-measure with `scripts/state_accounting.py`; see
  `research/results/STATE_ACCOUNTING.md`. Never quote 0.75 as a training-memory figure.
- `TEMP` is a Windows system variable — never use it as an env knob name (`GEN_TEMP` in
  `scripts/infer_counter_cpu.py` exists for exactly that reason).
- Packed kinds need `in_features % 4 == 0`; strict Triton group update needs a
  power-of-two group size (guard rejects others before launch).
- Counter layers are bias-free; the swap preserves donor bias via `CounterLinearWithBias`.
- `weight_flips` does NOT increment on fused/Triton paths — use the decoded-sample
  telemetry (`flip_rate_alt`, `observe_flip_sample`) instead.
- Recovering a full-precision donor's outputs is a NETWORK-level effect (composed layers +
  distillation). A single layer does not self-improve from its own optimum — do not write
  that test.
- Evaluate and select checkpoints at strict ternary `alpha=0`; homotopy (`alpha>0`) PPL is
  a diagnostic, not a deployable number.
- Salient channel: base (t, c) is zero and FROZEN at salient entries; checkpoints with
  salient load into fresh layers (buffers resize on load).
- torch.compile of the update chain: `dynamic=False` only (dynamic emits slower code);
  SR stays eager to preserve the RNG stream.

## Where the code lives

- `src/memory_native/counter.py` — counter layers; `_carry_resolve` is the single source of
  truth for carry/saturation (`packed.py`, `group_scale_packed.py` import it).
- `packed.py` / `fused_update.py` — 6-bit storage and the Triton update kernels;
  `group_scale_packed.py` + `group_scale_kernels.py` — group-128 act-ordered format.
- `donor/` — HF donor swap, the v3 PTQ solver, streaming conversion, asymmetric cascade;
  `recovery/` — KD recovery finetune and resumable runners.
- `glm.py`, `moe_ffn.py`, `reversible.py`, `budget.py` — receiver architecture, MoE,
  reversible blocks and the memory model. `rounding.py` — opt-in named rounding streams.

## Repository conventions

- Keep user guides in `docs/`, runnable examples in `examples/`, and current workflow tools in `scripts/`.
- Put experiment runners, historical notebooks, campaign recipes and evidence in `research/`.
- Preserve raw evidence and report negative results. Historical paths and source hashes describe their original commits; consult `research/layout-map.json` after the layout migration.
- Keep one forward per backward. Do not remove execution guards to make an unsupported workflow appear to work.
- External campaigns retain their donor/cache/preflight requirements; see `research/campaigns/PREFLIGHT_CONTRACT.md` before a launch.
- Run directories remain append-only. Never remove an existing run to make room for a new one.

## This box

On this box the checked-in `.venv` is uv-managed (`uv 0.11`, CPython 3.13, torch 2.12.1+cpu)
and has no pip inside; either activate it and use its `Scripts/*.exe` entry points, or
`uv pip install --python .venv\Scripts\python.exe -e ".[dev,donor]"`. The CPU suite is
green here (2026-10-17); the one ever-failing case was the HF-snapshot symlink test, now
skipped when the host cannot create symlinks.

Recovery runs live in `scripts/` (`run_ptq_recovery.py` is the resumable v3 runner;
env-driven: MODEL, DATA_DIR, CKPT_DIR, STEPS, GRID, SALIENT_FIRST, …).
