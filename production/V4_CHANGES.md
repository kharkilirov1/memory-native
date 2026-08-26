# V4 (recovery-3k) vs archived v3: knob mapping & runner gaps

Status legend: **[v3-ok]** expressible with existing strict-v3 env knobs ·
**[PATCH]** requires a `kd_cached_strict_v3.py` runner-side change.

The canonical config is `production/qwen38_27b_recovery_3k.yaml`.
The notebook is `notebooks/MN_Qwen38_27B_Recovery3k_V4_RTXPRO6000_H100.ipynb`.

## 1. Knob mapping

| Recipe item | Status | How it runs today |
|---|---|---|
| steps=3000, batch=2, seq_len=512 | [v3-ok] | `STEPS/BATCH/SEQ` env |
| bf16 | [v3-ok] | `DTYPE=bf16` |
| alpha=0, decimation=1 | [v3-ok] | strict v3 default (`DECIMATION=1`) |
| kd/ce/T = 1.0, grad clip 0.5 | [v3-ok] | unchanged env |
| counter LR 1.25e-4 → 1.0e-5 | [v3-ok]¹ | `COUNTER_LR_START/END` |
| scale LR 2.5e-5 → 5e-6 | [v3-ok]¹² | `SCALE_LR_START`/`SCALE_LR_END`² |
| warmup 5% + piecewise/cosine decay (150→2000→3000) | **[PATCH]** | runner interpolates START→END linearly over STEPS; no warmup hook, no multi-phase schedule |
| grad_accum = 8 (effective batch 16) | **⚠ CONFLICT** | see §2; do NOT enable without runner support |
| GRAD_CKPT auto probe | [v3-ok]³ | stage P decides 0/1 whole-model toggle only |
| selective checkpointing (top blocks off) | **[PATCH]** | runner has whole-model toggle only |
| teacher cache reuse, K=1024 | [v3-ok] | v4 makes it RESTORE-ONLY: missing/mismatched cache → hard stop |
| full eval every 500 @24k tokens | [v3-ok] | `EVAL_EVERY=500`, `EVAL_MAX_TOKENS=24000` |
| proxy eval every 100 (~2–4k tok/domain) | **[PATCH]** | no proxy-eval mode exists in the runner |
| periodic full checkpoint every 500 | **[PATCH]** | runner persists only gate-authoritative `best.pt` |
| resume state every 100–200 | **[PATCH]** | no crash-resume state in strict-v3 runner |
| never delete run dirs | [v3-ok] | enforced by v4 notebook layout (`run_<ts>/`, append-only sync) |
| early stop patience 3 / min improvement 0.001 | [v3-ok] | `EARLY_STOP_PATIENCE/MIN_IMPROVEMENT` |
| donor Qwen3.8 verification | new (repo) | `scripts/check_donor_config.py` + Stage D gate |

¹ Runner's exact LR interpolation shape (linear vs cosine) is confirmed for linear
   START→END; cosine is [PATCH].
² `SCALE_LR_END` support is assumed present-if-harmless: if the runner ignores unknown
   env names, scale LR stays flat at START (2.5e-5) — verify once in the Stage P log.
³ Stage P measures real peak VRAM via nvidia-smi polling and writes `metrics/gck_decision.json`.

## 2. ⚠ The grad_accum conflict — read before touching BATCH knobs

Strict KD has a documented invariant (CLAUDE.md gotchas; v3 notebook doctrine):
**counter layers mutate state inside every backward pass.** Calling N backward passes
before one optimizer step therefore does NOT mean ordinary gradient accumulation:
updates n+1..N would run against an already-mutated automaton state with stale scales.

So `grad_accum: 8` from the recipe is valid ONLY IF the runner implements a safe
accumulation path (e.g., queue quanta deltas and apply once per optimizer step). Until
that patch exists and is unit-gated:

- keep `micro_batch=2`, **one update per micro-batch**;
- effective-batch arithmetic (2×8=16) must NOT be quoted in results;
- if more sequence throughput is needed, prefer raising `SEQ` within probe-proven VRAM,
  not accumulation.

## 3. Minimal runner patch sketch (for when we do it)

```
lr(step):        piecewise + warmup per yaml lr_schedule block
proxy_eval(step):if step % EVAL_PROXY_EVERY == 0: domains[:6] at PROXY_MAX_TOKENS,
                 record to metrics.json under "proxy"; NEVER feeds selection/gates
ckpt(step):      if step % CKPT_FULL_EVERY == 0: stream slim full ckpt to checkpoints/
resume(step):    if step % CKPT_RESUME_EVERY == 0: atomic small state (step, opt_pos,
                 rng, last metrics) -> resume_state.pt (CKPT_TMP + rename)
safe_accum:      REJECTED unless a unit test proves byte-equality of a single backward
                 of batch 2x512 vs accumulated 1x512 twice, on quanta AND scales
```

Everything else in this document runs as-is on the current v3 release.
