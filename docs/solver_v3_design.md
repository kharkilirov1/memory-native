# mn-solver v3 — group PTQ connected to counter recovery

## Root cause fixed

Solver v2 produced a strong group-128 reconstruction, but Stage B converted the donor again with
per-row GPTQ. The best PTQ state therefore never entered the trainable counter model; recovery
actually started from the older ~14k-PPL state.

v3 adds `GroupScaleCounterLinear`, whose visible weight is

```text
W[i,j] = S[i, g_idx[j]] * (t[i,j] + alpha * c[i,j] / C)
```

`S` is one scale per output-row/group, `g_idx` preserves GPTQ act-order, and `(t,c)` is the same
finite-state synapse. At `alpha=0` inference is strictly ternary. At `alpha>0` recovery can use the
residual already stored in the finite-state code; no FP master weight is introduced.

## Solver changes

1. **True alternation.** v2 refit a provisional support before GPTQ. v3 performs a fixed-scale GPTQ
   sweep, refits scales against the support actually produced, then runs GPTQ again. Refinement is
   accepted only when the Hessian reconstruction objective does not increase.
2. **Hessian-weighted refit.** `hdiag` is the stable default, calibration-salience weighting.
   `hessian_cd` adds exact-H coordinate updates for research runs where extra solver cost is
   acceptable.
3. **Trainable group state.** `ptq_warm_start(mode="gptq_group")` imports `(S,t,c,perm)` directly
   into group counter layers instead of collapsing to a row scale.
4. **Recovery recipe.** The v3 runner defaults to `C=11`, counter cosine `0.002 -> 1e-4`, a held
   then cosine-decayed residual homotopy, optional intermediate feature KD, and best-checkpoint
   selection by the geometric mean of domain PPL.

## Required gates

- Unit: exact group/permutation reconstruction; homotopy endpoints; no FP master parameters;
  update changes finite state; refinement is non-worsening under the calibration Hessian.
- 1.5B engagement: warm PPL must match the dense v3 group reconstruction rather than the old
  per-row GPTQ warm PPL.
- Recovery ablation: `{per-row, group-v3} x {alpha=0, homotopy} x {logit KD, +feature KD}` with the
  same data stream and token budget.
- Final claim uses benchmark retention, not PPL alone.

## Run

```bash
PYTHONPATH=src \
PTQ_MODE=gptq_group GROUP=128 C=11 \
COUNTER_LR_START=0.002 COUNTER_LR_END=0.0001 \
FEATURE_KD_ALPHA=0.05 STEPS=6000 \
python scripts/run_ptq_recovery.py
```
