# Adaptive carry threshold: normalise after the integrator, not before it

Design note, 2026-10-28, branch `mxfp4-native`. Paper only; §6's flags are the future implementation
surface. Context: `docs/PHASE2.md` "Round 2", `docs/UPDATE.md`, `docs/LITERATURE.md` §5.

## 1. The measurement this has to explain

| arm (char-LM, 800 steps, lr 0.01) | bits/char | rail (`|idx|=7`) | mean `|acc|` |
|---|---:|---:|---:|
| per-row RMS + 9-state integrator (base) | **2.7707** | 2.0% | 1.76 |
| second moment per 32-block, E8M0 / fp8 | 3.5965 / 3.5842 | 11.7% | 0.71 |
| Adafactor rank-1 (row×column, fp8) | 3.5830 | 11.7% | 0.87 |
| cautious tick gating (0 b/param) | 3.0110 | — | — |

All four put adaptivity in the **tick** (or gated the tick) and lost by 0.25-0.83 bits/char; the
losing arms rail. That is the diagnosis.

## 2. Mechanism: a raw tick carries magnitude, a normalised tick carries only sign

Base tick `t = −lr·(g/√v_row)/d·C` carries when `|t|` reaches `C`, at rate `lr·|g|/(√v_row·d)`. Inside
a row the ordering of `|g|` survives: a coordinate 10× below the row RMS makes 10× smaller ticks and
carries rarely. With block/column `v` it is compared to its own scale, `g/√v ≈ 1`, the quantiser hands
out ±1 ticks, and carried distance stops depending on gradient size — the update degenerates to
`sign(g)` (signSGD/Signum [2]), discarding the magnitude that makes quiet coordinates harmless. A
unit-step walk reaches ±6 in ~5 ladder steps (top gaps `2s`): 11.7% at the rail, where `blocked` pins
`acc` at `C−1` (the `mean|acc|` collapse). Adam divides the **momentum**, not the gradient, by `√v` [1].

## 3. The rule

Units as `mx6/update.py`: `d_e = MIN_GAP[|code_e|]·s ∈ [0.5s, 2s]`; `C0 = 5`, `Cmax = 7`.

```
v_i    ← beta*v_i + (1-beta)*mean_e g_ie²              # per row, beta = 0.9
g_ref  = sqrt(mean_i v_i)                              # one scalar/layer, frozen at step 1
C_i    = clamp(round(C0 * sqrt(v_i)/g_ref), 1, Cmax)   # per-row PATIENCE (integer)
tau_e  = -lr * g_e / (d_e * g_ref) * C0                # RAW tick: v does not appear
acc_e  ← acc_e + tau_e ;  k_e = trunc(acc_e/C_i)       # integrate raw gradients, then carry
acc_e  ← acc_e - k_e*C_i ;  idx_e = clamp(idx_e + k_e, -7, 7)   # one carry = one E2M1 step
# 4-bit v: code 8 == g_ref, 4 codes/octave, SR on the code so sub-code moves do not stall
L_i = log2(v_i/g_ref²); ell_i = (1-beta)*L_i + beta*(code_i-8)/2
code_i = clamp(SR(2*ell_i + 8), 0, 15); C_i = clamp(round(C0*2^((code_i-8)/4)), 1, 7)
# 4-bit acc (15 states −7..7): cc = SR(acc_e + tau_e); k_e = trunc(cc/C_i); acc_e = cc − k_e*C_i
# |acc_e| < C_i <= 7 always fits; if idx_e was clipped: acc_e = sign(cc)*(C_i−1)  (hold, not dither)
```

Init `v_i(1) = mean_e g_ie²`, `g_ref = sqrt(mean_i v_i(1))` → `C_i = C0` on step 1, no bias
correction. `g_ref` is one E8M0 byte/layer; `acc` reuses today's 4-bit integrator word.

## 4. Why this is Adam-shaped, and where the rails went

With per-row `v` this is *the same update in expectation* as the base (carry rate `lr|g|/(√v_i·d)` both
sides), so it cannot by itself close the 0.079 gap; it is a platform that makes finer `v` safe at 4 bits
instead of 32. With finer `v`: (i) `v` never multiplies `tau`, so no coordinate gets unit-tick `sign(g)`
updates; (ii) patience clamps to `[1,7]`, bounding rate amplification to `C0/C_i ∈ [5/7, 5]` instead of
unbounded `1/√v` (a column 16× below reference got 16×+ in the tick arms, 5× here); (iii) the carry is
exactly one ladder step and the remainder conserves pressure, so distance moved is driven by accumulated
raw gradient, never a per-step rescale; (iv) at the rail, blocked carries hold `C_i−1` pressure *against*
further motion. Predicted rail: ~2% per-row `v`, 3-6% per-column — not 11.7%; T3 (§6) is the falsifier.

## 5. Bits

State is E2M1 + E8M0-scale (4.25) + 4-bit accumulator (4) + 4-bit `v_i` per row (`4/n_in` =
**0.0234**; 2304 rows, N = 393,216) + one `g_ref` byte per layer. Total **8.273 b/param**, versus
8.437 for the measured base with fp32 rows: 0.16 b/param *cheaper*. Per-column variant adds
`4/n_out` ≈ 0.018. Both fit the ≤1 extra bit/param budget; the leak costs 0 (constant multiply).

## 6. Failure modes, logs, leak, test plan

| mode | signature | log (every 50-100 steps) | action |
|---|---|---|---|
| limit cycle | carry sign flips at lag 1-4, net `|Δidx| ≤ 1` | `cycle_frac` (≥3 reversals/10 steps, net ≤1); acc histogram | enable leak |
| threshold jitter | code crosses an octave boundary → `C_i` jumps 19% | code flips/row/100 steps; `C_i` histogram; frac at 1 and 7 | hysteresis, or 3-bit `v` |
| rail / clamp | blocked carries hold pressure; weight stuck at ±6 | `rail_frac`, `blocked_frac`, time-to-rail, `p99(|tau|/C_i)`, max `|cc|` | cap `|k| ≤ 2` or `tau ≤ 2·Cmax`; reset acc at rail |
| SR/trunc interaction | pressure conserved in expectation, `trunc` nonlinear | `Σ|k|·d` vs `Σ|lr·g|` over 500 steps (must → 1.0); pressure balance, not per-step bias | none if ratio ≈ 1 |
| 4-bit EMA stall | code frozen while `mean g²` moved | frac rows with unchanged code over 200 steps | SR (already in code), else reset `v` |
| adaptivity collapse | all `C_i` equal or at bounds | `C_i` entropy; frac `C_i = 5` | widen range or 5-bit `v` |
| lr shift | raw rule rescales lr by `g_ref` | first-100-step loss vs base; median `|tau|`/median base `|t|` | `lr' = lr × ratio` |

**Leak.** `cc = SR(rho·acc_e + tau_e)`, `rho = 1 − 1/(2·C0) ≈ 0.9`, then carry/remainder unchanged;
**0 extra bits**, one hyperparameter. Leak-free is a boxcar memory of depth `C_i`; the leak makes it an
EMA of depth 10 and bounds stale pressure — the documented failure of low-precision state, where
quantised EMAs stall and old pressure must be decayed or reset [6]. Not required by construction (the
remainder conserves pressure, the rail branch damps): enable iff `cycle_frac > 1%` or `blocked_frac > 5%`;
expected loss effect ±0.01 — robustness, not quality.

**Test plan** (`probes/charlm_phase2.py`, 800 steps, lr 0.01, seeds 0-4, medians). Reference
`--integrator 9 --second-moment row` = 2.7707. Flags: `--carry {tick,threshold}`, `--v-bits {32,4}`,
`--leak rho`, `--threshold-col`.

```
... --arm mx6 --integrator 9 --steps 800 --lr 0.01 --second-moment row --carry threshold --v-bits 4  # T1
... --carry threshold --v-bits 4 --leak 0.9                            # T2 = T1 + leak
... --carry threshold --v-bits 4 --second-moment col --threshold-col   # T3 = T1 with per-column v
```

| arm | predicted bits/char | predicted rail | mean `|acc|` | read |
|---|---:|---:|---:|---|
| T1 | **2.74-2.79** (same rule in expectation) | 1.5-2.5% | 1.5-2.0 | keep; if > 2.80 the `g_ref`/lr scale is wrong |
| T2 | 2.73-2.78 | ≤ 2.5% | 1.0-1.5 | keep only if T1 shows cycles |
| T3 | 2.85-3.05 (3.3-3.6 if rails persist) | 3-6% (11%+ if wrong) | 1.0-1.5 | ≤ 2.85 and rail ≤ 4% ⇒ position-of-`v` hypothesis supported |

T1 within ±0.02 of base is the designed outcome — a platform (4-bit `v`, no rails), not a quality
win. T3's rail fraction *is* the experiment: if it stays 11.7%, amplification is not the cause and
the next probe clamps `v` to `[1/16, 4]·g_ref²`. Log `rail_frac`, `blocked_frac`, `mean|acc|`,
`cycle_frac`, `C_i` histogram and pressure balance in the `RESULT` line.

## 7. Literature status

- Adam normalises the **momentum** by `√v` after the EMA [1]; this rule is its integer, threshold-
  discretised form, and signSGD/Signum is the pre-integration counterpart [2]. Trust-ratio
  normalisation after the fact is LARS/LAMB [3,4]; per-block state scaling is 8-bit Adam [5];
  quantised EMA state needs anti-staleness treatment [6]; threshold alignment (carry amount must be a
  grid level) is documented for dithered weight state [7] — our carry is one E2M1 step.
- **No published adaptive carry threshold with an integer accumulator exists.** Searches on
  2026-10-28 found no optimiser that normalises at the carry stage; nearest precedents are [3,4] and
  quantised-momentum storage [8]. TMLR 2026 "Ternary Momentum For Quantized Training" (OpenReview
  `A3mVmPlahU`) was inaccessible behind a browser check, not relied on. §3 is ours.

## References

[1] Kingma, Ba, *Adam* — https://arxiv.org/abs/1412.6980 · [2] Bernstein et al., *signSGD* — https://arxiv.org/abs/1802.04434
[3] You et al., *LARS* — https://arxiv.org/abs/1708.03888 · [4] You et al., *LAMB* — https://arxiv.org/abs/1904.00962
[5] Dettmers et al., *8-bit Optimizers* — https://arxiv.org/abs/2110.02861 · [6] Topollai, Choromanska, *Quantization of Optimizer States* — https://arxiv.org/abs/2603.16731
[7] Bonin, *Error Diffusion Replaces Latent Weights* — https://arxiv.org/abs/2610.34566 · [8] Tang et al., *1-bit Adam* — https://arxiv.org/abs/2102.02888
