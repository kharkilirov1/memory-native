# PTQ recipe + rotation, measured 2026-10-17 (Qwen2.5-0.5B, RTX Pro 6000)

Evidence for three questions asked that day: which solver knobs actually buy quality,
whether an incoherence rotation helps the group-ternary grid at model level, and whether
token-subspace sketching of `grad_w` is viable. One model, one calibration slice, no
recovery — these are PTQ-init numbers, not training results.

Setup: Qwen2.5-0.5B (494M, fp32 on GPU), calibration = wikitext-2 train, 32 × 512
tokens; evaluation = wikitext-2 validation, 8 × 512 tokens, token-weighted PPL
(fp32 reference 11.7644). Dense-weight substitution path
(`quantize_dense_group_ternary`), group 128, C=11, `residual_alpha=0`.

## 1. Which knobs buy quality

| config | PPL | quantize time |
|---|---:|---:|
| fp32 (pre-quantization) | 11.76 | — |
| `hdiag`, iters 2, no salient (**defaults**) | 3436.39 | 69 s |
| `align`, iters 2 | 1493.92 | 102 s |
| `align`, iters 2, `salient_first=0.01`, sym | 346.94 | 123 s |
| **`align`, iters 4, `salient_first=0.01`, sym** | **279.62** | 202 s |
| `align`, iters 2, `salient_first=0.01`, itf | 516.15 | 129 s |

- `scale_refit="align"` (A7 closed-form H-metric scale solve) is worth **2.3x**.
- `salient_first=0.01` is worth **4.3x** on top of align at iters=2.
- `refine_iters` 2 -> 4 adds **1.24x** more; past 4 it saturates (see layer table).
- The convenience defaults (`hdiag`, no salient) are **12.3x worse** than the recipe at
  **0.34x** of the time — a footgun for anyone calling the dense helper directly.
- `grid="itf"` is WORSE on this dense path at iters=2 (516 vs 347): itf pays off only
  through the exact sym re-solve that `solve_group_state` performs, not in the dense
  helper.

Production runners already default to `SCALE_REFIT=align`, `SALIENT_FIRST=0.01`,
`GRID=itf`; the one stale knob is `REFINE_ITERS=2` in `scripts/run_ptq_recovery.py`
(measured sweet spot >= 4 with salient).

## 2. Layer-level H-error (3 real layers, `(w-q)^T H (w-q) / w^T H w`)

| config | L0 q_proj | L5 down_proj | L11 gate_proj |
|---|---:|---:|---:|
| iters 2 hdiag | 2.452% | 6.848% | 3.894% |
| iters 2 align | 0.670% | 6.186% | 3.773% |
| iters 4 align | 0.561% | 6.091% | 3.747% |
| iters 8 align | 0.510% | 6.068% | 3.737% |
| iters 4 align + in-sweep refit | 0.529% | 6.038% | 3.743% |
| iters 4 align itf | 0.734% | 8.729% | 4.011% |
| iters 4 align itf + salient 1% | 0.286% | 5.108% | 3.359% |

Same ordering as the PPL table: align > hdiag, salient adds more, iterations saturate.

## 3. Counter channel at different `residual_alpha`

Same solver solution read as `W = s (t + alpha c / C)`, H-error vs the original `w`:

| alpha | L0 with c | L0 with c=0 | L11 with c | L11 with c=0 |
|---|---:|---:|---:|---:|
| 0 | 0.263% | **0.263%** | 3.216% | **3.216%** |
| 0.25 | 0.261% | 0.263% | 1.945% | 3.216% |
| 0.5 | 0.687% | 0.263% | 1.497% | 3.216% |
| 1.0 | 2.822% | 0.263% | 3.074% | 3.216% |

At the deployable readout (`alpha=0`) the counter is **exactly invisible**: `c=0` gives the
identical weight. So "Hessian-metric recovery of c" has no deployment objective; `c` is fit
to the GPTQ feedback target `W_adj`, which is why the alpha=1 weight can be 10.7x further
from `w` than the deployable one. Improving `c` can only target training dynamics (the
homotopy), and the right objective there is the loss, not `||W-Q||_H`.

## 4. Incoherence rotation (R1) at model level

`donor/rotate.py::rotate_residual_stream` (seed 0), same solver config `align, iters 2, sym`:

| rotation | salient | PPL | quantize time |
|---|---|---:|---:|
| — | 0 | 1493.92 | 101 s |
| — | 1% | 346.94 | 123 s |
| yes | 0 | 1295.03 | 102 s |
| **yes** | **1%** | **247.86** | 124 s |

- Invariance verified: PPL 11.764424 -> 11.764425 before/after rotation (fp32 noise), so
  the rotation is exactly output-preserving and free at runtime.
- Rotation without salient: 1.15x better. Rotation **does not remove the salient channel**:
  rotated+salient is 5.2x better than rotated-only.
- Rotation **with** salient adds 1.40x (346.9 -> 247.9) for +1 s of solve time.

Caveat: the pinned synthetic witness `test_rotation_hurts_group_ternary_fit_documented_negative`
still shows rotation hurting a single outlier-heavy layer (both with hdiag and with
align+salient: 1.53x / 3.67x worse on that toy). The model-level gain above is therefore
**not yet explained** and the flag stays opt-in (`ROTATE=1`, `rotate=True`) until a
deploy-scale warm gate reproduces it on 1.5B/12B.

## 5. Token-subspace sketching of `grad_w` — not viable as written

Heavy-tailed synthetic stand-in (N=K=896, B=512, row norms lognormal(0,1)), estimator as
proposed (HT, `p ∝ ||d_m|| ||x_m||`, no replacement):

| r | rel err ||G~-G||/||G|| | code agreement | ternary sign disagreement | mean |dc| |
|---:|---:|---:|---:|---:|
| 64 | 1.010 | 23% | 26.4% | 4.04 |
| 128 | 0.890 | 22% | 25.9% | 3.95 |
| 256 | 0.905 | 17% | 27.2% | 4.16 |
| 512 | 0.938 | 14% | 28.4% | 4.39 |

The variance is dominated by the spread of `p`, so the error stays ~0.9 even when
`r = B`; ~27% of the visible ternary signs would move in a single update.

Ceiling, measured on the same GPU (4096x4096, B=512, `kernel_mode="gemm"`):
`forward 1.045 ms | grad_x 0.985 ms | update 0.594 ms`, and the correlation `go^T x` inside
the update is ~0.2 ms of a 2.6 ms step. Even a 16x faster `grad_w` therefore saves ~7% of
the step: this architecture's step is decode/automaton-bound, not GEMM-bound. Sketching is
not a priority for the counter path.

## What to adopt

1. Recipe: `align` + `salient_first=0.01` + `refine_iters>=4` (+ rotation, gated) — 12.3x
   PPL over the old defaults at 2.9x solve time.
2. `ROTATE=1` as an opt-in for the dense warm-start path (measured 1.40x on 0.5B, pending
   a deploy-scale gate; the streaming converter materializes blocks incrementally, so it
   does not support it yet).
3. Do not pursue `grad_w` sketching (7% ceiling, unusable variance) and do not drop the
   salient channel in favour of rotation (5.2x regression).

Reproduce: the four rotation cells are a ~9 min GPU run; the config table is ~8 min; both
were driven from Colab cells that only need `transformers`, `datasets`, and the repo's
`donor.ptq` / `donor.rotate` (no recovery loop, no donor weights beyond the 0.5B model).
