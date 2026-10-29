# Scales: how low-bit training chooses and tracks them

Survey note, 2026-10-28, branch `mxfp4-native`. Context (`docs/PHASE2.md` Round 3): at 3000 steps
Adam-QAT is 2.4097, our rule with an fp32 master 2.3951, in-weight accumulator + headroom 1.5 =
2.4415 bits/char — a +0.032 gap to Adam-QAT after headroom removed 0.136 of 0.168. This note is
about the **scale**, not the accumulator: `s = 2^(byte-127)` per 32 weights, recomputed from amax
with a sticky band (`mx6/pack.py:scale_bytes`).

## 1. The rules that exist

| family | rule | tracking behaviour | source |
|---|---|---|---|
| absmax / MX | `s = 2^ceil(log2(amax/6))` (ours), OCP maps amax to `(4,8]` | exact, per step, no headroom | OCP MX [1], OAS paper [2] |
| headroom | `s = 2^ceil(log2(amax/h))`, `h < 6` | exact, leaves climb room | NVFP4 headroom calibrator [4]; ours |
| percentile | `s = 2^ceil(log2(q_p(abs(w))/h))` | robust to outliers, clips top `1-p` | SmoothQuant `alpha*max` [7], PE-QAT `quantile_0.995` [8] |
| EMA amax | `s from ahat = b*ahat + (1-b)*amax` | smooth, lags on spikes | PyTorch `MovingAverageMinMaxObserver` [5]; DTS `exp_smooth` [3] |
| windowed max | `s from max(amax[t-H:t])` | conservative, no saturation inside H | DTS `max`, H=64 [3] |
| current-step | `s from amax[t]` | exact but jittery | DTS `CTS` [3]; 8-bit optimizer blocks [6] |
| MSE-optimal | `s = argmin sum_e (w_e - Q_s(w_e))^2` | chooses clipping, not amax | classical Max (1960), see [9]; PTQ practice |
| learnable | `s` trained by STE (LSQ), clip trained (PACT) | adapts to loss, not to amax | LSQ [10], PACT [11] |

What is measured about them:

- **The MX/absmax rule is not neutral.** The OAS paper states the OCP rule maps amax to `(4,8]`,
  i.e. it deliberately allows overflow, and that E2M1's dynamic range is only 12x (6.0/0.5), so
  blocks with high variance flush small values to zero (20% of activation elements at block 32,
  13% at block 16, ~1 dB QSNR). E8M0 also has no mantissa: values between 4 and 6 can carry up to
  20% representation error [2]. Our repo rule (`ceil`, maps amax to `(3,6]`) avoids the overflow
  but has no headroom for updates — which is exactly the rail clipping we measured.
- **DTS/EMA tracking has a documented failure called amax saturation.** With delayed scaling the
  scale is computed from history; when the current amax exceeds the estimate, values clip and the
  STE passes gradients as if they had not — delayed estimates "silently corrupt knowledge-sensitive
  representations via forward-pass clipping", invisible in training loss. The study finds: 1-step
  lag saturates; `exp_smooth` fails on non-monotone spikes; and even current-step exact amax (`CTS`)
  was the *worst* on MMLU because per-step scale fluctuations "corrupt stable knowledge
  representations". A 64-step max window was the fix, and its accepted cost was over-conservatism:
  "raising the scale denominator and slightly reducing quantization resolution", which "acts as
  beneficial regularization" [3].
- **Percentile scales** are standard for activations (SmoothQuant's `alpha*max` with `alpha<1`,
  PE-QAT learns `quantile_0.995` clip thresholds) [7,8], and NVIDIA's headroom calibrator anchors an
  NVFP4 global scale to a low percentile times `rho` with the top 0.01% of blocks clipped [4]. We
  found no published weight-block percentile rule for *training*.
- **Learnable scales** (LSQ step size, PACT clip) are the strongest low-bit QAT baselines [10,11],
  but they need a gradient and an optimizer state per group; at block 32 an fp32 scale is 1 b/param,
  i.e. 12% more state than our whole 8.25 b/param budget.
- **MSE-optimal clipping** is the classical alternative to absmax: choose the clip/scale that
  minimises `E[(w-Q(w))^2]`, which for a Gaussian-like block lands well below the block max (a few
  sigma), i.e. *clipping* rare large values [9]. Note this is the **opposite direction to
  headroom**: MSE-optimal says "clip the tail", headroom says "leave room above amax". Our data say
  the dynamics want the latter; the two coincide only if the block's MSE is flat in the octave
  choice.
- "MSE-optimal clipping (Bondarenko et al.)" could not be verified. The Bondarenko et al. paper we
  can locate is per-embedding-group transformer PTQ [12]; it does not present an MSE-optimal
  clipping rule in its abstract/scope. Treat that attribution as unconfirmed.

## 2. Headroom is a known trick, and it has a name and a price

**Name.** It is called *headroom* in quantization tooling — NVIDIA ModelOpt ships
`NVFP4ActHeadroomCalibrator`, described as calibrating the NVFP4 global scale "with headroom for
unseen outliers" by anchoring to a low percentile times `rho` [4] — and *headroom / guard bits* in
block-floating-point DSP. The DTS paper reaches the same place from the other side and calls it
"mild over-conservatism" [3]. What we found no precedent for is headroom on **MX weight-block
scales during training**; the nearest published object is an activation *global* scale [4].

**Price.** Resolution lost at the bottom of the ladder. E2M1 has eight magnitudes, 0.5s up to 2s,
then 1s to 4s, then 2s to 6s; lowering `s` so amax sits at level `h` moves every element down the
ladder, and elements below `0.5s` flush to zero (the 12x range / flush-to-zero problem [2]). Our
own sweep at 800 steps: h=6: 2.7707, h=4: 2.7410, h=3: 2.7410, h=2: 2.7187, h=1.5: 2.7187,
h=1: 2.7728 — i.e. headroom helps until the ladder is wasted, and h=1 is past the knee. The h
pairs collapse because the scale is a power of two and only shifts in octaves.

## 3. Scale lag in our harness is structural

`scale_bytes` keeps the current scale whenever `0.5*h*s <= amax <= 6*s`; with h=6 that is
`3s..6s` (the documented band, `docs/FORMAT.md`). With h=1.5 the same code keeps the scale over
`0.75s..6s` — an 8x window in which amax can grow 4x above the target and sit on the rail before
anything is recomputed. The sticky rule exists for a good reason (re-quantising an already
quantised block on a zero-gradient step rewrites every code, `docs/UPDATE.md`), but it was not
re-tuned when h changed: **the tracking band is not the target band**. That is a concrete, testable
scale-lag source, of the same species as DTS's stale amax [3].

## 4. Per-block vs per-tensor

Published granularity evidence is all in one direction: block 32→16 buys ~1 dB QSNR [2], NVFP4's
16-element E4M3 scale buys +3-4 dB over E8M0 [2], and macro-block mantissa scaling (1x128) protects
outliers [2]. Per-channel/per-group weight scales are standard in PTQ [12]. Our harness is already
per-32, so granularity is not the remaining lever; the choice *inside* the block is.

## 5. Candidate rules for our harness

Baseline: `s = 2^ceil(log2(amax/h))`, h=1.5, kept while `0.75s <= amax <= 6s`.

| # | one-line change | evidence | expected effect on the 0.032 gap |
|---|---|---|---|
| S1 | band-consistent keep: keep only while `0.5h*s <= amax <= h*s`, else recompute | none for MX weights; DTS shows stale scales saturate [3] | -0.005 to -0.02; removes the 8x lag window |
| S2 | windowed amax: `s = 2^ceil(log2(max(amax[t-H:t])/h))`, H=16-64 | DTS max-window, FP8 activations [3] | -0.005 to -0.015; over-conservative on an 800-step distribution |
| S3 | per-block MSE octave: `s = argmin_{s in {2^m, 2^(m+1)}} sum_e (w_e - Q_s(w_e))^2`, starting at the h=1.5 scale | MSE clipping is classical [9]; no training evidence | -0.01 to +0.01; control arm — it may re-introduce clipping |

All three are one-line changes to `f(block)` (S1 changes the keep mask, not `s`) and cost 0 extra
bits. Honest expectation setting: headroom already removed 0.136 of 0.168, so the remaining 0.032 is
probably no longer dominated by scale dynamics — it may be the 4-bit grid itself (1.41% representation
error, `docs/FORMAT.md`). Before spending an arm on S2/S3, measure the block-wise reconstruction MSE
of the trained weights against the master; if that MSE is already at the format floor, no scale rule
moves the gap and the next lever is the ladder (E4M3/NVFP4-style scale mantissa), not tracking.

## 6. What is not published

- No MX weight-block **headroom** rule for training (NVFP4 headroom is an activation global scale).
- No **scale-lag** study for per-block weight scales in training; the evidence is FP8 activation
  scales [3] and QAT observers [5].
- No **MSE-optimal scale** evidence for training dynamics (only PTQ/rounding theory [9]).
- The **Bondarenko et al. MSE-clipping** attribution is unverified (see §1).
- Our h=1.5-2 knee is a measurement on one char-LM, not a reported result.

## References

[1] OCP, *MX Specification v1.0* — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf · [2] Chhugani et al., *MXFP4 quantization error reduction* (OAS, MBS) — https://arxiv.org/abs/2603.08713
[3] Cheng et al., *Max-Window Scale Estimation for HiF8 W8A8 QAT* (DTS, amax saturation) — https://arxiv.org/abs/2605.26189 · [4] NVIDIA ModelOpt, `NVFP4ActHeadroomCalibrator` — https://nvidia.github.io/Model-Optimizer/pr-preview/pr-2235/reference/generated/modelopt.torch.quantization.calib.nvfp4_act_headroom.html
[5] PyTorch, `MovingAverageMinMaxObserver` — https://docs.pytorch.org/docs/2.14/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html · [6] Dettmers et al., *8-bit Optimizers via Block-wise Quantization* — https://arxiv.org/abs/2110.02861
[7] Xiao et al., *SmoothQuant* — https://arxiv.org/abs/2211.10438 · [8] Mishra, *PE-QAT* (learned clip, `quantile_0.995`) — https://aclanthology.org/2026.acl-srw.63/
[9] Max, *Quantizing for minimum distortion* (1960), as surveyed in Gholami et al. — https://arxiv.org/abs/2103.13630 · [10] Esser et al., *Learned Step Size Quantization* — https://arxiv.org/abs/1902.08153
[11] Choi et al., *PACT* — https://arxiv.org/abs/1805.06085 · [12] Bondarenko, Nagel, Blankevoort, *Efficient Transformer Quantization* — https://arxiv.org/abs/2109.12948
