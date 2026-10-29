# Literature: memory-cheap adaptive optimizers for a bit-budgeted counter

Surveyed 2026-10-28 for the `mxfp4-native` design. The question is narrow: at 4-bit weights plus
a 4-bit gradient integrator (8.25 bits/param, `docs/PHASE2.md`), which published mechanism can
supply the missing per-element adaptive step for **≤ 1 extra bit/param** (0.125 B/param = 1.5 GB
at 12B)? All numbers below are *reported by the cited paper* unless marked as ours.

Reference arms (ours, tiny char-LM, 800 steps, bits/char): dense fp32+AdamW 2.6619; post-hoc
MXFP4 floor 2.6916; MXFP4+SR no state 3.1287; MXFP4+9-state integrator 2.7707. So the format
costs 0.03 and the remaining 0.079 is update dynamics, not representation.

Notation: `w = s*v`, `g` = grad of the tile, `R_i` = per-row second moment (the existing
4 B/row state), `C_j` = per-column second moment (the state we do not have).

---

## 1. Momentum, old and new

**Heavy ball** (Polyak 1964) and **Nesterov** both keep one buffer and one history length:

```
heavy ball:   z_{t+1} = beta*z_t + g_t ;              w_{t+1} = w_t - alpha*z_{t+1}
Nesterov:     z_{t+1} = beta*z_t + grad_f(w_t - alpha*beta*z_t) ; w_{t+1} = w_t - alpha*z_{t+1}
```

The Nesterov look-ahead point is a function of `w_t` and `z_t`, so Nesterov needs **no extra
state over heavy ball** [2]. What momentum fixes, in three separable parts:

1. **Noise averaging.** For iid gradient noise of variance `sigma^2`, the EMA has
   `Var(z) = sigma^2 (1-beta)/(1+beta)`. At `beta = 0.9` that is a 4.4x reduction in gradient
   SD; the averaging window is `1/(1-beta)` steps. This is why momentum survives a noisier
   update rule (e.g. stochastic rounding) where a per-step rule would not.
2. **Conditioning.** On a quadratic with condition number `kappa`, heavy ball with optimal
   `beta = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2` converges at `(sqrt(kappa)-1)/(sqrt(kappa)+1)`
   instead of `(kappa-1)/(kappa+1)`, and Nesterov's method meets the `O(1/k^2)` lower bound
   [1]. The buffer buys a per-direction rescaling without storing one number per direction —
   the cheapest conditioning mechanism that exists.
3. **Plateaus / low-curvature regions.** Accumulated velocity carries through regions where
   the instantaneous gradient is small, which is exactly the "no tick fires for many steps"
   regime we saw at 3 states.

**What the theory says about memory depth.** State count and memory depth are related
logarithmically: an accumulator that must reach `C` needs `ceil(log2 C)` bits, while the depth
is `C` steps (boxcar) or `1/(1-beta)` (EMA). So *one extra bit doubles the memory depth*. Beyond
the point where depth exceeds the objective's curvature timescale, extra bits are wasted. Our
sweep is consistent: 3 states (2 bits) recovers half the gap, 9 states (4 bits, ~9-step
boxcar ≈ `beta ~ 0.89`) recovers ~83%, 17 states is *worse* than 9 (non-monotone: the tick
granularity `lr/C` decouples from the step size) and 33 states buys only ~0.006 bits/char
[PHASE2].
AdEMAMix finds that in LLM pretraining gradients stay useful for tens of thousands of steps
(`beta_3 = 0.9999`, 1.3B/101B tokens matches AdamW at 197B tokens) [11] — i.e. the depth
requirement is a property of the task horizon, and our 800-step tiny LM sits far below it.

Sutskever et al. 2013 is the empirical precedent for our result: with a good init and a
momentum schedule, *"carefully tuned momentum methods suffice for dealing with the curvature
issues ... without the need for sophisticated second-order methods"* [2]. Treating the
integrator as momentum, not as an optimizer, is the reading our measurements support.

---

## 2. Error feedback vs momentum — different problems

Error feedback (EF) is the recursion

```
x_t = Q(g_t + e_{t-1}) ;   e_t = g_t + e_{t-1} - x_t          # Q = compressor/quantizer
```

`e` carries the *quantization error* forward. 1-bit SGD is the original instance (1 bit per
gradient, essentially unchanged speech-DNN accuracy) [4]; Karimireddy et al. prove that
EF-SGD recovers the SGD rate for an **arbitrary biased** compressor, and that plain signSGD
can fail to converge on convex counterexamples [5].

| mechanism | carries | solves | state / param | fails when |
|---|---|---|---|---|
| momentum / integrator | gradient history | noise averaging, conditioning, plateaus | log2(depth) bits | depth << task timescale |
| error feedback | rounding error | **bias** of a biased quantizer | same as error magnitude | rounding already unbiased |
| sigma-delta / noise shaping | same loop, higher order | shaped (high-pass) error spectrum | 1 buffer per order | no optimizer-side evidence found for order > 1 |
| SR (ours) | nothing | bias, for free in expectation | 0 | adds variance, no memory |

**EF is first-order sigma-delta.** The recursion above is exactly a first-order noise-shaping
(ΣΔ) modulator; the optimizer literature uses the same loop [4,5]. We found **no peer-reviewed
application of higher-order ΣΔ noise shaping to optimizer state** (searched; nothing found).
The nearest modern work is error *diffusion* for dithered weight state, which reports 2-8-bit
buffers and 4-bit ≈ 8-bit quality under a "threshold-alignment" rule: the buffer only works if
its grid has a level exactly at the carry threshold [20]. Our carry is exactly one E2M1 step,
i.e. the carry amount is itself a grid level — the alignment property [20] identifies as
necessary.

Our measurements separate the two mechanisms cleanly (`docs/UPDATE.md`): with stochastic
rounding the quantization error is already mean-zero, so the EF buffer carries pure noise and
makes things *worse* (rel. MSE 0.0279 → 0.0295), while deterministic rounding without EF cannot
train at all (3.0 → 0.0319 with EF). The literature agrees on the diagnosis: EF exists to fix
bias [5], and it *is* needed when a deterministic step is applied to quantized weights
(ECO injects the quantization error into momentum and proves a constant-radius limit; naive
master-weight removal has stationary error scaling as `1/lr`) [21]. We do not have that failure
mode because we round stochastically — but we do have the momentum-shaped gap.

**Consequence for the design:** do not spend bits on an error buffer; spend them on a *memory*
of gradients (already done) and, next, on a *per-element scale* (this survey).

---

## 3. Memory-cheap adaptivity

Adam's per-element step is `u_ij = m_ij / (sqrt(v_ij) + eps)` with `m,v` two 4-B buffers
(64 bits/param) [26]. Each row below trades some of that away.

| method | update (sketch) | per-param state | reported quality/memory trade |
|---|---|---|---|
| AdamW [26] | `m = b1 m + (1-b1) g`; `v = b2 v + (1-b2) g^2`; `w -= lr*m/(sqrt(v)+eps)` | 64 bits (m, v) | reference |
| **Adafactor** [6] | keep row RMS `R_i`, col RMS `C_j`; `vhat_ij = R_i C_j / mean(R)`; `w -= rho_t g/sqrt(vhat)`, `rho_t = min(1e-2, 1/sqrt(t))` | `n+m` floats = `32(1/n_in + 1/n_out)` bits | "similar results to baseline"; comparable to Adam on WMT14 En-De while dropping momentum and using sublinear optimizer memory |
| **LARS** [7] | layer-local lr `lambda = eta*||w||/(||g|| + beta_wd*||w||)` | 0 persistent, 2 scalars/layer | AlexNet batch 8K, ResNet-50 batch 32K at no accuracy loss |
| **LAMB** [8] | `r = ||w||/||u||` applied to the Adam update; `w -= eta*r*u` | 64 bits + O(1)/layer | BERT batch 32K, 76 min; fixes LARS' attention failure |
| **Lion** [9] | `c = b1 m + (1-b1) g`; `w -= lr*sign(c)`; `m = b2 m + (1-b2) g` | 32 bits (m only, no v) | up to 5x pretraining compute on JFT, +2% ViT; needs a smaller lr than Adam; **removes** v, does not add it |
| **Sophia** [10] | `u = clip(m/(rho*h + eps), -1, 1)`; `h` = EMA of diagonal Hessian (Hutchinson, every ~10 steps) | 64 bits (m, h) | 2x fewer steps / same perplexity, 125M-1.5B; Hessian estimate is lazy, so amortized memory is "negligible" but the h buffer is full-size |
| **AdEMAMix** [11] | `u = (m1 + alpha*m2)/(sqrt(v)+eps)`, `m1` fast, `m2` slow (`b3 ~ 0.9999`) | 96 bits (m1, m2, v) | 1.3B/101B tokens ≈ AdamW at 197B; slows forgetting |
| **Muon** [12] | `m = b m + (1-b) g`; `O = NS5(m)` (orthogonalize); `w -= lr*(scale*sqrt(max(n,m))*O + wd*w)`, `scale = 0.2` | 32 bits (m) + Newton-Schulz compute | ~2x compute efficiency vs AdamW (scaling laws), Moonlight 3B/16B; fixes conditioning without a second moment, but its state is direction-sensitive to quantization [17] |
| **Schedule-free AdamW** [13] | gradient evaluated at `y`, an interpolation of fast iterate `z` and running average `x`; `z -= lr*m/(sqrt(v)+eps)`; `x` tracks `z` | 64 bits + one extra iterate (96 bits) | matches tuned schedules without knowing `T`; MLCommons AlgoPerf 2024 self-tuning winner; costs *more* state, not less |
| ADOPT [22] | normalize by `max(v_t, v_{t-1})` | 64 bits (+1 if both kept) | fixes Adam's non-convergence for any `b2`, optimal rate; no extra memory if the max replaces the EMA |
| Cautious [23] | mask update by `sign(m) == sign(g)` | 0 extra | consistent LLM pretraining speed-up, one line of code |

Only two families get **sublinear** state: factored second moments (Adafactor) and layer-wise
scalars (LARS/LAMB). Everything else is a full per-element buffer at reduced precision
(section 4) or a different per-element buffer (Lion trades `v` for `m`).

---

## 4. Low-precision optimizer state — what accuracy is retained

| state precision | mechanism | reported retention |
|---|---|---|
| 8-bit (m, v) | block-wise dynamic quantization (block 2048 → 256 bins) + stable embedding layer [14] | 32-bit performance on 1.5B LM, GLUE, ImageNet, WMT, RoBERTa; Adam state 8 B → 2 B/param (75% less) |
| 1-bit momentum | sign-compressed `m` after warmup + error compensation [15] | same convergence speed as uncompressed Adam, up to 5x less communication; requires warmup because Adam's `v` stabilizes |
| 4-bit AdamW | NF4 first moment, zero-inclusive **preconditioner-space** stochastic rounding of `v` (ZIP-SR) [16] | reduces TorchAO 4-bit AdamW's val-loss gap to 32-bit by up to 70%, 130M-2.7B; key claim: small state error does not imply small *preconditioner* error next step |
| 4-bit Muon | pre-quantization normalization, singular-direction-preserving decomposition, `mu`-law companding [17] | most of full-precision Muon's loss and downstream accuracy; up to 7.3x optimizer-state reduction (4-bit is harder for Muon than for Adam) |
| NVFP4 AdamW state | ordered sqrt + tile-mean + Hadamard, per-tile scaling [18] | 3B/64B tokens: 2.286 vs BF16 2.267 (0.838% gap), zero-shot PPL tied; 37.9-42.5% lower AdamW peak memory |
| 2-8 bit dithered weight state | error diffusion, stochastic-rounded residual buffer with threshold alignment [20] | 4-bit buffers match 8-bit; 3-9 bits/weight total, up to 19x less training state; 5-11% loss gap vs latent-weight reference at 2k steps |
| low-precision EMA state | staleness analysis + state resets [19] | quantization makes many nominal EMA updates round back to the same code (stalling); scheduled resets recover the lost quality |
| FP8 (E4M3) state | block-wise scaling for optimizer + activations [25] | near-lossless FP8 training at reduced memory |

**Rules the literature supports.** (i) 8 bits per state is effectively free [14]. (ii) 4 bits
is free-to-small-gap *only* when rounding is done in the preconditioner space [16] and with
shared-exponent block scaling (per-32/64 E8M0, or per-tile mean) [18,24]; state-space rounding
perturbs the recurrence. (iii) 1-2 bits per element
is not reported for `v` — the 1-bit result is for *momentum* [15] and works because momentum is
linear in `g`; `v` is quadratic, so the same trick does not transplant obviously. (iv) Coarse
state is tolerable because it is scalar per element and the update is normalized anyway; this
is the gap we can exploit.

---

## 5. Ranked shortlist for our constraint

Budget: **≤ 1 bit/param** on top of 4-bit weight + 4-bit integrator. Bit costs below are exact
for the char-LM's eight quantized linears (2 blocks × {qkv 128→384, proj 128→128, fc 128→512,
out 512→128}, N = 393,216 params): a per-row vector costs `32/n_in` bits/param, a per-column
vector `32/n_out`, per-element 1 bit = 1.0, one extra float per row is `0.1875`. At 12B/width
8192 a row or column vector is `32/8192 = 0.004` bits/param — i.e. the factorized options get
*cheaper* with scale while per-element options do not.

| # | mechanism | formula (our variables) | extra bits/param | expected gain (guess) | backward-only implementation |
|---|---|---|---|---|---|
| 1 | **Adafactor column factor** (rank-1 `v`) | `g_hat = g / sqrt((R_i*C_j/mean(R)).clamp(eps))`; `C = b2*C + (1-b2)*colmean(g^2)` | **+0.146** (bf16: +0.073); `32/n_out` in general | most likely real: supplies the column half of a per-element scale; Adafactor ≈ Adam [6]; low risk | one `bf16[in_features]` vector per counter layer next to `v_rms`; fused `rsqrt` on the tile; bias-correct `C` with a per-layer step counter |
| 2 | **1-bit per-element variance class**, anchored on `R_i` | `s_ij = R_i * 2^{(2*b_ij-1)/2}`; `u = w - lr*g/sqrt(s)`; `b` = 1-bit log-EMA of `g^2/R_i` (hysteresis at `2^{±1/2}` or SR) | **+1.0** (exactly the budget); **+0.0** if packed into the integrator's 7 unused codes (3-bit counter + 1-bit flag) | largest headroom: approaches per-element Adam; 4-bit states are nearly free [16,18], 1-bit is an extrapolation | 4 B/32-param plane (or repack); flag update and multiplier are integer logic in the same backward that ticks the counter; SR on the weight absorbs the flag's rounding error |
| 3 | **Cautious tick gating** | tick only if `sign(g_ij) == sign(c_ij)`; else skip the tick | **+0.0** | moderate: removes ticks that push against the accumulated direction; consistent LLM speed-up [23] | one comparison in the tick loop; no state change; combine with larger lr |
| 4 | **Per-layer trust ratio** (LARS/LAMB) | `eta_local = eta * clamp(||w||/(||u||+eps), lo, hi)` computed per tile from codes | **+0.0** (2 scalars/layer, not per param) | small at d=128, grows with depth/width; normalizes against the fixed grid quantum `s` | norms from the tile's nibbles and the proposal `u`; scalar rescale before SR |
| 5 | **ADOPT-style max normalization** for `R_i` | divide by `sqrt(max(R_t, R_{t-1}))` instead of `R_t` | **+0.0** (replaces the EMA) or +0.1875 if both kept | small: stabilizes per-row scale for any `b2`, optimal rate [22] | one `max` on the existing row scalar; no new buffer |
| 6 | **Sophia-lite row clip** | `u = clip(m_row/sqrt(h_row), -rho, rho)` with `h_row` = per-row curvature EMA | **+0.0** if `h` reuses `v_rms`; +0.1875 if separate | small at this scale; tames late-training spikes; the full method needs per-element `h` [10] | clip the proposal norm per row before quantizing; one branch |
| — | **Muon orthogonalization** | decode integrator → `O = NS5(m)`; `u = 0.2*sqrt(max(n,m))*O` | 0 persistent, but faithful Muon needs an fp32 `m` (**+32**) | best pure-conditioning mechanism [12], but 4-bit state is known to hurt its direction [17], and NS is ~5 matmuls/tile/step | compute-bound; only worth it if a per-tile fp32 `m` is allowed |
| — | **AdEMAMix / schedule-free / Lion** | — | +32 to +96, or removes adaptivity | not budget-compatible or wrong direction | — |

**Not found:** no paper on a ≤1-bit *per-element second moment* for weight training. Nearest
points are 4-bit optimizer state [16,17], dithered 2-8-bit buffers [20], and 1-bit momentum
[15]. Candidate 2 would be a new result; candidates 1 and 3-6 are literature-backed.

### Top 2 recommendations

1. **Adafactor column factor (rank 1).** One `bf16` vector per counter layer, `+0.073` to
   `+0.146` bits/param on this model and `+0.004` at LLM width; it is the cheapest thing that
   converts our per-row scale into a rank-1 per-element scale, and the only option that is both
   literature-standard [6] and strictly generalizes what we already have. Implement as
   `g_hat = g * rsqrt((R_i * C_j / mean(R)).clamp_min(eps))` between the gradient tile and the
   proposal `u`; nothing else changes.
2. **1-bit per-element variance class (rank 2), packed free into the 4-bit integrator.** Spend
   the 4-bit word as 3-bit counter (8 states; measured 9-state arm is 2.7707, so one state is
   worth ~0.01-0.02 bits/char) plus 1-bit `v`-class, so the added adaptivity costs **0 bits**;
   fall back to a real 1-bit plane if 8 states is measurably worse. This is the only candidate
   that attacks the diagnosis directly (per-element, not per-row).

Suggested arms, 800 steps, same 5 seeds, compare against 2.7707 (9-state) and 2.6916 (floor):
(A) +column factor, (B) 3-bit counter + 1-bit class, (A+B), (A+Cautious). If (A) recovers less
than ~0.02 of the 0.079, the gap is not rank-1 and (B) is the next bit to spend.

---

## References

1. Goh, *Why Momentum Really Works*, Distill 2017 — https://distill.pub/2017/momentum/
2. Sutskever, Martens, Dahl, Hinton, *On the importance of initialization and momentum in deep learning*, ICML 2013 — https://proceedings.mlr.press/v28/sutskever13.html
3. *(Nesterov 1983 acceleration and lower bound: cited via [1].)*
4. Seide, Fu, Droppo, Li, Yu, *1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs*, Interspeech 2014 — https://www.isca-archive.org/interspeech_2014/seide14_interspeech.pdf
5. Karimireddy, Rebjock, Stich, Jaggi, *Error Feedback Fixes SignSGD and other Gradient Compression Schemes*, ICML 2019 — https://arxiv.org/abs/1901.09847
6. Shazeer, Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*, ICML 2018 — https://arxiv.org/abs/1804.04235
7. You, Gitman, Ginsburg, *Large Batch Training of Convolutional Networks* (LARS), 2017 — https://arxiv.org/abs/1708.03888
8. You et al., *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes* (LAMB), ICLR 2020 — https://arxiv.org/abs/1904.00962
9. Chen et al., *Symbolic Discovery of Optimization Algorithms* (Lion), NeurIPS 2023 — https://arxiv.org/abs/2302.06675
10. Liu, Li, Hall, Liang, Ma, *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*, 2023 — https://arxiv.org/abs/2305.14342
11. Pagliardini, Ablin, Grangier, *The AdEMAMix Optimizer: Better, Faster, Older*, 2024 — https://arxiv.org/abs/2409.03137
12. Liu et al., *Muon is Scalable for LLM Training*, 2025 — https://arxiv.org/abs/2502.16982
13. Defazio et al., *The Road Less Scheduled* (schedule-free), 2024 — https://arxiv.org/abs/2405.15682
14. Dettmers, Lewis, Shleifer, Zettlemoyer, *8-bit Optimizers via Block-wise Quantization*, ICLR 2022 — https://arxiv.org/abs/2110.02861
15. Tang et al., *1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed*, ICML 2021 — https://arxiv.org/abs/2102.02888
16. Li et al., *Rounding in Preconditioner Space: Redesigning 4-bit AdamW Optimizer-State Quantization*, 2026 — https://arxiv.org/abs/2610.12444
17. Su et al., *MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization*, COLM 2026 — https://arxiv.org/abs/2605.11396
18. Ding et al., *Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention*, 2026 — https://arxiv.org/abs/2607.04422
19. Topollai, Choromanska, *Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets*, 2026 — https://arxiv.org/abs/2603.16731
20. Bonin, *Error Diffusion Replaces Latent Weights: Training Binary-Weight Language Models on a Few Bits of Dithered State*, NeurIPS 2026 SLM-Agents workshop — https://arxiv.org/abs/2610.34566
21. Nikdan, Zandieh, Alistarh, Mirrokni, *ECO: Quantized Training without Full-Precision Master Weights*, 2026 — https://arxiv.org/abs/2601.22101
22. Zhang et al., *ADOPT: Modified Adam Can Converge with Any beta_2 with the Optimal Rate*, NeurIPS 2024 — https://proceedings.neurips.cc/paper_files/paper/2024/hash/84d286e32bbee8fa3a86ee9c50e00081-Abstract-Conference.html
23. Liang, Chen, Liu, Liu, *Cautious Optimizers: Improving Training with One Line of Code*, 2024 — https://arxiv.org/abs/2411.16085
24. OCP, *Microscaling Formats (MX) Specification v1.0* (E8M0 scaling) — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
25. Wang et al., *COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training*, 2024 — https://arxiv.org/abs/2410.19313
26. Kingma, Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015 — https://arxiv.org/abs/1412.6980
