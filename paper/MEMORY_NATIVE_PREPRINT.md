# Training Without Master Weights: a 6-bit Finite-State Synapse with Optimizer-in-State

*Preprint draft v0.1 — formalization of the memory-native training method implemented and validated
in this repository. All empirical claims cite a witness in `results/`; claims not yet validated at
scale are marked OPEN.*

## Abstract

Quantization-aware training of ternary networks (BitNet-class) stores, per parameter, a full-precision
master weight and Adam moments — ≈16 bytes of training state per 1.58-bit inference weight. We
formalize a **finite-state counter synapse** in which the *entire* per-parameter training state is a
single 6-bit code: a visible ternary weight and a bounded stochastic-rounding accumulator, updated
in-place by exact backpropagated gradients. We show the automaton implements, in expectation,
RMS-normalized SGD on a latent weight, with the visible ternary weight a quantized readout whose
residual is tracked exactly in state (hardware-style error feedback). Composed with reversible
coupling blocks (O(1)-in-depth activation memory), a counter-state mixture-of-experts with an
exact-for-active update, and int8/bf16 tensor-core compute, the method trains a 1.21B-parameter
model end-to-end on a single 14.6 GiB GPU in 2.25 GiB peak — where the dense+Adam equivalent cannot
allocate its 18 GiB of state — at ≈0.9× dense step speed. Convergence parity with AdamW at scale is
the open question; we specify the falsifying experiment.

---

## 1. The finite-state counter synapse

### 1.1 State space

Fix an integer $C \ge 2$ (default $C=11$). Each scalar parameter is a pair

$$\sigma = (t, c), \qquad t \in \{-1, 0, +1\},\quad c \in \{-(C\!-\!1), \dots, C\!-\!1\},$$

encoded injectively into one code

$$\mathrm{enc}(t,c) = (t+1)(2C-1) + (c + C - 1) \in \{0, \dots, 3(2C-1)-1\}.$$

For $C=11$: $3 \cdot 21 = 63$ states, $\lceil \log_2 63 \rceil = 6$ bits; four codes pack into three
bytes (0.75 B/parameter persistent). A weight matrix $W \in \mathbb{R}^{n_\text{out} \times n_\text{in}}$
carries per-row scales $s \in \mathbb{R}^{n_\text{out}}_{>0}$ and per-row second-moment estimates
$v \in \mathbb{R}^{n_\text{out}}_{\ge 0}$ — $O(n_\text{out})$ fp32, amortized to $o(1)$ bits per
parameter.

**Visible weight (readout).** The forward pass uses only the ternary component:
$$W^{\text{vis}}_{oi} = s_o\, t_{oi}, \qquad y = x\, (W^{\text{vis}})^\top .$$
The counter $c$ is invisible to the forward — verified exactly (zeroing $c$ changes the forward by
$0.0$; `scripts/fusion_invariants.py`). This is what makes the forward a *ternary* mixed-input GEMM
(BitNet-inference class) rather than an exotic 6-bit one.

### 1.2 Latent-weight view

Define the **latent position** and latent weight
$$u_{oi} = \Big(t_{oi} + \frac{c_{oi}}{C}\Big) s_o .$$
Then $W^{\text{vis}}_{oi} = u_{oi} - s_o \frac{c_{oi}}{C}$ with residual bounded by
$|s_o c_{oi}/C| \le s_o \frac{C-1}{C} < s_o$: the visible weight is a ternary quantization of the
latent weight whose quantization error is **stored exactly in the state** rather than discarded —
error feedback in the sense of Seide et al. (2014), realized at 6 bits total.

### 1.3 Update rule (the automaton)

Let $g = \nabla_W \mathcal{L}$ be the exact backpropagated weight gradient (formed per layer,
transiently). One update step with learning rates $\eta, \eta_s$, EMA factor $\beta$, floor
$\varepsilon$ ("exact" mode; the shipped default):

1. **Row statistic:** $\bar g^2_o = \tfrac{1}{n_\text{in}} \sum_i g_{oi}^2$; $\quad v_o \leftarrow \beta v_o + (1-\beta)\bar g^2_o$; $\quad D_o = \max(\sqrt{v_o},\, \varepsilon)$.
2. **Scale learning:** $\gamma_o = \tfrac{1}{\sqrt{n_\text{in}}} \sum_i g_{oi} t_{oi}$; $\quad s'_o = \mathrm{clip}(s_o - \eta_s \gamma_o;\ 10^{-5}, 10)$.
3. **Tick:** $\Delta_{oi} = -\eta\, \dfrac{g_{oi}}{D_o} \cdot \dfrac{C}{s'_o}$ (counter units); rebase $\tilde c_{oi} = c_{oi}\, s_o / s'_o$.
4. **Stochastic rounding:** $\hat c_{oi} = \mathrm{SR}(\tilde c_{oi} + \Delta_{oi})$, where $\mathrm{SR}(x) = \lfloor x \rfloor + \mathrm{Bern}(x - \lfloor x \rfloor)$, so $\mathbb{E}[\mathrm{SR}(x)] = x$.
5. **Carry / saturation:** $k = \mathrm{trunc}(\hat c / C)$; $\ t' = \mathrm{clip}(t + k; -1, 1)$; remainder $r = \hat c - kC$, and if the clip was active, $r = \mathrm{sign}(\hat c)(C-1)$; $r$ clamped to $\pm(C-1)$. New state $(t', r)$, new scale $s'$.

The optimizer **is** steps 1–5: there is no other per-parameter state, no `.grad` retained, no
parameter registered with an outer optimizer. The update executes inside the layer's backward
(one fused kernel launch; §4).

### 1.4 Expected dynamics

**Proposition 1 (unbiased latent RMS-SGD).** Away from ternary saturation ($|t + k| \le 1$) and
scale-clip boundaries, one step satisfies
$$\mathbb{E}\left[u^{(k+1)}_{oi} \,\middle|\, g\right] = u^{(k)}_{oi} \; - \; \eta\, \frac{g_{oi}}{D_o},$$
i.e., the automaton performs, in expectation, RMS-normalized SGD on the latent weight, with per-step
quantization noise bounded by one counter quantum $s'_o/C$.

*Sketch.* $\Delta u = s'( \hat c + (t+k)C - \tilde c - tC )/C$ telescopes; SR contributes zero mean
by construction; the rebase $\tilde c = c\,s/s'$ keeps the latent value invariant under the scale
change. The carry moves integer overflow from $c$ into $t$ without changing $t + c/C$. $\square$

Saturation at $|t|=1$ clamps the latent to $\pm s(2C-1)/C$ — the (intended) bias of a bounded
weight range; the remainder rule keeps the state at the boundary rather than wrapping.

**Proposition 2 (unbiased low-bit gradient substitution).** With conditionally independent
stochastic quantizers $Q_a, Q_b$ (per-column symmetric int8/int4),
$\mathbb{E}[\,Q_a(\Delta)^\top Q_b(X)\,] = \Delta^\top X$, so replacing the update correlation with
an integer tensor-core GEMM preserves the expected update; the same argument covers the unbiased
low-bit *saved activation* $Q(x)$ used in place of $x$ ($\mathbb{E}[Q(x)] = x$). (Tests:
`test_int8_compute.py`, `test_actquant.py`.)

**Determinism variants.** Two SR families are implemented: `torch.rand` SR (supports the one-pass
"lagged" mode) and a deterministic hash-SR (MurmurHash of the element index XOR seed) used by the
fused kernels. Hash-SR makes updates bit-reproducible and **unconditionally DDP-safe**: with the
weight-gradient all-reduced inside backward, replicas' packed states remain byte-identical
(verified empirically: 0 differing bytes across ranks with different per-rank data;
`test_ddp_decimation.py`).

### 1.5 Memory accounting

**Proposition 3 (training-state footprint).** For an $N$-parameter counter linear stack, persistent
training state is $6N$ bits $+\;O(\text{rows})$ fp32 $= 0.75N$ bytes $+ o(N)$, versus $16N$ bytes
for fp32 weights + Adam ($m, v$) — a ≈21× reduction; versus 2-byte bf16 master + 8-bit Adam
(≈4 B/param) — ≈5×. Measured at 1.21B parameters: **0.87 GiB counter state vs 18.0 GiB** dense+Adam
(`results/SCALE_1B.md`).

## 2. Activation memory: reversible coupling with anchors

Blocks are additive couplings $y_1 = x_1 + F(x_2),\ y_2 = x_2 + G(y_1)$ with exact inverse
$x_2 = y_2 - G(y_1),\ x_1 = y_1 - F(x_2)$ (RevNet). The backward reconstructs inputs from outputs,
then recomputes the local forward under autograd — gradients equal standard backprop exactly (the
recompute is the same function), at $\Theta(1)$-in-depth activation memory instead of
$\Theta(L)$, for +1 forward-equivalent of compute. `anchor_every=A` interpolates: store every
$A$-th activation and checkpoint-recompute (no inverse), $O(L/A + A)$ memory.
**Constraint:** $F, G$ deterministic — hence the int8 forward uses round-to-nearest, not SR.
Measured: anchors $A{=}2$ recover +35% step speed over pure reversible at +0.11 GiB, loss identical
(`results/PERF_ANATOMY.md`); in the GLM stack, ×3.1 lower training peak vs non-reversible
(`results/MN_GLM_1B5.md` §5c).

## 3. Counter mixture-of-experts with exact-for-active updates

FFN = top-$k$ of $E$ counter-state experts (SwiGLU), fp router, switch-style load-balance auxiliary.

**Proposition 4 (exactness).** A token not routed to expert $e$ contributes exactly zero to
$\partial \mathcal{L} / \partial W_e$; therefore updating each expert from precisely its routed
token batch is the *exact* gradient — sparse-expert training incurs no gradient approximation on
top of §1. (This also satisfies the one-forward-per-backward contract of the self-updating layer.)

Equal-active-compute sizing $h = \lceil 8d/(3k) \rceil_8$ makes top-$k$ SwiGLU experts match a dense
FFN's active MACs, so $E$ scales capacity at constant per-token compute. Witnesses: MoE beats the
dense FFN at equal active compute on real text at two scales (isolated FFN: 1.632 vs 1.655; full
model: 1.6176 vs 1.6243, monotonic in $E$) — `results/MOE_FFN.md`.

## 4. Systems realization (measured)

- **Fused update kernels** (Triton): packed per-row kernel (×17.6 isolated update); stacked-expert
  kernel — one launch per expert matrix over $[E, \text{out}, \text{in}]$, replacing ~15 elementwise
  passes; accepts bf16 gradients in-register. Kernel ≡ CPU reference up to one SR quantum on an
  $O(1)$ fraction (chunked fp reduction); 19 GPU kernel tests.
- **Loop-free MoE step:** grouped GEMMs (`torch._grouped_mm`) for forward/grad_x; per-expert grad_w
  via zero-padded bmm with a skew guard; batched update — **zero per-expert Python loops**. End-to-end
  MoE training throughput ×6.2 over the naive loop (27.2k → 167.8k tok/s, quality preserved).
- **Width-dependent dtype law (measured):** int8 tensor-core forward/update wins at $d \ge 768$
  (×2.05 fwd) and *loses* below (−27% at $d{=}512$: quant epilogue > GEMM saving). bf16 expert GEMMs:
  ×1.5–2.06 step at $d{=}1536$; quality parity at short horizon FAILED (+0.09 val) → gated (§6).
- **1.21B end-to-end:** single T4, 2.25 GiB peak, loss 9.16→2.05 (enwik8) / FineWeb val 6.29@849
  steps on 2×T4 DDP at ~730 tok/s (≈0.9× dense-class step speed; the prize is memory, not speed).

## 5. Positioning

| axis | BitNet b1.58 / ternary QAT | 8-bit Adam / GaLore / LoMo | MeZO / zero-order | **this work** |
|---|---|---|---|---|
| gradient | exact BP | exact BP | perturbation estimate | exact BP |
| master weight | fp16/32 (2–4 B) | fp32 or bf16 | full-precision | **none — 6-bit total state** |
| optimizer state | Adam (8 B) | 1–2 B / low-rank / none | none | **in-state (0 extra)** |
| activation memory | standard | standard | O(1) (forward-only) | O(1) (reversible, exact) |
| inference artifact | ternary | full-precision | full-precision | ternary (engine-ready) |

The distinguishing claim is the **elimination of the master weight and optimizer moments under
exact backpropagation**: 6 bits/parameter of trainable state, with Prop. 1 giving the latent view
that explains *why* it can converge (it is RMS-SGD on a hidden 6-bit accumulator, not sign-SGD on a
bare ternary weight).

## 6. Open questions and the falsifying experiment

1. **OPEN (decisive): convergence parity with AdamW at scale.** All quality evidence is small-scale
   (≤25M tokens per arm). A 6-bit accumulator bounds the representable update sum between flips; whether
   this caps late-training quality on ≥1B tokens is unknown. *Falsifier:* matched-token loss curves,
   counter vs dense-AdamW vs memory-matched 8-bit-Adam, ~50–150M params, ≥1B FineWeb tokens
   (harness: `notebooks/MN_convergence_colab.ipynb`, chained weekly-quota script).
2. **OPEN: bf16 gradient parity long-horizon** (short-horizon lag +0.09 val; gated flag).
3. **OPEN: reversible-coupling quality at scale** (RevNet-form ≠ plain residual; equal to itself,
   toy-scale gap vs plain observed).
4. Multi-GPU stacked-MoE requires adding the grad_w all-reduce (documented; single-GPU exact).

## 7. Reproducibility

Pure-PyTorch reference implementation (CPU/CUDA identical dynamics; kernels are drop-in), ~139 unit
tests including bit-exact CPU oracles for every kernel, all reported numbers generated by scripts in
`scripts/` + a public ZeroGPU harness; honest negative results are recorded alongside positives
(`results/VERIFICATION_RESULTS.md`: rejected M8, refuted router-starvation hypothesis, int4-on-T4
Amdahl rejection, failed short-horizon bf16 parity).
