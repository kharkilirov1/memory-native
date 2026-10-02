# Training Without Master Weights: a 6-bit Finite-State Synapse with Optimizer-in-State

*Preprint draft v0.2 — formalization of the memory-native training method implemented in this
repository. Empirical claims refer to reports in `results/`; a report is not necessarily a complete
public reproduction artifact. Claims not yet validated at scale are marked OPEN.*

## Abstract

Conventional Adam training with fp32 weights and moments uses 12 bytes per parameter, or 16 bytes
including a materialized fp32 gradient buffer; ternary QAT commonly retains a master weight. We
formalize a **finite-state counter synapse** in which the *entire* per-parameter training state is a
single 6-bit code: a visible ternary weight and a bounded stochastic-rounding accumulator, with
additional shared per-row scales and RMS statistics. The reference update consumes backpropagated
gradients in-place; optional low-bit paths estimate the update correlation. Away from saturation,
we derive an expected latent-weight update consisting of an RMS-normalized gradient step **and a
visible-weight displacement from scale learning**. It reduces to unbiased latent RMS-SGD when
the scale is fixed. The visible ternary readout's residual is stored in state; the shipped rebase
preserves that residual, not the entire latent weight, when the scale changes. Composed with reversible
coupling blocks (O(1)-in-depth activation memory), a counter-state mixture-of-experts, and optional
int8/bf16 compute, the method has a historical report of training 1.21B counter coefficients on
a single 14.6 GiB GPU in 2.25 GiB peak. The primary log for that run is absent from the public
repository; the report is evidence of a claimed fit and short training run, not a matched-quality
or throughput comparison. Convergence parity with AdamW at scale remains open; we specify the
falsifying experiment.

---

## 1. The finite-state counter synapse

### 1.1 State space

Fix an integer $C \ge 1$ ($C=11$ in the reported packed experiments; the unpacked reference defaults
to $C=8$). The six-bit format requires $3(2C-1) \le 64$, hence $1 \le C \le 11$.
Each scalar parameter is a pair

$$\sigma = (t, c), \qquad t \in \{-1, 0, +1\},\quad c \in \{-(C\!-\!1), \dots, C\!-\!1\},$$

encoded injectively into one code

$$\mathrm{enc}(t,c) = (t+1)(2C-1) + (c + C - 1) \in \{0, \dots, 3(2C-1)-1\}.$$

For $C=11$: $3 \cdot 21 = 63$ states, $\lceil \log_2 63 \rceil = 6$ bits; four codes pack into three
bytes (0.75 B/parameter persistent). A weight matrix $W \in \mathbb{R}^{n_\text{out} \times n_\text{in}}$
carries per-row scales $s \in \mathbb{R}^{n_\text{out}}_{>0}$ and per-row second-moment estimates
$v \in \mathbb{R}^{n_\text{out}}_{\ge 0}$ — $O(n_\text{out})$ fp32, amortized over each row's
$n_\text{in}$ coefficients. This overhead vanishes per coefficient only as fan-in grows.

**Visible weight (readout).** The forward pass uses only the ternary component:
$$W^{\text{vis}}_{oi} = s_o\, t_{oi}, \qquad y = x\, (W^{\text{vis}})^\top .$$
For the strict ternary readout ($\texttt{residual\_alpha}=0$ in group-scale layers), the counter
$c$ is invisible to the forward — verified exactly (zeroing $c$ changes the forward by
$0.0$; `scripts/fusion_invariants.py`). This is what makes the forward a *ternary* mixed-input GEMM
(BitNet-inference class) rather than an exotic 6-bit one.

### 1.2 Latent-weight view

Define the **latent position** and latent weight
$$u_{oi} = \Big(t_{oi} + \frac{c_{oi}}{C}\Big) s_o .$$
Then $W^{\text{vis}}_{oi} = u_{oi} - s_o \frac{c_{oi}}{C}$ with residual bounded by
$|s_o c_{oi}/C| \le s_o \frac{C-1}{C} < s_o$: the visible weight is a ternary quantization of the
latent weight whose readout residual is **stored exactly in the state** rather than discarded.
The update accumulation is an error-feedback mechanism, subject to stochastic-rounding noise,
bounded state, and the scale displacement derived below; it is not an exact floating-point
master-weight trajectory encoded in six bits.

### 1.3 Update rule (the automaton)

Let $g$ be the backpropagated weight-gradient correlation (formed per layer, transiently).
For a full-precision linear forward and saved activation, $g = \nabla_W \mathcal{L}$; low-bit
compute and saved activations have the qualifications in Proposition 2. One direct-pulse,
eager-rebase update with learning rates $\eta, \eta_s$, EMA factor $\beta$, floor $\varepsilon$
("exact" RMS mode), and optional local row-norm clipping threshold $h$ is:

1. **Row statistic:** $\bar g^2_o = \tfrac{1}{n_\text{in}} \sum_i g_{oi}^2$; $\quad v_o \leftarrow \beta v_o + (1-\beta)\bar g^2_o$; $\quad D_o = \max(\sqrt{v_o},\, \varepsilon)$.
2. **Scale learning:** $\gamma_o = \tfrac{1}{\sqrt{n_\text{in}}} \sum_i g_{oi} t_{oi}$; $\quad s'_o = \mathrm{clip}(s_o - \eta_s \gamma_o;\ 10^{-5}, 10)$.
3. **Tick:** let $a_{oi}=g_{oi}/D_o$. If $h>0$, replace $a_o$ by
   $a_o\min(1,h/\max(\|a_o\|_2,10^{-30}))$. Then
   $\Delta_{oi} = -\eta\, a_{oi} C/s'_o$ (counter units); rebase
   $\tilde c_{oi} = c_{oi}\, s_o/s'_o$. Scale learning uses the raw $g$, not $a$.
4. **Stochastic rounding:** $\hat c_{oi} = \mathrm{SR}(\tilde c_{oi} + \Delta_{oi})$, where $\mathrm{SR}(x) = \lfloor x \rfloor + \mathrm{Bern}(x - \lfloor x \rfloor)$, so $\mathbb{E}[\mathrm{SR}(x)] = x$.
5. **Carry / saturation:** $k = \mathrm{trunc}(\hat c / C)$; $\ t' = \mathrm{clip}(t + k; -1, 1)$; remainder $r = \hat c - kC$, and if the clip was active, $r = \mathrm{sign}(\hat c)(C-1)$; $r$ clamped to $\pm(C-1)$. New state $(t', r)$, new scale $s'$.

The optimizer **is** steps 1–5: there is no other per-parameter state, no `.grad` retained, no
parameter registered with an outer optimizer. The update executes inside the layer's backward;
the packed GPU path can fuse it into one kernel launch (§4).

### 1.4 Expected dynamics

**Proposition 1 (expected latent dynamics of the shipped eager rebase).** Condition on the old
state $(t,c,s,v)$ and supplied gradient $g$, so $a$ and the actual clipped scale $s'$ are fixed.
If neither possible stochastic-rounding outcome saturates ($|t+k|\le1$ for every outcome with
positive probability), the direct-pulse, eager-rebase update satisfies
$$\mathbb{E}_{\mathrm{SR}}[u'_{oi}\mid t,c,s,v,g]
  =u_{oi}-\eta a_{oi}+t_{oi}(s'_o-s_o).$$
The deviation from this conditional mean is bounded in magnitude by one counter quantum
$s'_o/C$. With fixed scales and no local clipping, the identity reduces to unbiased latent
RMS-normalized SGD. With learned scales it includes a deterministic scale displacement;
Proposition 1 is a one-step identity, not a convergence theorem.

*Proof.* Write $z=c\,s/s'-\eta a C/s'$ and $\hat c=z+\xi$, with
$\mathbb{E}_{\mathrm{SR}}[\xi]=0$ and $|\xi|<1$. Without saturation, the carry and remainder obey
$t'+r/C=t+\hat c/C$. Hence
$$u'=s'(t+\hat c/C)=s't+sc/C-\eta a+(s'/C)\xi
   =u-\eta a+t(s'-s)+(s'/C)\xi.$$
Taking the conditional expectation proves the claim. The rebase preserves $sc/C$, the residual,
while the visible component changes from $st$ to $s't$. This algebra also holds when the scale
clip is active, provided $s'$ denotes the **clipped** scale and carry does not saturate. It does
not justify replacing the clipped scale step by its unclipped formula. $\square$

For example, $C=11$, $t=s=v=1$, $c=0$, $g=0.1$, $\beta=0.9$, $\eta=0.1$,
$\eta_s=0.2$ give $v'=0.901$, $s'=0.98$. Neither SR outcome carries or saturates.
The conditional latent mean is $0.9694649$, not $1-0.1(0.1/\sqrt{0.901})=0.9894649$:
the difference is $t(s'-s)=-0.02$. The analytic regression in `tests/test_latent_dynamics.py`
enumerates both SR outcomes, including fixed-scale carries and clipped variants.

Saturation projects the post-rounding latent value to the representable boundary
$\pm s'(2C-1)/C$. It introduces a further bias: SR followed by this projection is not generally
unbiased. The remainder rule retains pressure at the boundary rather than wrapping.

**Proposition 2 (unbiased raw low-bit correlation, conditional on operands).** For fixed
operands $\Delta,X$, assume $Q_a,Q_b$ are conditionally independent and each is unbiased
for its operand. In exact arithmetic,
$$\mathbb{E}[Q_a(\Delta)^\top Q_b(X)\mid\Delta,X]=\Delta^\top X.$$
This follows by factoring each expected product in the sum. Per-column symmetric int8/int4 SR
quantizers satisfy this identity when their range covers the operands; numerical rounding and
integer-accumulator overflow must still be controlled in an implementation. Tests
`test_int8_compute.py` and `test_actquant.py` check the quantizer/correlation numerics.

This identity applies to the **raw correlation**, not to the whole adaptive optimizer.
RMS statistics square the sampled correlation, and normalization, local clipping, and saturation
are nonlinear; an unbiased $\hat g$ does not imply
$\mathbb{E}[\hat g/\sqrt{\hat v}]=g/\sqrt{v}$ or an identical expected counter update.
Likewise, $\mathbb{E}[\Delta^\top Q(X)\mid X,\Delta]=\Delta^\top X$ for a saved activation
requires $\Delta$ to be conditionally independent of its quantization noise. Saving a quantized
copy while computing the isolated layer's forward from the original $X$ permits that condition,
but it must be checked for the whole backward graph. In general a downstream gradient may
depend on quantization noise. The presaved int8 path is conditionally unbiased for
$\Delta^\top\hat X$ given the saved $\hat X$; fresh activation rounding across steps alone does
not establish unbiasedness for the original gradient or the training trajectory.

**Determinism variants.** Two SR families are implemented: `torch.rand` SR (supports the one-pass
"lagged" mode) and a deterministic hash-SR (MurmurHash of the element index XOR seed) used by the
fused kernels. For a fixed seed, hash-SR is deterministic; the ideal unbiased-SR calculation
above models independent uniform draws, not an arbitrary fixed sequence of hash values.
Hash-SR removes dependence on the global RNG stream. Replicas' packed states remain byte-identical
when they start identically and use identical reduced gradients, optimizer statistics,
hyperparameters, element indexing, and seed/update schedules
(verified empirically: 0 differing bytes across ranks with different per-rank data;
`test_ddp_decimation.py`).

### 1.5 Memory accounting

**Proposition 3 (packed counter-state footprint).** For $N$ coefficients in packed counter
linear layers, codes use $0.75N$ bytes plus per-row packing padding. Shared fp32 buffers and
padding add $O(\text{rows})$ bytes. This accounts for those layers, not activation memory,
temporary decoded weights/gradients, or a model's remaining floating-point parameters.
The unpacked reference uses one byte per code rather than six-bit storage.

For comparison, fp32 weights plus Adam moments $(m,v)$ cost $12N$ bytes: a nominal 16× ratio
against the code bytes alone. Adding a materialized fp32 gradient buffer gives $16N$ bytes and
a nominal 21.3× ratio; that buffer is not permanent when released between steps. Bf16 weights
plus two eight-bit Adam moments nominally cost $4N$ bytes before gradients, block scales, and any
extra master copy, a 5.3× ratio. These are storage calculations, not peak-VRAM measurements.
`results/SCALE_1B.md` reports **871.7 MiB counter model state versus an 18.0 GiB dense estimate**
and a **2.25 GiB counter training peak**. The historical primary log is missing, and dense OOM
does not provide a matched measured training peak or matched-quality result.

## 2. Activation memory: reversible coupling with anchors

Blocks are additive couplings $y_1 = x_1 + F(x_2),\ y_2 = x_2 + G(y_1)$ with exact inverse
$x_2 = y_2 - G(y_1),\ x_1 = y_1 - F(x_2)$ (RevNet). The backward reconstructs inputs from outputs,
then recomputes the local forward under autograd. In exact arithmetic, with unchanged deterministic
$F,G$, gradients equal those of standard backpropagation through the same architecture; finite
precision reconstruction can accumulate error. Activation memory is $\Theta(1)$ in depth instead of
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
- **1.21B end-to-end:** the historical single-T4 report gives 2.25 GiB peak and validation loss
  2.95 at step 250, 2.05 at step 2000, and 2.10 at final evaluation on enwik8; its initial
  **training** loss was 9.16 (`results/SCALE_1B.md`, primary log absent). A separate 2×T4 DDP
  report gives FineWeb validation loss 6.29 at 849 steps and ~730 tok/s. These runs do not
  establish matched-quality throughput parity with a dense baseline.

## 5. Positioning

| axis | BitNet b1.58 / ternary QAT | 8-bit Adam / GaLore / LoMo | MeZO / zero-order | **this work** |
|---|---|---|---|---|
| gradient | BP / quantization surrogate | BP | perturbation estimate | BP; optional low-bit correlation |
| master weight | fp16/32 (2–4 B) | fp32 or bf16 | full-precision | **none — 6-bit total state** |
| optimizer state | Adam (8 B) | 1–2 B / low-rank / none | none | **counter in code + shared row/group statistics** |
| activation memory | standard | standard | O(1) (forward-only) | O(1) for reversible architecture |
| inference artifact | ternary | full-precision | full-precision | ternary at strict readout; export available |

The proposed contribution is the **joint six-bit encoding of a ternary readout and update
accumulator**, with only shared row/group optimizer statistics and no per-weight master or Adam
moments. Proposition 1 gives its one-step latent dynamics; it does not establish convergence or
quality parity. Master-weight elimination itself has prior work.

**Related work and scope.** ECO [1] applies updates directly to quantized weights and injects
quantization error into optimizer momentum. It removes master weights while retaining optimizer
state; this project's finite-state accumulator uses a different memory/accuracy tradeoff. WAGE
[2] discretizes weights, activations, gradients, and propagated errors for integer training and
inference, so low-bit training state and stochastic update quantization are not new in isolation.
BitNet b1.58 [3] motivates the ternary inference readout; equal inference bits do not imply equal
training dynamics or model quality. RevNet [4] supplies the additive reversible-coupling mechanism,
which requires an appropriate architecture rather than applying unchanged to an arbitrary donor.
QLoRA [5] freezes a quantized pretrained base and trains low-rank adapters. It is a practical
memory-limited fine-tuning baseline, despite having a different trainable parameterization from
full-weight counter recovery. A useful comparison must match donor, data, tokens, hardware, and
evaluation, and report quality, peak memory, and elapsed time.

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
5. **OPEN: fine-tuning utility versus QLoRA and full-weight baselines.** Counter recovery currently
   lacks a matched comparison measuring quality, peak memory, and time on the same pretrained donor.

## 7. Reproducibility

The pure-PyTorch reference runs on CPU and CUDA. Tests compare bit packing, deterministic updates,
and kernel numerics; CPU checks do not validate GPU execution, and floating-point reductions may
change rounding decisions across implementations. Scripts and notebooks reproduce portions of
the experimental work, but not every historical report has a public raw log, checkpoint, or
pinned environment. Negative results are recorded alongside positives
(`results/VERIFICATION_RESULTS.md`: rejected M8, refuted router-starvation hypothesis, int4-on-T4
Amdahl rejection, failed short-horizon bf16 parity). `tests/test_latent_dynamics.py` supplies an
analytic witness for Proposition 1; it does not require Monte Carlo sampling or a GPU. See
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for public artifact availability and reproduction limits.

## References

1. Mahdi Nikdan, Amir Zandieh, Dan Alistarh, and Vahab Mirrokni. **ECO: Quantized Training without
   Full-Precision Master Weights.** 2026. [arXiv:2601.22101](https://arxiv.org/abs/2601.22101).
2. Shuang Wu, Guoqi Li, Feng Chen, and Luping Shi. **Training and Inference with Integers in Deep
   Neural Networks.** ICLR 2018. [arXiv:1802.04680](https://arxiv.org/abs/1802.04680).
3. Shuming Ma et al. **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.** 2024.
   [arXiv:2402.17764](https://arxiv.org/abs/2402.17764).
4. Aidan N. Gomez, Mengye Ren, Raquel Urtasun, and Roger B. Grosse. **The Reversible Residual
   Network: Backpropagation Without Storing Activations.** NeurIPS 2017.
   [arXiv:1707.04585](https://arxiv.org/abs/1707.04585).
5. Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. **QLoRA: Efficient
   Finetuning of Quantized LLMs.** NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
