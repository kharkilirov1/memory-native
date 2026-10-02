# What the new-math pilots showed

The floating structured operator and PAM arithmetic merit further investigation.
The tested all-counter factor recipe loses quality and needs a better learning
rule. None of these experiments demonstrates faster GPU pretraining.

The primary study ran eight scratch character-model variants on TinyShakespeare,
three seeds each, 400 steps × 4 sequences × 32 characters = 51,200 training
tokens per run. Each body LR was selected from three candidates on a separate
100-step, seed-90 validation run. Test text was evaluated only at the end.
See [full tables, provenance and plots](REPORT.md) and [raw index](SUMMARY.json).

| Operator | Mean held-out PPL ± seed SD | Body coefficients | Interpretation |
|---|---:|---:|---|
| Dense, width 64 | 12.80 ± 0.30 | 98,304 | Reference |
| Dense, width 32 | 13.95 ± 0.26 | 24,576 | Approximately matched total parameter budget |
| Dense counter | 13.80 ± 0.42 | 98,304 | Original finite-state training baseline |
| Floating BLAST | 13.56 ± 0.20 | 21,504 | Less body arithmetic; modest quality loss vs wide dense |
| Counter BLAST | 14.90 ± 0.44 | 21,504 | Combining counters with three factor stages needs work |
| Plain low-rank | 14.19 ± 0.34 | 20,736 | Approximately matched body coefficient budget |
| PAM, exact backward | 12.65 ± 0.28 | 98,304 | Quality retained in this small pilot |
| PAM, surrogate backward | 12.76 ± 0.29 | 98,304 | Quality retained in this small pilot |

PAM exact's paired test-CE difference from dense is −0.0116 ± 0.0041 nats per
character. This small three-seed result does not establish an advantage across
datasets or scales, or refute the published paper's other-task results favoring
surrogate derivatives.

## The two mathematical changes

BLAST uses blocks

\[
W_{ij}=U_i\operatorname{diag}(s_{ij})V_j^T.
\]

The implementation forms the projected input blocks once, mixes them per rank
coordinate, and expands output blocks once. It computes their derivatives
through the same factors without constructing full body matrices or their full
weight gradients. All-counter BLAST stores finite-state codes for U, V and S;
each local backward computes its input gradient before changing its own state.
Independent dense reconstruction tests verify the complete chain.

The primary factor geometry reduces body forward/backward pair work by
98,304 / 21,504 = **4.57×**. This is an arithmetic count, not a measured speedup.
At block width 16 and rank 8, square width-64 layers have global rank at most 32.
BLAST is a restricted operator family. Its advantage over plain rank-9 factors
at a similar coefficient budget is a useful signal, not free dense capacity.

PAM changes the scalar product itself. For positive normalized numbers
\(a=2^{e_a}(1+m_a)\) and \(b=2^{e_b}(1+m_b)\), it uses

\[
\operatorname{PAM}(a,b)=2^{e_a+e_b+c}(1+m_a+m_b-c),
\qquad c=\mathbf{1}[m_a+m_b\ge1].
\]

Signs are handled separately. Exact backward uses power-of-two slopes and
exponent shifts; surrogate backward applies PAM to the other operand and the
incoming derivative. This pilot replaces the six body linears per block;
attention, tied output head, normalization, loss and Adam remain conventional.
It is not a completely multiplication-free training system. PAM still requires
quadratic numbers of pair evaluations in an unrestricted dense layer.

BLAST and PAM are existing research methods: [BLAST](https://arxiv.org/abs/2410.21262)
and [Kosson and Jaggi](https://arxiv.org/abs/2305.17190). The experimental hypothesis
here is their usefulness in memory-native, especially counter-trained structured
factors; invention of these operators is not claimed.

## What failed, and what that means

On the representable floating-BLAST teacher, primary 400-step relative MSE was
0.0461 for floating factors and 0.4646 for counter factors. On a general dense
teacher it was 0.5460 and 0.8836 respectively. The latter task also exposes the
rank and shared-basis restrictions. PAM fit both teachers with about 0.00073 MSE.

The [1,000-step rank/C diagnostic](capacity_ablation/README.md) did not fix the
counter gap by increasing the horizon, changing C or expanding factor rank.
The expanded counter operator had rank 28 against a rank-16 teacher but still
had relative MSE 0.3899. Rank alone cannot account for the failure.

The same diagnostic found scales almost frozen: less than 0.91% change from
initialization with lr_scale=2e-4. These runs do not prove an intrinsic impossibility
of counter factors. They test one optimizer and a nearly fixed-amplitude ternary
factor family.

The [covariance/scale diagnostic](covariance_ablation/README.md) tests a different
factor update without changing the forward or input derivative:

\[
G\leftarrow0.95G+0.05X^TX/B,\qquad
\widetilde H=H(G+\lambda\max(\operatorname{tr}(G)/d,\epsilon)I+\epsilon I)^{-1}.
\]

The modified correlation enters both RMS/counter ticks and scale learning.
This local input-metric preconditioner is related to established preconditioning
methods; its use with discrete BLAST factors is the hypothesis being tested.
Eight predeclared, paired seed-0 runs of 1,000 steps gave:

| C / scale LR | Baseline relative MSE | Covariance ridge 0.1 | Covariance ridge 0.01 |
|---|---:|---:|---:|
| C=8 / 2e-4 | 0.5377 | 0.4890 | 0.4526 |
| C=2 / 2e-4 | 0.5338 | 0.6350 | 0.5875 |
| C=8 / 0.01 | 0.3791 | 0.3655 | Not run |

It helps C=8 modestly and hurts C=2. Faster scale learning helps more than the
default covariance change but still leaves a large gap to floating factors.
These are fixed-horizon diagnostics, not a held-out LR-selection exercise or
a language-model validation of the new update. The preconditioner adds 1,536
FP Gram bytes plus 672 bytes of metric configuration/counters to this small
operator: model state grows from 1,128 to 3,336 bytes. It also costs Gram and
Cholesky work. It cannot be presented as free six-bit optimizer state.

## Compute, state and scope

The floating BLAST CPU reference took about 5.8× the dense training time despite
fewer factor products. Many small matmuls, Python loops and intermediate tensors
dominate this small model. PAM took about 113× for exact and 129× for surrogate
backward. Three pinned, single-thread primary processes shared a host; these
wall times are implementation witnesses, not isolated hardware benchmarks.

The model/Adam/retained-gradient accounting is 1.605 MiB for dense, 0.434 MiB
for floating BLAST and 0.152 MiB for counter BLAST. It excludes activations,
decoded factor weights, intermediate tensors and allocator/transient peaks.
The counter reference uses uint8 codes, not packed six-bit storage. Extra FP
Gram state in the covariance diagnostic must also be counted.

No GPU, MXFP4, FP4, FP8 or integer hardware experiment was run. Numerical
portability to those formats, training quality at scale and their actual
throughput remain open questions.

The next mathematically useful direction is a structured operator with a
carefully conditioned factor update and explicitly controlled scale dynamics.
PAM provides a separate way to change product arithmetic while preserving
small-model learning. Any larger experiment should compare time to the same
held-out quality and include the operator's capacity restrictions and all state.

## Reproduce

Install `pip install -e '.[dev,research]'`, obtain the corpus bytes whose SHA-256
is recorded in the manifests, and use the commands in [REPORT.md](REPORT.md)
with fresh output directories. Existing run records are protected from
overwriting. The scripts, raw metrics, source hashes and PNG/SVG plots are
included in this branch; trained checkpoints are not included.
