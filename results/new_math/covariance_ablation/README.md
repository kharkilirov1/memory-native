# Small-factor input-metric pilot

Eight arms were declared before execution. Each trains the same 32×32 BLAST
student (block size 8, factor rank 4) on a floating BLAST teacher for 1,000
steps. Teacher seed 123, student seed 0, Gaussian-data seed 12345, batch-stream
seed 10000 and counter-rounding seed 100000 are fixed. There are 1,536 training
and 512 held-out examples, with batch size 32 and body learning rate 0.03.
The held-out relative MSE is evaluated at fixed steps, without early stopping
or choosing a learning rate. This is an auxiliary single-seed regression,
not language-model evidence.

For every small U/V/S factor, the new rule observes the uncentered input Gram
`G ← 0.95 G + 0.05 XᵀX/n` during its current training forward, starting at
identity, and supplies
`H̃ = H [G + (ridge × max(trace(G)/d, 1e-6) + 1e-6) I]⁻¹`
to the existing RMS counter update. An SPD Cholesky solve applies the metric
without forming an inverse. Both ticks and scale learning use `H̃`; forward
and the input derivative using the pre-update visible weights are unchanged.
The Gram uses a factor's local input width: 8 for V, 4 for S and U, rather
than the full operator width 32.

| C | Input metric | Scale LR | Train relative MSE | Held-out relative MSE | Scale relative L2 change |
|---|---|---:|---:|---:|---:|
| 8 | Baseline | 0.0002 | 0.527142 | 0.537664 | 0.267% |
| 8 | Ridge 0.1 | 0.0002 | 0.482633 | 0.488987 | 0.318% |
| 8 | Ridge 0.01 | 0.0002 | 0.442677 | 0.452634 | 0.372% |
| 2 | Baseline | 0.0002 | 0.519823 | 0.533777 | 0.296% |
| 2 | Ridge 0.1 | 0.0002 | 0.621197 | 0.635006 | 0.356% |
| 2 | Ridge 0.01 | 0.0002 | 0.590530 | 0.587535 | 0.388% |
| 8 | Baseline | 0.01 | 0.362561 | 0.379062 | 10.483% |
| 8 | Ridge 0.1 | 0.01 | 0.351032 | 0.365471 | 12.579% |

The input metric helps C=8 in this diagnostic and hurts C=2. Increasing scale
learning has a larger effect than the ridge-0.1 preconditioner. The original
small scale LR leaves factor amplitudes almost fixed, so poor results there
cannot alone establish an intrinsic failure of counter learning. These arms
still cannot uniquely separate representation limits, scale dynamics and
optimization noise. No configuration is promoted to the primary experiment.

The baseline persistent model state occupies 1,128 bytes; the preconditioned
version occupies 3,336 bytes. Added state is 1,536 bytes of fp32 Grams plus
672 bytes of configuration and observation counts across 12 factors. This
excludes activation/transient memory. Baseline training takes 4.65–4.96 s,
versus 6.56–6.90 s with the metric, on the single-core CPU PyTorch reference.
Covariance accumulation and Cholesky work are additional computation; this
pilot offers neither a speedup nor a six-bit total-training-state claim.

Input-metric preconditioning is established work. Relevant context is
[K-FAC: Optimizing Neural Networks with Kronecker-factored Approximate
Curvature (2015)](https://arxiv.org/abs/1503.05671) and
[Shampoo: Preconditioned Stochastic Tensor Optimization
(2018)](https://arxiv.org/abs/1802.09568). This rule is a one-sided input metric
followed by RMS and discrete counter dynamics, not a full K-FAC/Shampoo
implementation or a proven natural gradient. Its combination with counter
factors is the hypothesis tested here.

`manifest.json` records the predeclared protocol, exact source hashes,
environment and dataset/operator hashes. Per-arm files contain the complete
fixed-step curves, initial/final operator spectra, scale statistics and
counter activity. `completion.json` confirms unchanged source hashes and
identical initial counter state and sampled batches within each C/scale-LR
pair. The first source version's completed execution was superseded after
adding conversion guards and is excluded from these published records; its
results remain outside the repository at
`/workspace/new-math-covariance-aborted`.

Reproduce from the repository root with one CPU thread:

```sh
python scripts/new_math_covariance_ablation.py --output /tmp/covariance-pilot --steps 1000 --seed 0 --eval-every 50
python -m pytest tests/test_research_covariance.py -q -W error
```

The 18 tests cover the closed-form metric, current-batch EMA, pre-update input
derivative, the actual conditioned signal supplied to RMS, exact factory
state/RNG preservation, local statistic dimensions, checkpoint continuation
and refusal of incompatible configurations, pending graphs, lossy dtype
conversion, invalid observations and unsupported update modes.
