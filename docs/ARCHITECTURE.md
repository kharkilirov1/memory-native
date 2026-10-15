# Architecture

[Home](../README.md) / [Documentation](README.md) / Architecture

Memory Native combines two independent memory-saving mechanisms: finite-state training
coefficients and reconstruction of interlayer activations. Both have costs and constraints.

## 1. Coefficient state

For threshold `C`, a coefficient has a visible ternary value `t ∈ {-1, 0, +1}` and a
bounded accumulator `c ∈ {-(C-1), …, C-1}`. The code is

```text
z = (t + 1) * (2*C - 1) + c + (C - 1)
```

At `C=11`, there are 63 valid codes. `PackedRMSCounterLinear` packs four codes into
three bytes. The ordinary row-scale forward uses `weight = scale * t`; the residual
counter drives subsequent updates. Other experimental layers have separate contracts.

The counter code replaces a separate coefficient master copy and update accumulator.
It does not replace every floating-point value in the model.

## 2. Updates and RMS

The counter layer is an `nn.Module` with buffers rather than ordinary weight parameters.
Its custom backward computes input gradients using the pre-update weights and then
updates the coefficient state. No external `optimizer.step()` is needed for those codes.
Ordinary trainable parameters in a mixed model still need their own optimizer.

`RMSCounterLinear` already uses **one FP32 second-moment value per output row**, rather
than a full weight-shaped RMS matrix. For a `4096 × 4096` layer:

| Second-moment representation | Storage for the statistic only |
|---|---:|
| Full FP32 matrix | 64 MiB |
| Factored row + column vectors | 32 KiB |
| Current row-only RMS | 16 KiB |

A factored RMS is not a memory-saving replacement for the current row-only statistic.
It would change normalization and requires a controlled quality experiment before adoption.
These numbers exclude coefficient codes, scales, other buffers and temporary tensors.

## 3. Single-pass reversible backward

The coupling splits channels into two halves:

$$y_1 = x_1 + F(x_2), \qquad y_2 = x_2 + G(y_1).$$

Given the final output and its upstream gradients, backward proceeds as follows:

1. Evaluate `G(y1)` with autograd enabled. Reconstruct `x2 = y2 - G(y1)` without
   differentiating this subtraction. Reuse the same `G` graph to compute its local VJP.
2. Accumulate `dx1 = dy1 + J_G(y1)^T dy2`.
3. Evaluate `F(x2)` once. Reconstruct `x1 = y1 - F(x2)` without differentiating the
   subtraction, and reuse the `F` graph to compute `dx2 = dy2 + J_F(x2)^T dx1`.
4. Return input gradients and the parameter gradients from the two local VJPs.

Each branch is evaluated **once in forward and once in backward**. The previous inverse
path evaluated each branch twice during backward: once for reconstruction and again
for differentiation. Halving those recompute calls does not halve the complete training
cost: local backward work and counter updates remain.

Reconstruction precedes the corresponding local VJP because counter layers update their
state during that VJP. Branches must be deterministic and independent of forward-time
state changes. Shared counter modules remain unsupported.

| Container | Saved interlayer activations | Reconstruction |
|---|---|---|
| `ReversibleCouplingBlock` | One block output | Float inverse |
| `ReversibleSequential` | One output per block; O(L) | Float inverse |
| `ReversibleSequence` | One final output; O(1) in depth | Float inverse |
| `ReversibleSequence(anchor_every=A)` | Anchors plus local graphs; O(L/A + A) | Forward from anchors |

O(1) concerns activation memory **as depth varies**, not total model memory, sequence
length or the temporary working set of an individual branch. Float reconstruction is
approximate. Rounding deltas onto a grid does not guarantee bitwise inversion at arbitrary
magnitudes; that change from the NumPy prototype was not adopted.

Covered tests compare nonlinear input/parameter gradients with stored-activation autograd,
check frozen parameters, count branch calls, and compare packed-counter updates. See
[the witness](../results/REVERSIBLE_SINGLE_PASS.md). Higher-order differentiation is not
supported. The changed recompute order can change RNG consumption for quantized saved
activations; historical stochastic training trajectories are not promised to be identical.

## 4. Count the complete memory budget

`memory_report(model)` reports registered model parameters and buffers. Its packed-code
estimate is separate from actual persistent bytes. It excludes external optimizer state,
activation graphs and temporary scratch; use measured CUDA peak allocation for a full step.

The ordinary forward decodes weights to dense tensors. Default correlation can materialize
a dense weight gradient; row tiling bounds its temporary size. Strict from-IO kernels avoid
that buffer but have their own throughput trade-offs. Six-bit persistent codes therefore
do not imply a six-bit-per-coefficient training peak or proportional wall-time savings.

## 5. Integration contract

- Use one forward followed by one backward; do measurements under `torch.no_grad()`.
- Do not reuse counter modules, accumulate multiple forward graphs, apply external
  checkpointing, or capture the self-update with `torch.compile` without a scheduler.
- No dropout or forward-time state mutation inside reversible branches. Do not mutate
  weights between forward and backward.
- The custom data-parallel path synchronizes counter updates explicitly; ordinary DDP
  does not discover these buffer updates through parameter gradients.
- Keep hardware-specific results scoped to their recorded shape, runtime and device.

[Project status](../PROJECT_STATUS.md) · [Reproduce](../REPRODUCIBILITY.md) ·
[Results](../results/README.md)
