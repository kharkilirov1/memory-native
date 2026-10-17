# Experimental counter stability controls

[Documentation](README.md) / Counter stability

These controls are opt-in. They correct specific numerical mechanisms; they do not
establish better language-model quality or faster training. See the
[CPU ablation](../research/results/COUNTER_STABILITY.md) for positive and negative results.

```python
import torch
from memory_native import PackedRMSCounterLinear, configure_rounding_streams

model = torch.nn.Sequential(
    PackedRMSCounterLinear(
        32, 32, C=11,
        rms_bias_correction=True,
        scale_rebase="threshold",
    ),
)
# Assign after construction/conversion, before loading a checkpoint.
configure_rounding_streams(model, seed=42)
# model.load_state_dict(checkpoint) restores stream positions and moment counts.
model(torch.randn(16, 32)).square().mean().backward()
```

## RMS startup correction

`rms_bias_correction=True` divides the row EMA by `1 - beta**n` before taking
its square root and applying the existing epsilon floor. `n` counts actual updates
to that row, not forwards or global training steps. Tiling and update decimation
therefore do not artificially age the moment estimate. `expm1` avoids cancellation
for beta close to one. Scale learning still uses the raw gradient.

The option supports `RMSCounterLinear`, `PackedRMSCounterLinear` and its Triton
forward subclass, with exact or proxy RMS. Lagged RMS is rejected because its first
denominator contains no observed moment; bias correction alone does not define a
useful cold start for that mode. Group-scale layers do not expose this option yet.

Storage: an additional int64 count per output row, 32 KiB at 4096 output rows.
The existing row moment remains 16 KiB. No full second-moment matrix is introduced.
Counts are checkpointed and reset by dense/counter imports. Loading an old checkpoint
into the corrected mode fails if the counts are absent: their history cannot be
inferred from the moment. Use the old configuration to resume old training, or
explicitly import weights to start a new optimizer history.

## Cumulative threshold rebasing

`scale_rebase="threshold"` retains `s_base`, the scale at which the stored counter
is calibrated. New ticks use that retained basis. A row rebases when
`abs(s_base / s_new - 1) > rebase_threshold`, default `0.5 / C`.
Consequently, many individually small changes eventually cause a rebase.

A row also rebases before stochastic rounding could produce a carry. This ensures
the ternary transition and residual subtraction use the current scale. Away from
rebases, the pending physical residual is `s_base * c / C`, rather than `scale * c / C`.
The visible forward remains strictly ternary. This mode is not a residual-readout
implementation and preserves neither an arbitrary dense donor nor a whole latent
weight when its ternary scale changes.

The existing `s_base` buffer suffices; no additional persistent row vector is needed.
Temporary operations remain bounded by `tile_rows`. Row-wide carry pressure can
trigger frequent rebases, so this is not a guarantee that every inactive synapse
will remain untouched. Do not switch rebase modes on live optimizer state.

Both new dynamics options bypass the packed fused-update kernel and the compiled
fast chain. They use the PyTorch reference path on CPU/CUDA. GPU performance and
CUDA numerical behavior have not been validated for these options.

## Named rounding streams

`configure_rounding_streams(model, seed)` assigns a 32-bit ID derived from SHA-256
of the seed and each supported module's stable path. Duplicate IDs are rejected.
The helper resets the stream positions, so call it only when starting new streams.
It supports compact/row-RMS counters and packed group-scale counters. Unpacked
group-scale counters and MLX are outside its scope.

The existing integer hash generates the rounding draws from stream position and
element index. Packed kernels receive the mixed seed through their existing API;
their math is unchanged. The CPU row reference uses hash rounding when configured.
IDs never depend on allocation addresses, process IDs or distributed ranks.

Each configured layer saves two int64 metadata values (16 tensor-payload bytes) in
its state dict: ID and position. They are Python integers during execution, avoiding
a per-step device-to-host read. Checkpoints restore them automatically, even without
calling the helper on the receiving model. Loading a historical checkpoint without
these keys restores legacy rounding. Dense/group-state imports reset the position
while preserving the configured identity.

Exact replay requires the same backend, tiling, update schedule, initial state and
hyperparameters. Lazy rebase consumes two draws per tile; eager/threshold consume
one. Global RNG operations such as activation quantization, stochastic ternary
pulses or dropout are not controlled by this helper. Matching streams alone cannot
make differing inputs or gradients produce identical updates across DDP replicas.
Statistical independence of streams is not claimed; the 32-bit position mixing
repeats after 2**32 draws. No multi-process DDP or GPU execution was available for
the new stream tests; CPU kernel-reference scheduler tests cover seed plumbing.

Default construction keeps historical dynamics and rounding behavior. Save the
constructor configuration with a checkpoint, as for the existing learning rates
and RMS modes; a state dict alone does not describe the training recipe.
