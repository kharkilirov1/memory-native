# Binary counter factors with geometric initialization

Research date: 2026-10-08/09. Base: `851c91ed4bb088424cbf05af8ca83a2bf1c76844`.
Status: opt-in CPU research. Production behavior is unchanged.

## Implementation and attribution

This is an independent implementation from equations, not copied Samsung code.
The binary factor architecture and Joint-ITQ are credited to
[LittleBit](https://arxiv.org/abs/2506.13771) and
[LittleBit-2](https://arxiv.org/abs/2603.00042).
This is NOT the authors' complete SmoothSign/residual-factor/distillation recipe.

`src/memory_native/research/binary_factor.py` implements

    W_hat = diag(h) B_left diag(ell) B_right diag(g), B_left,B_right in {-1,+1}.

SVD initialization splits square-root singular values between factors. A shared
orthogonal rotation preserves their continuous product but changes binary error.
Joint-ITQ alternates signs and orthogonal Procrustes. `guarded=True` selects among
SVD, seeded random rotation and final Joint-ITQ using actual weight reconstruction
or supplied calibration activations, never held-out test inputs. It guarantees
only no worse initializer-selection score than SVD, not downstream quality.
This is a separate two-factor operator; arbitrary rotations would generally destroy
BLAST's diagonal middle-factor structure.

`PackedBinaryLinear` stores codes z in [0,63], four per three bytes. Its latent
q=(z-31.5)/16 reads as +1 for z>=32, otherwise -1. The counter update is

    v_new = beta*v + (1-beta)*mean_row(gradient**2)
    position = z - 16*lr*gradient/max(sqrt(v_new),eps)
    z_new = clip(stochastic_round(position),0,63)

Ideal SR without clipping preserves the expected latent row-RMS step. This is
not a convergence theorem. This 64-state binary format is NOT compatible with
the original 63-state ternary (t,c) format. The scales h/ell/g are normal trainable
Parameters. There are no persistent floating-point master factors.

The factorized forward/backward avoids virtual dense W and dW, but decoded signs,
factor gradients and temporary proposals exist. Initialization uses dense SVD and
candidate matrices. Persistent factor codes cost 0.75*r*(N+K) bytes, plus RMS,
metadata, shared scales and their optimizer state. The ordinary FP model shell
also remains. No peak-RAM/VRAM or GPU-performance claim is made.

Input gradients use old signs before mutation. One forward/backward per update;
CPU first-order eager only. CUDA/AMP, DDP, sharing/accumulation and higher-order
derivatives are unsupported. No whole-model rollback is promised after a later
backward error. Local stochastic-rounding streams and controls are checkpointed.

`export_binary_factors` and `BinaryFactorInference` store one bit per sign plus
padding and FP32 scales. This is an inference-only artifact, not a resumable
checkpoint, Samsung file format or optimized kernel. Training and export use the
same hard readout: no hidden alpha-residual is removed at deployment.

## Matched matrix experiment

48x64 target, rank 8, 600 updates, batch 64. Independent Gaussian inputs split
1536/256/512 into train/val/test. Tuning seeds 0-2, LR grid [.001,.005,.02].
LR selection averages validation over BOTH initializers, then holds LR fixed
across them: native .005, QAT Adam/RMS .02; shared-scale Adam .003.
Confirmation seeds 20-31 were declared before their runs.

| Rule | Init | Spectral target test relative MSE | Binary-factor teacher test relative MSE |
|---|---|---:|---:|
| Native | SVD | 0.460317 | 0.171893 |
| Native | guarded geometry | 0.394582 | 0.064304 |
| QAT Adam | SVD | 0.373490 | 0.003298 |
| QAT Adam | guarded geometry | 0.385726 | 0.012705 |
| QAT row-RMS | SVD | 0.426425 | 0.011963 |
| QAT row-RMS | guarded geometry | 0.412661 | 0.018751 |
| Frozen signs | SVD | 0.456963 | 0.361454 |
| Frozen signs | guarded geometry | 0.371000 | 0.059671 |

Geometry improves native spectral-target MSE by 14.28%, winning 10/12 pairs.
Paired bootstrap absolute-gain interval: [.02949,.10743], 20000 resamples.
But frozen signs beat native+geometry, and QAT Adam is much stronger on the
representable binary teachers. QAT can prefer SVD after learning. This is not
counter superiority or a large-model result. Isolated-layer persistent tensors
plus optimizer: native 2380 B, QAT Adam 12436 B; native is slower on CPU.

## Small pretrained byte-GPT recovery

Two blocks, d=64, four heads, context=64, batch=8, byte vocabulary=256, tied
embedding/head. Dense teacher trained 1200 steps (614400 sampled byte positions).
Only four FFN matrices are replaced by rank-16 factors (20480 binary coefficients).
Attention, embeddings/head, norms and shared scales remain FP and train.

Data: 146 top-level Python 3.13.5 stdlib source files, split by whole-file SHA(name).
Train/val/test: 3961479/354863/226768 bytes. Windows cannot cross file boundaries.
Evaluation uses 8192 val and 32768 test positions in fixed sampled windows that
may overlap. This is not full-corpus PPL. Exact manifests/tensors are in the
conversation research archive; a different local stdlib changes the benchmark.

Tuning seed 91, 300 steps, 12 configurations. Factor LR .001 for both native and
QAT Adam, scale LR .003, FP-shell LR .0003. Protocol fixed before seeds 40-42.
Each arm runs 600 recovery steps (307200 positions), KL(teacher||student)+.3CE at
T=1. Our QAT control uses clipped STE, not LittleBit's complete training recipe.
No test-based selection. Frozen signs are an essential control.

| Rule | Init | Final CE | exp(mean CE) | Training seconds | Persistent model+optimizer tensor bytes |
|---|---|---:|---:|---:|---:|
| Dense continued recovery | none | 2.084075 | 8.037157 | 7.321 | 1433168 |
| QAT Adam | SVD | 2.136684 | 8.471303 | 8.070 | 911504 |
| QAT Adam | guarded | 2.124679 | 8.370211 | 7.844 | 911504 |
| Native | SVD | 2.133990 | 8.448506 | 12.477 | 681200 |
| Native | guarded | 2.128088 | 8.398796 | 12.463 | 681200 |
| Frozen signs | SVD | 2.128012 | 8.398152 | 8.247 | 681200 |
| Frozen signs | guarded | 2.124931 | 8.372318 | 8.122 | 681200 |

Each row averages three paired seeds. Native+geometry is +.003409 nat/byte,
+.342% perplexity versus matched QAT, with 25.27% less persistent state but 58.9%
more CPU training time. Three seeds do not establish equivalence. Continued dense
is stronger. Frozen signs slightly beat native, so recovery cannot be credited
to counter sign changes. Native+geometry averages 173 flips across 600 steps;
all 8 banks update 600 times. Frozen controls retain banks for matched layout;
a stripped frozen artifact could use less memory.

Timing includes teacher forward, excludes initialization, batching and evaluation.
Persistent bytes exclude teacher, retained gradients, activations and temporary
storage. The 5.23x isolated-layer state ratio is NOT the whole-model ratio.

## Tests, artifacts and reproducibility

61 feature/runner cases passed locally. In the complete pinned upstream archive,
128 targeted tests passed, including 67 existing structured/covariance/packing/
latent-dynamics/carry tests. This is not the entire upstream suite.

Core blob: `863638eb5c235c6c13749821f3da592cd3d77569`.
Core SHA256: `a9b6a974739e311785b81dd54cc4190c9eced57767837175325ba0fe5daa8491`.
This core was unchanged throughout the experiments. A copied FFN expression in
the published runner was corrected to the tested source, final text-runner blob
`b4c2d8be74a9e5c445615c3db8de61168f92b241`. Both runners have execution smoke tests.
Runner bookkeeping/data-precast changes do not change the learning rule. Timed-out
runs were restarted with identical seeds, not counted as additional evidence.

The archive retains all raw JSON, protocols, teacher and native seed-40 checkpoint.
Loading the native checkpoint exactly reproduces CE 2.1285827104002237. A unit
test independently verifies counter+scaleAdam continuation. This is a small byte
model, not a useful chatbot. The pyproject fix removes unsupported authors.role
while retaining the sponsor note; production algorithm defaults are unchanged.

```python
from memory_native.research.binary_factor import (
    NativeBinaryFactorLinear, BinaryFactorInference, export_binary_factors,
)
import torch

source = torch.nn.Linear(64, 48, bias=False)  # use a pretrained CPU layer here
layer = NativeBinaryFactorLinear.from_linear(
    source, rank=8, method="joint_itq", guarded=True, lr=.001, seed=0,
)
optimizer = torch.optim.AdamW(layer.parameters(), lr=.003, weight_decay=0.)
x = torch.randn(16, 64)
optimizer.zero_grad(set_to_none=True)
loss = (layer(x) - source(x).detach()).square().mean()
loss.backward()       # counter factors update
optimizer.step()      # shared scales update
inference = BinaryFactorInference(export_binary_factors(layer))
```

```sh
PYTHONPATH=src python -m pytest -q tests/test_binary_factor.py tests/test_binary_factor_runners.py
python scripts/binary_factor_witness.py --phase confirm --start 20 --seeds 12 --steps 600 \
  --selection results/binary_factor_20261009/matrix_protocol.json --output /tmp/binary-new-matrices
python scripts/binary_factor_text.py --phase teacher --root /tmp/binary-new-text --steps 1200
python scripts/binary_factor_text.py --phase confirm --root /tmp/binary-new-text --steps 600 \
  --start 40 --seeds 3 --selection results/binary_factor_20261009/text_protocol.json
```

Use a fresh output directory and the pinned corpus manifest for exact comparisons.
For archive replay, copy its data directory and teacher.pt to a new root rather
than regenerating data with an unmatched Python installation. No priority claim
for SVD, Joint-ITQ, binary factorization, SR or master-weight elimination is made.
