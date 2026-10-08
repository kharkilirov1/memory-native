# Binary counter factors with geometric initialization

Research date: 2026-10-08/09. Base: `851c91ed4bb088424cbf05af8ca83a2bf1c76844`.
Status: opt-in CPU research; production defaults unchanged.

## Delivered scope

An independent implementation of the factorized architecture and initialization
ideas in [LittleBit](https://arxiv.org/abs/2506.13771) and
[LittleBit-2](https://arxiv.org/abs/2603.00042), with a finite-state learning backend.
No Samsung source was copied. This is NOT the authors' complete SmoothSign,
residual-factor, layerwise distillation or model-scale recipe.

`research/binary_factor.py` supplies SVD/random/Joint-ITQ initialization,
`PackedBinaryLinear`, `NativeBinaryFactorLinear`, matched continuous QAT controls,
validated checkpoint continuation, and a one-bit inference-only factor artifact.
The effective operator is

    W_hat = diag(h) B_left diag(ell) B_right diag(g),  B_left,B_right in {-1,+1}.

Both native and QAT controls start from the same decoded latent lattice and shared
scales. The effective matrix is not restricted to binary/ternary entries, but its
rank is restricted. Arbitrary internal rotations are valid for the two-factor
product; they cannot simply be inserted into BLAST's diagonal middle factor.

`guarded=True` selects among SVD, seeded random rotation and final Joint-ITQ using
actual weight reconstruction error, or explicitly supplied calibration inputs.
This only guarantees no worse *initialization selection score* than SVD. It is
not a guarantee for downstream training or unseen data.

## State and execution contract

Each factor coefficient has code z in [0,63], latent q=(z-31.5)/16, and visible
sign +1 for z>=32, otherwise -1. Four codes occupy three bytes. Shared row-RMS,
step indices, h/ell/g and their optimizer state are additional memory.

    v_new = beta*v + (1-beta)*mean_row(gradient**2)
    z_proposal = z - 16*lr*gradient/max(sqrt(v_new),eps)
    z_new = clip(stochastic_round(z_proposal),0,63)

The latent conditional mean follows the row-RMS step under ideal SR without
clipping. This is not a convergence or visible-sign unbiasedness theorem.
The format is NOT compatible with the original 63-state ternary (t,c) code.

There are no persistent FP master *factors*. Scales/bias and the rest of a model
can still be ordinary Parameters. Factorized forward/backward does not construct
the virtual full W or full dW; decoded sign factors, factor gradients and proposals
DO exist. Initialization performs dense SVD and dense candidate scoring. No
streaming large-model conversion or fused GPU execution is asserted.

Input derivatives use old weights before mutation. Training is first-order CPU
eager only, one forward/backward per update, with checkpointed independent SR
streams. CUDA/AMP, DDP, shared-layer accumulation and higher-order derivatives
are unsupported. A later backward error does not promise whole-model rollback.

Training and inference use the same hard readout: there is no alpha-residual to
remove at export. The separate export stores one bit per factor sign plus padding
and FP32 scales; it discards hidden/RMS state and is not a resumable checkpoint or
an optimized inference kernel. It is not the Samsung/BitNet file format.

## Matrix experiment

48x64 target, rank8, 600 steps, batch64, 1536/256/512 train/val/test Gaussian rows.
Three tuning seeds0-2; LR grid {.001,.005,.02}. LR is selected over BOTH
initializers, then held fixed across initializers within each learning rule.
Native=.005, QAT Adam/RMS=.02; shared-scale Adam=.003. Confirmation seeds20-31.
The pre-confirmation selection is recorded in `matrix_protocol.json`.

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

Geometry improves native spectral-target MSE by14.28%, 10/12 pairs, paired
bootstrap gain interval [.02949,.10743]. It does NOT establish counter superiority:
frozen signs beat native+geometry, and QAT Adam is especially stronger on the
representable binary teachers. QAT Adam can prefer SVD *after* learning despite
worse warm initialization. All results are final held-out metrics, not best-test
checkpoint selection. Exact records are retained in the conversation research
archive. No large-model inference follows from these small regressions.

## Pretrained byte-GPT recovery

Two blocks, d64, four heads, context64, batch8, byte-vocabulary256, tied embedding
and head. A dense teacher is trained1200 steps (614400 sampled byte positions).
Only four FFN matrices are replaced by rank16 binary factors (20480 coefficients).
Attention, embedding/head, norms and scales remain FP and train during recovery.

Corpus:146 top-level Python3.13.5 stdlib files, whole-file SHA(name) split;
train3961479, val354863, test226768 bytes. Windows do not cross file boundaries.
Evaluation uses8192 val and32768 test positions in fixed sampled windows, which
may overlap; not full-corpus PPL. Data manifests and exact split tensors are in
the research archive. A different local stdlib changes the benchmark.

Tuning seed91,300 steps; selected factor LR=.001 for both native and QAT-Adam.
Scale LR=.003 and FP-shell LR=.0003. Before new seeds40-42, the protocol was saved.
Each arm runs600 recovery steps (307200 positions), KL(teacher||student)+.3CE at
T=1. No test-based selection. QAT uses clipped STE, NOT LittleBit SmoothSign100.

| Rule | Init | Mean final CE | exp(mean CE) | Training seconds | Persistent model+optimizer tensor bytes |
|---|---|---:|---:|---:|---:|
| Dense continued recovery | none | 2.084075 | 8.037157 | 7.321 | 1433168 |
| QAT Adam | SVD | 2.136684 | 8.471303 | 8.070 | 911504 |
| QAT Adam | guarded | 2.124679 | 8.370211 | 7.844 | 911504 |
| Native | SVD | 2.133990 | 8.448506 | 12.477 | 681200 |
| Native | guarded | 2.128088 | 8.398796 | 12.463 | 681200 |
| Frozen signs | SVD | 2.128012 | 8.398152 | 8.247 | 681200 |
| Frozen signs | guarded | 2.124931 | 8.372318 | 8.122 | 681200 |

Each row averages three paired seeds. Native+geometry is +.003409 nat/byte,
+.342% perplexity versus matched QAT, with25.27% less persistent state but58.9%
more CPU time. Three seeds do not establish equivalence. Continued dense is
stronger. Frozen signs slightly beat counter learning, so recovery cannot be
credited to counter sign changes. Native+geometry averages173 flips across600
steps; all8 banks receive600 updates. Frozen controls retain banks for matched
layout; a stripped frozen artifact could use less memory.

Timing includes student training and teacher forward, excludes initialization,
batch construction and evaluation. This is not time-to-quality or a GPU result.
Persistent bytes exclude teacher, retained gradients, activations and temporary
storage; no peak-memory claim. On the isolated layer native state is2380B versus
12436B for QAT Adam, but this5.23x ratio is NOT the whole-GPT ratio.

## Verification and provenance

61 feature/runner cases passed locally. In the complete pinned upstream archive,
128 targeted tests passed, including67 existing structured/covariance/packing/
latent-dynamics/carry tests. This is not the entire upstream suite.
Core blob863638eb5c235c6c13749821f3da592cd3d77569 (SHA256
`a9b6a974739e311785b81dd54cc4190c9eced57767837175325ba0fe5daa8491`) was unchanged
through all experiments. The published text runner's copied FFN expression was
corrected to the tested source; final blobb4c2d8be74a9e5c445615c3db8de61168f92b241
matches local. Both runner scripts now have execution smoke tests.

The archive includes teacher and native seed40 checkpoints. Loading the latter
reproduces CE2.1285827104002237 exactly. A separate unit test verifies continuation
of counter+scaleAdam with bit-identical state. This is a small byte-model artifact,
not a useful chatbot. Some process timeouts were resumed deterministically;
restarts are not extra independent evidence. Runner bookkeeping/precast data
changes did not alter the core learning rule. Extra pyproject metadata repair
removes unsupported authors.role while retaining the sponsor note.

## Use and reproduce

```python
from memory_native.research.binary_factor import (
    NativeBinaryFactorLinear, BinaryFactorInference, export_binary_factors,
)
import torch

source = torch.nn.Linear(64,48,bias=False)  # replace with a pretrained CPU layer
layer = NativeBinaryFactorLinear.from_linear(
    source, rank=8, method="joint_itq", guarded=True, lr=.001, seed=0,
)
optimizer = torch.optim.AdamW(layer.parameters(),lr=.003,weight_decay=0.)
x = torch.randn(16,64)
optimizer.zero_grad(set_to_none=True)
loss = (layer(x)-source(x).detach()).square().mean()
loss.backward()       # factor counter update
optimizer.step()      # shared scales update
inference = BinaryFactorInference(export_binary_factors(layer))
```

```sh
PYTHONPATH=src python -m pytest -q tests/test_binary_factor.py tests/test_binary_factor_runners.py
python scripts/binary_factor_witness.py --phase confirm --start20 --seeds12 --steps600 \
  --selection results/binary_factor_20261009/matrix_protocol.json --output /tmp/binary-new-matrices
python scripts/binary_factor_text.py --phase teacher --root /tmp/binary-new-text --steps1200
python scripts/binary_factor_text.py --phase confirm --root /tmp/binary-new-text --steps600 \
  --start40 --seeds3 --selection results/binary_factor_20261009/text_protocol.json
```

CLI option names require a separating space before numbers; for example use
`--start 20 --seeds 12 --steps 600`, `--steps 1200`, and `--start 40 --seeds 3`.
Use a fresh output directory and the pinned data manifest for exact comparisons.
Independent implementation and analysis only: no priority claim for SVD, Joint-ITQ,
low-rank binary factors, stochastic rounding or master-weight elimination.
