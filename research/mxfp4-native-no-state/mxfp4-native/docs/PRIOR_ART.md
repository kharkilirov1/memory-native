# Prior art: who already trains in 4 bits, and what is left

Surveyed 2026-10-28. The point of this page is to keep the novelty claim honest: most of
the format work is solved by others, and exactly one design choice is ours.

## The three axes that matter

Memory during training decomposes into four buffers:

```
weights + master copy + optimizer state (m, v) + gradients
```

Every published method attacks a subset:

| method | weights | master | optimizer | grad | what it removes |
|---|---|---|---|---|---|
| BF16 + AdamW | 2 B | 4 B | 8 B | 4 B | — |
| FP8-LM / DeepSeek-V3 | 1 B | 4 B | 8 B | 4 B | weight precision only |
| [ECO](https://arxiv.org/abs/2601.22101) | 1 B / 0.5 B | **0** | 8 B | 4 B | the master copy |
| [Full-Stack FP4](https://arxiv.org/abs/2607.04422) | 0.53 B | 0 | ~1.06 B | 4 B | master + optimizer precision |
| [BitNet b1.58](https://www.jmlr.org/papers/volume26/24-2050/24-2050.pdf) | 0.25 B | 0 | 8 B (fp32 tail) or 0 | 4 B | weight precision |
| **this project** | **0.53 B** | **0** | **0** | per-tile only | master **and** optimizer |

The last line is the whole claim: the 2-bit residual replaces the momentum buffer, so the
error feedback that ECO stores in `m` is stored *in the weight word itself*.

## What is already established (do not re-derive)

**MXFP4 pretraining works, with caveats.** [Cim et al., arXiv:2605.09825](https://arxiv.org/abs/2605.09825)
fine-tuned the format question on Llama-3.1-8B/C4: FP4 in Fprop and Dgrad costs a modest
increase in tokens, while **quantizing Wgrad is the primary driver of divergence**;
stochastic rounding and random Hadamard rotations do *not* rescue it, deterministic
Hadamard rotations do. Consequence for us: we quantize **weights only** and keep the
gradient path fp32. We are in the regime they report as safe.

**Recipes at scale need selective high precision.** [Agrusa et al., arXiv:2610.24343](https://arxiv.org/abs/2610.24343)
train 8B dense and 30B-A3B MoE for up to 1T tokens in NVFP4 and find that (i) every
ingredient of the recipe matters, (ii) some layers must stay high precision, (iii) an
unbiased gradient estimator beats stochastic rounding on the backward path. We should
expect to keep embeddings/head/norms in fp, and should consider their estimator instead of
plain SR inside the updater.

**Removing the master copy has a theory.** [ECO](https://arxiv.org/abs/2601.22101) proves
convergence to a constant-radius neighbourhood when the quantization error is injected into
momentum, and shows that *naive* removal gives a stationary error scaling as `1/lr` — it
diverges as the learning rate decays. Our 2-bit residual is a coarser version of their
injection, so their analysis is the template for ours, and their `1/lr` result is the
failure mode we must check on a long run.

**Ternary from scratch is a solved story** ([BitNet](https://www.jmlr.org/papers/volume26/24-2050/24-2050.pdf)),
which is why the previous project's converter existed: ternary quality comes from training
in ternary, not from converting into it.

**Tooling exists.** [torchao mx_formats](https://github.com/pytorch/ao/blob/tp/torchao/prototype/mx_formats/README.md)
ships MX tensor subclasses, packing and Triton matmuls; `torch` itself has
`float4_e2m1fn_x2`, `float8_e8m0fnu` and `scaled_mm`. We should build on the packing and
the matmul, not re-implement them.

## What is actually ours

1. **The error buffer is 2 bits inside the weight word**, not a separate momentum tensor.
   ECO needs `m` (4 B/param) to hold the error; we need 0.25 B. That is the difference
   between "25% less static memory" (their number) and "93% less".
2. **No optimizer object at all.** The update is applied inside the autograd Function's
   backward, per tile, and the gradient tile dies with the step. There is no full-model
   gradient buffer, which is the term nobody else removes (4 B/param, and the dominant one
   for MoE at scale).
3. **The deployed artifact is the training format minus two bits.** No conversion step, no
   salient channel, no recovery finetune — the things that made the previous project's
   quality numbers hard to trust.

## What is explicitly *not* claimed

- No speed claim on consumer Blackwell (see KERNELS.md: emulation only).
- No claim that 2-bit feedback matches a momentum buffer; the measured gap on a synthetic
  task is 1.41% (floor) vs 2.63% (SR without feedback), and closing it is a Phase-2 goal,
  not an assumption.
- No large-model result. Everything so far is a 16x16 synthetic teacher.
