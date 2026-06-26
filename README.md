# memory-native (PyTorch)

Finite-state **counter synapses** + **reversible activations** for memory-efficient training,
in **pure PyTorch** — no custom engine, runs on stock CPU/CUDA. This is the engine-independent
port of the [memory-native-training](../) method: everything that previously required the
MotifCL C++/OpenCL build runs here with `pip install`.

The method attacks all four memory pools of training at once:

| Pool | Lever | What it is |
|---|---|---|
| Parameters + optimizer + gradients | `CompactCounterLinear` / `RMSCounterLinear` | a ternary weight whose optimizer state lives inside a per-synapse finite-state automaton; the update is fused into backward — no FP master weight, no Adam moments, no full gradient buffer |
| Activations | `ReversibleCouplingBlock` | activations are recomputed in backward from the output instead of stored — depth-independent activation memory |

## Install

```bash
pip install -e .            # from this directory (memory-native-training/pytorch)
# or, once published: pip install memory-native
```

Only dependency is `torch>=2.1`.

## Quickstart

```python
import torch
from memory_native import RMSCounterLinear

layer = RMSCounterLinear(256, 256, C=11, lr=3e-3, lr_scale=2e-4)
x = torch.randn(64, 256)                 # note: no requires_grad needed
for _ in range(200):
    loss = (layer(x) - target).pow(2).mean()
    loss.backward()                       # the layer self-updates here; no optimizer.step()
```

The counter layer exposes **no `nn.Parameter`** — its weight is a `uint8` state buffer and it
updates itself during `backward()`. Mix it with normal modules; an `AdamW` over the rest of
the model (embeddings, norms, head) trains those as usual.

## Command-line gates

```bash
# char-LM parity: counter+RMS vs ternary-QAT vs dense AdamW, same arch/data/seed.
# Falls back to a synthetic corpus offline; use --data-path for real tinyshakespeare.
memory-native-charlm --config tiny --steps 600 --device cuda

# compare the dense baseline under a memory-efficient optimizer instead of FP32 AdamW:
memory-native-charlm --config tiny --kinds dense --optimizer galore --device cuda

# training-peak memory gate: counter_rms vs dense across optimizers (real peak on CUDA)
memory-native-memgate --config s512 --optimizers adamw,galore,lomo,bnb8 --device cuda
```

### Baselines (the honest comparison set)
`--optimizer` / `--optimizers` selects what the dense baseline is trained with, so the memory
claim is measured against real memory-efficient training, not only FP32 AdamW:

| name | what | optimizer-state memory | deps |
|---|---|---|---|
| `adamw` | FP32 AdamW | 2×params (fp32 m,v) | torch |
| `bnb8` | 8-bit AdamW (bitsandbytes) | ~0.5×params | bitsandbytes + CUDA |
| `galore` | low-rank projected AdamW | ~2×rank×dim | torch (built in) |
| `lomo` | fused-backward SGD | **zero** moments | torch (built in) |

GaLore and LoMo are implemented in plain PyTorch here (run on CPU, no extra deps); `bnb8`
is optional and used where bitsandbytes+CUDA are present (skips cleanly otherwise).

### Scale validation
The biggest open question is whether parity holds beyond micro/tiny. One command runs the
full sweep (parity across kinds + dense-vs-optimizers + memory gate) at d=512 and saves logs:

```bash
scripts/run_scale_validation.sh s512 2000 cuda      # config steps device
DATA_PATH=/path/to/tinyshakespeare.txt scripts/run_scale_validation.sh small 4000 cuda
```

Logs land in `results/` (not committed pre-filled — capture them on your GPU).

A 60-step micro run already reproduces the method's story: `counter_rms` lands within ~0.1%
of dense, beats vanilla counter, and isolates to ~+2.5% over ternary-QAT (the counter-
optimizer cost) — matching the larger-scale numbers in [`../docs`](../docs).

## What is measured vs what is roadmap

**Honest framing — read before quoting memory numbers.**

- **Realized now (pure PyTorch):** the *learning dynamics* and the *optimizer-state* saving.
  The counter layer carries no FP master weight and no per-weight Adam moments, so its
  persistent state and training peak beat dense+AdamW on the optimizer pool. `memory_report`
  / `memory-native-memgate` quantify this (real `torch.cuda.max_memory_allocated` on CUDA).
- **NOT realized in pure PyTorch:** the **sub-byte (0.75 byte/weight)** win. This layer
  decodes states to dense fp32 tensors around the GEMM and stores state as `uint8` (1 byte,
  not 0.75 packed). Reaching the true packed peak needs a custom kernel.

### Roadmap
1. **Triton packed kernel** — a 6-bit-state → ternary GEMM with a fused counter-update
   backward epilogue that forms `grad_w` in registers and updates the state in place (the
   CUDA analogue of the engine's OpenCL `*_fused_*` kernels). This is what makes the 0.75
   byte/weight *training peak* real on CUDA.
2. **Baselines that matter** — compare against 8-bit Adam (bitsandbytes), GaLore, LoMo, not
   only FP32+Adam, so "16× less memory" is measured against real memory-efficient training.
3. **Scale validation** — run the parity gate at d=512–768 on a real GPU before any
   larger-model claim. The current evidence is micro/tiny scale.

## Constraints (same eager-only contract as the engine)

- The in-backward update is **eager-only**: incompatible with gradient accumulation, weight
  sharing of a counter module, DDP all-reduce, activation checkpointing, and `torch.compile`
  graph capture without an explicit scheduler. One forward → one backward per step; do
  measurements under `torch.no_grad()`.
- `ReversibleCouplingBlock` needs deterministic `F`/`G` (no dropout/RNG). Float reconstruction
  is exact enough at tested depth (≤12 blocks); deeper stacks need a depth-sweep and possibly
  anchors.

## Layout

```
src/memory_native/
  counter.py      CompactCounterLinear, RMSCounterLinear, encode/decode, stochastic rounding
  reversible.py   ReversibleCouplingBlock, ReversibleSequential (recompute backward)
  baselines.py    TernaryQATLinear, make_linear factory
  models.py       swappable-linear GPT harness + configs
  memory.py       memory_report, peak_training_memory, compare_training_peak
  data.py         char corpus loader with offline synthetic fallback
  cli.py          memory-native-charlm / memory-native-memgate entry points
tests/            pytest: encode/decode, learning, reversible grad-check, memory gate
```

## License

MIT.
