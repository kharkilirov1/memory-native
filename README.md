# memory-native (PyTorch)

Finite-state **counter synapses** + **reversible activations** for memory-efficient training,
in **pure PyTorch** — no custom engine, runs on stock CPU/CUDA. The standalone, engine-independent
package: everything that previously required the MotifCL C++/OpenCL build runs here with
`pip install -e .`.

> **GPU-validated** (Tesla T4 / T4×2): [`results/KERNEL.md`](results/KERNEL.md) (fused update
> ×45.9 / step ×1.26) · [`results/SCALE_1B.md`](results/SCALE_1B.md) (1.21B params on one T4 at
> 2.25 GiB; dense+Adam OOMs at 18 GiB) · [`results/POOLS.md`](results/POOLS.md) (all four memory
> pools) · [`results/SHOOTOUT.md`](results/SHOOTOUT.md) (vs AdamW / 8-bit Adam / GaLore / LoMo).

> **Validated value proposition (real runs on a Tesla T4 — [`results/SHOOTOUT.md`](results/SHOOTOUT.md)):**
> the lowest training-memory of every contestant (AdamW, 8-bit Adam, GaLore, LoMo) at
> competitive-to-better quality, for ~2× slower steps — and the advantage **grows with scale**.
> At d=768 counter+int4 beats AdamW and 8-bit Adam on memory *and* quality at once (peak 1.32×
> below AdamW / 1.62× below 8-bit Adam; val −4.7% vs AdamW; speed gap narrowing). The method
> wins because it cuts both the optimizer pool (zero state) and activations (int4), while the
> memory-efficient *optimizers* only shrink optimizer state — a small slice of the
> activation-bound peak.

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

- **Realized now (pure PyTorch, verified on CPU):** the *learning dynamics*, the
  *optimizer-state* saving (no FP master weight, no Adam moments), and — with
  `PackedRMSCounterLinear` (`kind="counter_packed"`) — **genuinely packed 0.75 byte/weight
  persistent state** (4 codes / 3 bytes, bit-identical to the engine's packing; round-trip
  and identical-dynamics tested). `memory_report` / `memory-native-memgate` quantify it (real
  `torch.cuda.max_memory_allocated` on CUDA, byte accounting on CPU).
- **Verified on GPU (Tesla T4):** the Triton forward kernel (`triton_counter.py`,
  `TritonCounterLinear`) decodes the packed state *inside* the GEMM (no dense weight) and
  matches the dense reference within f32 tol (err ≤ 3e-6). And at **d=512** counter+RMS keeps
  parity with — slightly beats — dense AdamW (val −1.7%). See [`results/SUMMARY.md`](results/SUMMARY.md).
- **Activation pool — collapsed (T4):** `ReversibleSequence` is a single whole-chain RevNet
  Function (stores only the final output, reconstructs the rest), so activation memory is **O(1)
  in depth** — 13.5× below plain at depth 256 (5.29 GiB → 392 MiB), gradients identical to the
  per-block version. With `act_save_bits` (unbiased int8/int4 saved activations) the saved-X cost
  drops too. See [`results/POOLS.md`](results/POOLS.md).
- **Fused update kernel — done (T4):** `memory_native.fused_update` collapses the per-element
  RMS+stochastic-rounding update into one launch (deterministic hash-SR, bit-quantified against a
  CPU reference), **×45.9 on the update / ×1.26 on the step**, wired into
  `PackedRMSCounterLinear`. See [`results/KERNEL.md`](results/KERNEL.md).
- **Scale — demonstrated (T4):** the full method (counter + reversible) trains a **1.21B-param**
  model on a single 14.6 GiB T4 at **2.25 GiB peak**, where dense+Adam needs 18 GiB of state
  and OOMs before step 0. See [`results/SCALE_1B.md`](results/SCALE_1B.md).
- **The one open milestone — strict update-from-IO:** the PyTorch update still consumes a
  materialized `grad_w` tile. The strict analogue of the engine's OpenCL `counter_*_fused` — a
  kernel taking `(state, scale, v, x or Q(x), grad_out)` that forms `grad_w` in registers so **no
  dense gradient is ever materialized** — is what makes the *training peak* (not just persistent
  state) sub-byte on CUDA. This is the remaining piece.

### Roadmap
1. **Strict update-from-IO kernel** — `counter_update_from_io(state, scale, v, x_or_Qx, grad_out)`
   with no materialized `grad_w`. The forward/grad_x decode-in-GEMM kernels are T4-verified but
   net-negative vs cuBLAS; the fused *update* (from `grad_w`) is done and pays off — the open ROI
   is fusing the `grad_w` formation into the update so the peak is sub-byte too.
2. **Multi-session scale** — 2×T4 data-parallel + checkpoint/resume
   ([`scripts/fineweb_1b_2xt4.py`](scripts/fineweb_1b_2xt4.py)); accumulate tokens across Kaggle
   sessions on real web text (FineWeb-Edu, BPE).
3. **Reversible at depth** — anchors (store every k blocks) to trade a little memory for less
   recompute, and a depth-sweep for float-reconstruction error beyond the tested range.

## Constraints (same eager-only contract as the engine)

- The in-backward update is **eager-only**: incompatible with gradient accumulation, weight
  sharing of a counter module, activation checkpointing, and `torch.compile` graph capture
  without an explicit scheduler. One forward → one backward per step; measure under
  `torch.no_grad()`. **Data-parallel is supported** a different way than DDP: the counter
  gradient is all-reduced *inside* backward (the optimizer is an in-place state update, so there
  is no Parameter `.grad`), keeping every replica's packed state bit-identical — validated on
  2×T4 (0 bytes differ across ranks). See `scripts/fineweb_1b_2xt4.py`.
- `ReversibleCouplingBlock` needs deterministic `F`/`G` (no dropout/RNG). Float reconstruction
  is exact enough at tested depth (≤12 blocks); deeper stacks need a depth-sweep and possibly
  anchors.

## Layout

```
src/memory_native/
  counter.py        CompactCounterLinear, RMSCounterLinear, encode/decode, stochastic rounding
  packed.py         PackedRMSCounterLinear — real 0.75 byte/weight storage (4 codes / 3 bytes)
  triton_counter.py TritonCounterLinear + in-GEMM packed decode (CUDA; T4-verified, net-negative)
  fused_update.py   one-launch RMS+SR counter update kernel (CUDA; T4-verified, ×45.9)
  reversible.py     ReversibleCouplingBlock, ReversibleSequential (recompute backward)
  actquant.py       unbiased low-bit saved activations (the counter layer's act_save_bits)
  budget.py         training_budget — symbolic 4-pool memory model (deep-v2, reproduced exact)
  baselines.py      TernaryQATLinear, make_linear factory (kinds incl. counter_packed)
  optimizers.py     build_optimizer: adamw / bnb8 / galore / lomo
  models.py         swappable-linear GPT harness + configs (micro/tiny/s512/small)
  memory.py         memory_report, peak_training_memory, compare_training_peak
  data.py           char corpus loader with offline synthetic fallback
  cli.py            memory-native-charlm / memory-native-memgate entry points
scripts/            run_scale_validation.sh (one-command GPU sweep -> results/)
tests/              pytest: encode/decode, learning, reversible grad-check, packed round-trip,
                    optimizers, memory gate, triton (CUDA-skipped)
```

## License

MIT.
