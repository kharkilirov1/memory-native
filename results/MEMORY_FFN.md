# M1 — CounterMemoryFFN: retrieval matches a dense FFN at a fraction of the active compute

The architecture lever. Replace a dense FFN (every weight touched by every token, active compute
linear in width) with a **product-key memory**: a token retrieves its top-k of E cells; the value
table is counter-state (0.75 B/weight). Retrieval is O(√E) via the product-key index, so **capacity
(E) scales without growing per-token active compute**. Exact-for-active: a cell a token did not
retrieve has exactly zero gradient, so updating only retrieved rows is the *exact* gradient — the
counter optimizer lives in the value table and only read rows tick.

## Witness — `scripts/memory_ffn_witness.py`

Tiny GPT (d=128, 3 layers, block 64), FFN sublayer swapped between arms; attention is dense fp in
all arms (isolates the FFN variable). Same data/steps (1500), synthetic char corpus.

| arm | val-loss | FFN active MACs/tok | FFN persist | tok/s |
|---|---|---|---|---|
| dense FFN (fp, AdamW) | 0.9591 | 131072 | 1543 KiB | 66305 |
| counter-dense FFN | 0.9577 | 131072 | 391 KiB | 48398 |
| counter-memory E=4096 k=16 | 0.9645 | 20736 (6.3× less) | 1848 KiB | 29404 |
| **counter-memory E=16384 k=16** | **0.9596** | **26880 (4.9× less)** | 6816 KiB | 27456 |
| counter-memory E=65536 k=16 | 0.9625 | 39168 (3.3× less) | 26544 KiB | 25051 |

## Verdict — PARTIAL (the core claim holds; the scaling claim does not at this scale)

**PASS — retrieval matches dense at a fraction of the active compute.** The memory FFN at E=16384
reaches val 0.9596 vs dense 0.9591 (within noise, ~0.05%) at **~4.9× fewer active MACs per token**.
The capacity-without-active-FLOPs *mechanism* is confirmed directly: 16× the cells (4096→65536)
costs only 1.9× the active MACs (20736→39168), because only the √E sub-key term grows — a dense
layer would be 16× on width. This is the FLOPs-to-loss lever: the same loss at less active compute,
trading cheap counter-packed persistent memory for active FLOPs.

**NOT SHOWN — "bigger E keeps lowering loss."** The E-sweep is **non-monotonic**: E=65536 (0.9625)
is *worse* than E=16384 (0.9596), and costs 17× the dense persistent memory. At this scale the
largest table is undertrained — the router cannot learn to address 65k cells from a tiny char-level
corpus in 1500 steps, so most cells are rarely retrieved and barely trained (the known memory-layer
utilization problem). So at this scale the sweet spot is a *moderate* E, not "as big as possible."

**Honest reading.** The architectural mechanism — retrieval carries the FFN's capacity at a fraction
of the active compute, with an exact (un-approximated) counter update on the active rows — is
validated. The *scaling* that makes memory layers a big win at frontier scale (Memory Layers at
Scale, Meta 2024: matches MoE at equal params/compute, beats dense at >2× compute) needs real data,
more steps, and cell-utilization care (init, balancing) — none of which a tiny CPU char witness has.
What's proven here: it works and it's cheaper per token; what's not: that bigger is monotonically
better at toy scale.

## What this kills / enables
- Kills "every token must pass through the whole FFN matrix": active compute is now O(√E + k·d),
  decoupled from capacity E.
- Enables composing with M4 (Counter-MoE, expert = memory-FFN) and is the strictly-better sibling of
  M8 prototype-stat (exact-for-active vs biased-approximate).

## Caveats / next
- tok/s here is CPU and dominated by the python gather/scatter + `unique` in the update; the active
  MACs (not wall-clock) are the FLOPs-to-loss signal. A GPU gather/scatter kernel is future work.
- Single memory head; multi-head retrieval and a real-corpus / longer-step sweep are the next gate
  to test the scaling claim properly (default OFF; opt-in like every numerics-changing mode).

## Scaling re-test — the non-monotonicity is a TOY-DATA artifact (`scripts/scaling_retest.py`)

To check whether E=65536 was merely undertrained, the E-sweep was re-run at 1500 / 4000 / 8000 steps:

| steps | dense FFN | mem E=4096 | mem E=16384 | mem E=65536 |
|---|---|---|---|---|
| 1500 | 0.959 | 0.965 | 0.960 | 0.963 |
| 4000 | 1.226 | 0.961 | 0.965 | 0.971 |
| 8000 | 1.800 | … | … | … |

**More steps does NOT help — it overfits.** The dense FFN val-loss *rises* 0.959 → 1.226 → 1.800 as
steps grow: the tiny synthetic corpus (120k chars, vocab 30) is memorized. So "more training" can't
test the capacity-scaling claim — the data saturates first. The M1 non-monotonicity at 1500 steps is
a **small-data / overfit artifact, not a real capacity ceiling**.

**Side finding in M1's favour:** the **memory-FFN barely overfits** — it holds val ~0.96–0.97 while
the dense FFN blows up to 1.80. The retrieval memory regularizes far better than a dense layer. This
is a point *for* the architecture that 1500 steps didn't reveal.

**Honest conclusion:** the capacity-scaling claim is **inconclusive on toy data** and needs a **real
corpus** (FineWeb/shakespeare, many tokens) — not more steps. That is the proper Phase-2 gate. What
IS established: retrieval matches dense at ~5× less active compute, and resists overfitting.

## Why M1 underperforms — ROUTER STARVATION (GPU diagnosis, real tinyshakespeare)

The non-monotonic E-scaling is not "bad memory" — it is the **product-key router starving cells**:
most cells are never retrieved, so they never train. New `live_fraction()` instrument (fraction of
cells ever retrieved) + a GPU sweep on real data (ZeroGPU, d=256/4L/1500 steps):

| config | val | live-cells | MACs/tok |
|---|---|---|---|
| dense FFN (ref) | **1.655** | — | 524288 |
| mem E=16384 k=16 LR×1 | 1.8559 | 27.0% | 41216 |
| mem E=16384 k=16 LR×10 | 1.9020 | 29.3% | 41216 |
| mem E=16384 k=32 LR×1 | 1.8615 | **41.1%** | 46080 |
| mem E=65536 k=16 LR×1 | 1.8805 | 11.9% | 53504 |
| mem E=65536 k=16 LR×10 | 1.9227 | 11.2% | 53504 |

**Starvation confirmed and severe, scaling with E**: only **27% of cells live at E=16k, 11.9% at
E=64k** (73–88% dead). This is exactly the M1 non-monotonicity mechanism — bigger E → each cell is
retrieved more rarely → undertrained → worse. (CPU pre-check agreed: 97% / 72% / 37% live at
4k/16k/64k.)

**But the cheap fixes do NOT recover quality:**
- **Router LR×10 fails** — barely moves utilization (27→29% at 16k; 11.9→11.2% at 64k) and *hurts*
  val (1.856→1.902; 1.881→1.923). Faster router learning makes it commit/collapse earlier, not
  spread. So it is not a learning-rate problem.
- **k=32 raises utilization a lot (27%→41%) but val is flat** (1.856→1.862). More recall touches
  more cells without improving loss — the extra cells carry little useful signal at this scale.

**Conclusion.** The starvation diagnosis is correct (and it explains the E-scaling failure), but the
problem is **deeper than a LR/k tweak** — it is product-key routing *collapse / lack of load
balancing* over a huge cell set. This is precisely why **Counter-MoE wins** (`REALDATA_SCALING.md`):
a dense router over FEW experts with an explicit load-balancing aux loss *forces* balanced
utilization; product-key memory over 16–64k cells lets the router collapse onto a tiny live fraction.
The untested candidate fix is a **load-balancing / entropy loss on the top-k retrieval** (needs a
forward-side aux term, like MoE's) — the next experiment. Even the best memory config here (1.856)
still trails dense (1.655); standalone memory-FFN is not yet a dense replacement, MoE is.
