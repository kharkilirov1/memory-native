# The four memory pools — what's demonstrated (T4)

The method's thesis (original research doc §1): real savings come only from attacking **all
four** training-memory pools together — params, optimizer state, gradients, activations. Here
is where each pool stands, measured on a real Tesla T4. (My earlier framing conflated the
weight-side 16× with the activation-bound training peak; this is the corrected, full picture.)

## Pools 1–3: params + optimizer + gradients — the 16× (counter), demonstrated

The counter synapse packs weight + optimizer + gradient into ~0.75–1 byte/coeff vs 16
byte/coeff for FP32+Adam. In the regime where these pools dominate the peak (large model, small
batch×seq), the saving is real and decisive — **counter trains models a T4 cannot fit with
AdamW** (`gpu_validate_T4_oom.log`, batch=1, seq=256):

| model | ~linear params | counter+int4 | dense+AdamW |
|---|---|---|---|
| d=1024, 12L | 151M | 566 MiB | 3.00 GiB (5.3× less) |
| d=1536, 16L | 453M | 1.21 GiB | 8.71 GiB (7.2× less) |
| d=2048, 24L | **1.2B** | **2.41 GiB** | **OOM (>14.6 GiB)** |
| d=2560, 32L | **2.5B** | **4.21 GiB** | **OOM** |

The ratio climbs toward the weight-pool advantage (16×) as the model grows; counter fits a
2.5B-param model on a 14.6 GiB T4 where AdamW OOMs by d=2048.

## Pool 4: activations — reversible, demonstrated (lever works; O(1) is the next step)

Reversible coupling recomputes activations in backward instead of storing them. Peak vs depth
(dim=512, 2048 tokens, counter MLP F/G) — `gpu_reversible_depth_T4.log`:

| depth | plain (stores acts) | reversible (recompute) | gap |
|---|---|---|---|
| 8 | 314 MiB | 158 MiB | 2.0× |
| 32 | 815 MiB | 279 MiB | 2.9× |
| 128 | 2.72 GiB | 761 MiB | **3.6×** |

The activation lever works and the gap **widens with depth** — reversible increasingly wins.
Honest limitation: the current `ReversibleSequential` makes each block its own autograd
Function that stores *its own output*, so activation memory is O(depth) with a *small* constant
(one [N,dim] per block) rather than the O(1) ideal (store only the final output, reconstruct the
whole chain). Plain stores O(depth) with a *large* constant (all per-block internals), hence the
3.6× gap. **True O(1) needs a single whole-chain reversible Function** (the classic RevNet
backward) — the next implementation step.

## Combined

Both levers are demonstrated on real hardware. The full method (counter + reversible) is what
the verified budget calculator projects at ~10× total (21 → 2 GiB at L=24/d=2048); these runs
substantiate each half. The honest status: pools 1–3 are realized and decisive at scale; pool 4
works and improves with depth but is not yet O(1) in this PyTorch implementation.
