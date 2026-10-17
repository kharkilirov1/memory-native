# Counter step speedup, 2026-10-17 (CPU, pure-PyTorch path)

Measured on the 20-core Windows CPU box (torch 2.12.1+cpu, no CUDA, 10 threads).
A/B between `origin/main` at the start of the change (`b4e163a`) and the branch that
made it, same machine, same script, same shapes.

## Result

| phase | before | after | change |
|---|---:|---:|---:|
| forward (decode + GEMM) | 31.3 ms | 22.9 ms | **−27%** |
| grad_x (decode + GEMM) | 31.3 ms | 25.6 ms | **−18%** |
| update (automaton chain) | 375.0 ms | 291.6 ms | **−22%** |
| **full fwd+bwd step** | **425.4 ms** | **359.6 ms** | **−15.5%** |
| dense linear + AdamW (reference) | 8.9 ms | 8.9 ms | — |

Layer 1536x1536, `group=128`, `C=11`, batch 64, `kernel_mode="gemm"`, strict
readout (`residual_alpha=0`). Medians from
`torch.utils.benchmark.blocked_autorange(min_run_time=3s)`; the harness is
[`scripts/bench_counter_step.py`](../../scripts/bench_counter_step.py).

The step is still ~40x the dense baseline on this CPU: the dominating cost is the
pure-torch automaton chain, not the GEMMs (a 0.6 GFLOP GEMM pair is ~8 ms of the
360 ms). This note reports a bit-identical improvement to that chain and the decode
path; it does not claim the CPU path is competitive with a dense step.

## What changed (all bit-identical, verified below)

1. **`_broadcast_groups`** ([group_scale_kernels.py](../../src/memory_native/group_scale_kernels.py)):
   `scale[:, arange(in)//group]` → `scale.repeat_interleave(group, dim=1)` when the last
   group is full. Same values, no index tensor, no gather: **3.45 → 0.40 ms** per call,
   and the update chain needs three of them. `visible_weight` uses it too.
2. **Strict-readout fast path** (`residual_alpha == 0`): skip `c.float()` and the
   `alpha*c/C` residual in both `visible_weight` and the scale-gradient term of the two
   update references (`+ 0.0` is a no-op).
3. **Mask-free hash-SR** (`hash_u32_unmasked` / `uniform01_seeded` in
   [fused_update.py](../../src/memory_native/fused_update.py)): the stochastic-rounding
   stream is `uniform01(seed ^ hash_u32(elem))`, i.e. two 32-bit hashes over the whole
   `[out, in]` element index. `elem < 2**32` makes the opening
   `to(int64) & 0xFFFFFFFF` and the closing `& 0xFFFFFFFF` no-ops, so two of ~11
   full-tensor passes per hash are dropped: **106.4 → 55.8 ms** per step (1.9x).
   The fallback expression is kept for `out*in >= 2**32`.
4. Not changed: the GEMMs, packing, telemetry, and the update math itself.

## Bit-exactness (the reason this is safe)

Checked directly against the pre-change expressions on a real layer:

```
scale broadcast identical: True          # gather vs repeat_interleave
alpha=0 value identical  : True          # t.float() + 0*c/C  vs  t.float()
SR identical seed=0/7/123456789: True    # masked vs mask-free double hash
```

Verified with `python -c` on the same shapes; the full test suite pins the rest.

## Reproduce the A/B

```bash
git worktree add --detach /tmp/mn-baseline <base-commit>
cp scripts/bench_counter_step.py /tmp/mn-baseline/scripts/
python /tmp/mn-baseline/scripts/bench_counter_step.py    # before
python scripts/bench_counter_step.py                     # after
```

The script puts its own `../src` first on `sys.path`, so each worktree copy measures
its own checkout.

## Where the remaining time goes, and the real levers

Profile of one update at these shapes (threads=10): hash-SR ~30%, code unpack/pack
~10%, and the rest is ~16 multiplies / 8 divides over `[out, in]` fp32 plus the
packed-code round trip. That is why the honest levers are structural, not more
micro-optimizations:

- **Triton fused update** (CUDA only) — one launch without the dense `grad_w`; this is
  what the GPU path uses and what the CPU reference cannot reach.
- **Decimation** (production recipe) — update every k-th step or on a group subset.
- **Fewer decodes per step** — forward and grad_x decode the same weight; caching it
  across backward would halve the decode but adds 4 B/param resident per layer, which
  breaks the memory story, so it stays off.
