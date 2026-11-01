# mx6 latent grid: native bf16 path vs a strong dense baseline (G4, 2026-11-01)

Hardware: Colab G4 = NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB),
torch 2.11.0+cu130, triton 3.6.0. Code: commits 96443eb and a2dc64f on
`research/mxfp4-native-no-state`. Raw logs: `mx6_g4_speed_native_2026_11_01/{g4g,g4h,g4i,g4j}/`.

Reference to beat (`dense_strong`): bf16 autocast, fp32 master weights, fused AdamW,
`torch.compile` of the whole model, SDPA attention. The lg arm (`lg6_fp4_h2`: 6-bit
FP4-aligned latent grid, NVFP4 block scales, headroom 2) uses the native path: bf16
decode kernel, cuBLAS bf16 GEMMs, one-launch Triton update in backward, custom ops so
`torch.compile(fullgraph=True)` keeps one graph; same model code and SDPA.

## Per-layer step (fwd+bwd+update, M=16384, g4i/b_fp4.log)

| layer (N,K) | nn.Linear + fused AdamW | lg (cuBLAS bf16) | lg (NVFP4 fwd, approx) |
|---|---|---|---|
| 2048x2048 | 1.00x | 0.98x | 1.00x |
| 8192x2048 | 1.00x | 0.93x | 0.85x |
| 2048x8192 | 1.00x | 0.93x | 0.85x |

lg update kernel: 0.05-0.10 ms per layer (59x faster than the eager torch update at
2048x8192, ~1 TB/s). torch._scaled_mm NVFP4 GEMM: 2.1-3.1x cuBLAS bf16; fused Triton
activation quantizer 33x faster than eager and bit-exact. `tl.dot_scaled` does not
compile on Triton 3.6 / sm_120.

## Whole-model step (bench_train_step, bf16, SDPA on, g4i + g4j)

| preset (params) | dense_strong | lg6 + compile | lg6 NVFP4 fwd | peak mem dense -> lg |
|---|---|---|---|---|
| d512 (26M) | 8.01 ms | 8.15 ms (1.02x) | 12.41 ms | 1.67 -> 1.38 GB |
| d1024 (102M) | 23.06 ms | 21.86 ms (0.95x) | 22.69 ms | 4.13 -> 2.65 GB |
| d2048 (807M) | 87.03 ms | 72.12 ms (0.83x) | 68.34 ms (0.79x) | 18.6 -> 5.6 GB |
| 1b (1.21B), cosine lr | 130.71 ms | 108.40 ms (0.83x) | 102.61 ms (0.79x) | 27.8 -> 8.4 GB |
| 1b + grad checkpoint | 156.00 ms | 133.41 ms (0.86x) | 121.81 ms (0.78x) | 19.4 -> 2.5 GB |

Where the time goes at d2048: dense optimizer step 15.5 ms vs lg update ~4.2 ms; the
rest of the model is compiled the same way for both. Without compile on the lg arm
(before the custom-op path) lg was 1.42-1.89x slower: the gap was the uncompiled
attention/LN/MLP, not the counter layers.

## Quality under the fast path (bf16 + compile + cosine, 2 seeds, held-out val)

| | tinyshakespeare d=128 | enwik8 d=512 (4 layers) |
|---|---|---|
| dense_nobias | 2.2729 | 1.7956 |
| lg6_fp4_h2 | 2.4097 (fp32 eager 4-seed: 2.4141) | 1.9850 (fp32 eager: 1.9771) |
| lg6_fp4_h2 NVFP4 fwd (approx) | 2.4470 | 1.9890 |

All compiled runs: graphs=1, graph_breaks=0 with a cosine schedule. GPU tests:
124 (g4i) and 82 (g4j) passed.

## Limits

- Speed numbers are a 100-step timed window after warmup, batch 8x512 at d2048/1b;
  quality numbers are small models (<=d512). No 1B quality run yet.
- lg updates inside backward, so it cannot do gradient accumulation; the whole batch
  must fit one micro-batch (checkpointing helps: 2.5 GB at 1.2B, batch 8).
- NVFP4-forward quantizes activations (rel. error ~9.5%); quality cost +0.04 at d=128,
  ~0 at d=512 (2 seeds).
- `mode="reduce-overhead"` (CUDA graphs) would freeze the update RNG seed; not used.