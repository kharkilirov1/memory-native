# Results and evidence index

This directory contains CPU witnesses, Tesla T4/2×T4 runs, Colab recovery reports,
forecasts, and negative results. Evidence completeness differs by experiment: many
early T4 logs are committed; some later runs exist only as Markdown summaries or
external Drive/Kaggle artifacts. A historical table is not a complete raw witness.

## Small-model memory and throughput

The tested short character-LM runs had lower counter training peak and similar or
lower validation loss among the compared configurations. The tested widths were
d=512 and d=768, not a demonstration of long-run large-model quality parity.
See [`SHOOTOUT.md`](SHOOTOUT.md) for seeds, budgets and baseline limitations.

| Measurement | d=512 | d=768 |
|---|---:|---:|
| Counter + int4 / AdamW peak | 0.99 / 1.19 GiB | 2.06 / 2.73 GiB |
| Counter / 8-bit Adam peak | 0.99 / 1.35 GiB | 2.06 / 3.34 GiB |
| Counter / AdamW val loss | 2.6390 / 2.6184 | 2.93 / 3.08 |
| Counter throughput after untiled update | About 2× slower than AdamW | About 1.7× slower |

The table uses the shared 200-step width-sweep log, where d=512 counter throughput
was 9.9k versus AdamW 19.5k tok/s. The first 400-step shootout used a slower tiled
update (3.3k tok/s); a separate counter_packed untiled timing reached 8.9k tok/s
at 1.30 GiB. Do not combine the first run's peak
and later run's throughput into one configuration measurement. Six-bit coefficient
storage is 0.75 B/coefficient plus scales/RMS statistics, metadata and the FP tail;
whole-step VRAM savings are smaller and include activations and temporary buffers.

## Core and hardware witnesses

| Topic | Evidence | Scope |
|---|---|---|
| T4 baseline parity and packed forward | [`SUMMARY.md`](SUMMARY.md), [`gpu_validate_T4.log`](gpu_validate_T4.log) | Short small-model run; decode-in-GEMM is correct but slower than cuBLAS |
| Saved int8/int4 activations | [`gpu_validate_T4_actbits.log`](gpu_validate_T4_actbits.log), [`gpu_validate_T4_int4packed.log`](gpu_validate_T4_int4packed.log) | Savings/quality on tested settings; int4 bit packing itself is lossless |
| Memory-efficient optimizer comparison | [`SHOOTOUT.md`](SHOOTOUT.md), [`gpu_shootout_T4.log`](gpu_shootout_T4.log), [`gpu_shootout_scale_T4.log`](gpu_shootout_scale_T4.log) | Single-seed, short runs; LR tuning incomplete |
| Reversible and persistent memory | [`POOLS.md`](POOLS.md), [`FULL_METHOD_RUN.md`](FULL_METHOD_RUN.md) | Architectural change; tested depths/data only |
| Row fused update | [`KERNEL.md`](KERNEL.md) | 45.9× isolated update, 1.26× tested layer forward/backward step; not all-model speed |
| 1.21B T4 scale | [`SCALE_1B.md`](SCALE_1B.md) | Historical 2000-step fit/learning report; primary log and exact runner are missing |
| Strict update-from-IO / cuBLAS frontier | [`ACCELERATION.md`](ACCELERATION.md), [`group_kernel_opt_stage01.md`](group_kernel_opt_stage01.md) | Strict memory path is dramatically slower on tested T4 shapes |
| Group-local fused kernel + decimation | [`GPU_GATE_T4_GROUPLOCAL.md`](GPU_GATE_T4_GROUPLOCAL.md), [`GROUP_LOCAL_UPDATE.md`](GROUP_LOCAL_UPDATE.md) | Dec4 update kernel 1.7–3.0× faster than dense update; full-group kernel slower; not whole-step speed |
| MLX/group/Bonsai compatibility | [`../docs/MLX_PORT.md`](../docs/MLX_PORT.md), MLX tests | Linux CPU gates; Metal performance/parity on Apple silicon still open |

## Donor conversion and recovery

Compare PPL only on the same tokenizer, corpus revision, held-out slice and evaluation
protocol. Visible ternary inference bit rates (~1.7–3.6 bpw in solver discussions)
exclude the update accumulator and are not the six-bit packed training representation.

| Experiment | Recorded result | Evidence boundary |
|---|---|---|
| Qwen 1.5B v3f then corrected v3f2 | v3f2 EN 34.41 vs donor 11.6, RU 30.34; mean accuracy retention 70.4% across five tasks | [`recovery_15b_v3_final.md`](recovery_15b_v3_final.md); v3f was FP-tail-only; Drive checkpoint/full logs not public here |
| Salient/asymmetric PTQ sweep | Same-slice classic EN 77.89 → s2i2 35.59; s3 31.68 uses more salient storage | [`solver_v3_salient_scope_asym_gate_colab.md`](solver_v3_salient_scope_asym_gate_colab.md); ephemeral logs/account notebook not public |
| Later s2i2 6000-step recovery | Reported EN 30.06/RU 22.39/aggregate 2.6779 | Same report; checkpoint not persisted, full per-domain metrics missing, different corpus slice from PTQ table and v3f2 |
| Group-local 0.5B recovery gate | Re-centered LR improved group arm; dec4 roughly matched row | [`GROUPLOCAL_KD_FULLMODEL_GATE.md`](GROUPLOCAL_KD_FULLMODEL_GATE.md); shallow pre-gate ranking reversed before LR tuning; 300 steps, single seed |
| 1.5B row/group/dec4 gate | Final metrics row 2.7120, group 2.7262, dec4 2.7107 on shared bins | [`PROD_GATE_15B.md`](PROD_GATE_15B.md); 1500 steps, one seed/LR point; no general quality guarantee |
| 12B streaming/restore | Conversion and restored inference reports | [`gemma4_12b_full_conversion.md`](gemma4_12b_full_conversion.md), [`GEMMA12B_TERNARY_INFERENCE.md`](GEMMA12B_TERNARY_INFERENCE.md) |
| 12B cached-KD failure | Warm EN 46.4 → strict final 46,608; final aggregate 11.5862 | [`GEMMA12B_CACHED_KD.md`](GEMMA12B_CACHED_KD.md); warm conversion remains better; external Kaggle artifacts |
| Standardized external evaluation | WikiText-2 protocol and baseline | [`WIKITEXT_EVAL_PROTOCOL.md`](WIKITEXT_EVAL_PROTOCOL.md); custom-slice PPL must not be placed beside published benchmark numbers |

## Negative results and open gates

- Triton packed forward/grad_x kernels matched references but lost to decode + cuBLAS
  on the tested T4 workloads.
- Short-run likelihood improvements did not improve mean task accuracy in v3f2.
- Group scope and decimation change update dynamics; their optimal LR does not transfer
  automatically across depth or donors. The 12B cached-KD recipe failed.
- Persistent-state savings do not imply the same reduction in training peak or wall time.
- Converged large-model quality, matched QLoRA comparisons for fine-tuning, and Apple-silicon
  Metal throughput/peak memory remain open.

See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for commands and missing artifacts,
and [`VERIFICATION_RESULTS.md`](VERIFICATION_RESULTS.md) for the historical lever ledger.
New reproductions must preserve commit, environment, dataset hashes and raw outputs.
