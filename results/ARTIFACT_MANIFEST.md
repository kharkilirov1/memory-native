# Evidence artifact manifest

This is an index of public evidence and known gaps, not a checksum attestation or a
claim that external artifacts were downloaded and verified. Paths below are relative
to the repository root. Historical values remain labeled as reported; missing logs
are never regenerated from narrative tables.

| Evidence | Public artifact or executable source | Completeness and limits |
|---|---|---|
| Review-fix CPU validation | `results/REVIEW_FIXES_VALIDATION.md`, raw pytest and public smoke logs | 434 passed / 21 skipped; synthetic artifact roundtrip only, no new large-model quality claim |
| Packed counter encoding/update | `src/memory_native/`, `tests/test_counter.py`, `tests/test_packed.py`, group tests | CPU regression gates are runnable; CUDA tests require their runtime/hardware. Packed `C` must be integer 1–11; a pass is not converged quality evidence |
| Small T4 parity/forward | `results/SUMMARY.md`, `results/gpu_validate_T4.log` | Raw log committed for covered settings |
| T4 baseline shootout | `results/SHOOTOUT.md`, `results/gpu_shootout_T4.log`, `results/gpu_shootout_scale_T4.log`, `results/gpu_throughput_T4_v11.log` | Separate runs and configurations; do not combine a peak from one with throughput from another |
| Reversible T4 memory | `results/POOLS.md`, `results/gpu_reversible_o1_T4.log`, `results/gpu_full_method_d1024_T4.log` | Logs committed for recorded receiver architecture/depth/data, not arbitrary donor swaps |
| Row fused kernel | `results/KERNEL.md`, `tests/test_fused_update.py` | Report plus executable gate; 45.9× update/1.26× layer step applies to the tested T4 configuration |
| Historical 1.21B single-T4 enwik8 | `results/SCALE_1B.md` | **Missing:** named primary `gpu_scale_1B_T4.log`; exact original runner/environment not pinned. The 2000-step curve and 2.25 GiB peak remain unverified historical report values |
| Later 1.21B FineWeb 2×T4 | `scripts/fineweb_1b_2xt4.py` | Different corpus/hardware from the scale report; script is public, a new execution is a separate witness |
| Qwen v3f/v3f2 | `results/recovery_15b_v3_final.md`, public recovery scripts | **External:** author's Drive checkpoint/corpus/full logs. v3f was FP-tail-only; corrected v3f2 EN 34.41/RU 30.34 and 70.4% mean accuracy retention are reported tables |
| Salient/asymmetric PTQ sweep | `results/solver_v3_salient_scope_asym_gate_colab.md` | **External/missing:** `/content/{classic,layer,layer_asym05}.log` was ephemeral and account Drive notebook is not public in this checkout |
| Later s2i2 6000-step recovery | Same report, public runner source | **Lost/not public:** checkpoint was not persisted; full raw metrics/domain digits are unavailable. Fresh 150M corpus differs from the 12M PTQ sweep and original v3f2 validation slice |
| Group-local/dec4 kernels | `results/GPU_GATE_T4_GROUPLOCAL.md`, `scripts/benchmark_group_kernels.py`, CUDA tests | Report + executable source. Raw Kaggle dataset/kernel outputs are external; 1.7–3.0× is an update-kernel result, not whole-training speed |
| 1.5B row/group/dec4 recovery | `results/PROD_GATE_15B.md`, `scripts/run_ptq_recovery.py` | Historical single-seed campaign on shared bins; Kaggle source/corpus/checkpoint outputs are external |
| 12B conversion/inference | `results/gemma4_12b_full_conversion.md`, `results/GEMMA12B_TERNARY_INFERENCE.md`, streaming source | Reports plus public implementation; model/data artifacts are external. Conversion completion alone is not a quality gate |
| 12B cached-KD failure | `results/GEMMA12B_CACHED_KD.md`, public cache/KD scripts | **External:** Kaggle warm dataset `mn-gemma12b-counter-state` v2 and kernel `mn-g12-kd-recovery` v5/v6 raw outputs. Recovery is worse than warm; preserve negative result and warm baseline |
| MLX/group/Bonsai | `src/memory_native_mlx/`, `tests/test_mlx_port.py`, `tests/test_mlx_group_scale.py`, `docs/MLX_PORT.md` | Linux CPU gates/source public. **Missing hardware witness:** Apple-silicon kernel parity/performance and peak unified memory. No full-model group solver-to-MLX bridge is claimed |
| Archived 27B external workflow | `notebooks/MN_Qwen38_27B_*.ipynb`, `production/` recipes | **Missing dependency:** externally supplied runner ZIP. Public recipes do not include or validate that private runner |
| New public conversion/cache/KD workflow | `notebooks/cached_kd_public.ipynb`, committed scripts | Uses public source without the private ZIP; donor/corpus/runtime must be supplied. Does not recreate missing historical outputs or establish new 27B quality metrics |

Six bits cover the coefficient and accumulator code, before scales/statistics,
metadata, FP tail and its optimizer, caches, activations and temporary buffers.
FP32 weights plus Adam moments cost 12 B/coefficient; 16 B includes gradients. Neither
number is a full-model peak measurement. Compare PPL only with identical tokenizer,
held-out slice, context and scoring protocol.

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for runnable gates and reporting.
New streaming manifest schema 2 records donor and calibration provenance, runtime/
solver options and per-block hashes. Legacy conversion outputs do not gain verified
provenance retroactively; create a new conversion before verified resume/recovery.
When recovering an original artifact, retain its raw bytes, source/version, commit,
software/hardware configuration, dataset identity and SHA-256 in a new manifest entry.
If an original artifact cannot be recovered, label a rerun as a new experiment.
