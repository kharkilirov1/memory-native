# Experiment runners

[Research](../README.md) / Experiments

Run these from the repository root after installing the package. They are individual
research protocols, with their own hardware, data and environment requirements.
Current conversion and recovery tools remain in [scripts/](../../scripts/README.md).

| Area | Entry points |
|---|---|
| Kernel performance | [Group kernels](benchmark_group_kernels.py), [GPU acceleration](gpu_acceleration_bench.py), [tiled updates](tiled_update_frontier.py) |
| Update dynamics | [Decimation](decimation_witness.py), [group counters](group_counter_witness.py), [slow/fast state](slowfast_witness.py) |
| Model components | [Memory FFN](memory_ffn_witness.py), [MoE](moe_ffn_witness.py), [multi-token prediction](mtp_witness.py) |
| Donor gates | [Group-local pre-gate](grouplocal_kd_pregate.py), [full-model gate](grouplocal_kd_fullmodel_gate.py), [salient refit](salient_refit_witness.py) |
| Alternative arithmetic | [Main protocol](new_math_experiments.py), [capacity](new_math_capacity_ablation.py), [covariance](new_math_covariance_ablation.py) |
| Scaling | [Real data](realdata_scaling.py), [scale retest](scaling_retest.py), [validation recipe](run_scale_validation.sh) |
| Figures and diagnostics | [Article charts](make_article_charts.py), [saturation](saturation_probe.py), [layer profiling](layer_profiler.py) |

Outputs belong in `research/results/` or an explicitly chosen run directory. Record exact
arguments and preserve raw outputs. Historical source hashes refer to the original paths
and revisions; see the [layout map](../layout-map.json).
