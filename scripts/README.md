# Workflow tools

[Memory Native](../README.md) / Scripts

Run commands from the repository root. Install donor dependencies with
`python -m pip install -e ".[donor,campaign]"` when using model conversion or recovery.
Some tools use environment variables instead of command-line flags; check the script's
configuration section before starting a run.

## Prepare and convert

| Tool | Purpose |
|---|---|
| [build_mix_corpus.py](build_mix_corpus.py) | Build tokenized data with identity metadata |
| [check_donor_config.py](check_donor_config.py) | Verify the requested donor configuration |
| [convert_streaming.py](convert_streaming.py) | Convert a donor block by block |
| [export_motifcl.py](export_motifcl.py) | Export packed state for the supporting native runtime |

## Evaluate and recover

| Tool | Purpose |
|---|---|
| [eval_wikitext_ppl.py](eval_wikitext_ppl.py) | Evaluate using the WikiText protocol |
| [infer_counter_cpu.py](infer_counter_cpu.py) | Run CPU inference on counter state |
| [restore_witness.py](restore_witness.py) | Restore streamed state and check execution |
| [kd_teacher_cache.py](kd_teacher_cache.py) | Build a teacher cache with a provenance contract |
| [kd_cached_recovery.py](kd_cached_recovery.py) | Recover and select an artifact using cached teacher outputs |
| [run_ptq_recovery.py](run_ptq_recovery.py) | Run resumable PTQ-start recovery |
| [run_ptq_witness.py](run_ptq_witness.py) | Check a calibrated donor conversion |
| [finetune_qwen_counter.py](finetune_qwen_counter.py) | Direct Qwen counter fine-tuning entry point |
| [run_tail.py](run_tail.py) | Train the floating-point model tail |
| [run_colab.py](run_colab.py) | Environment-driven recovery entry point |

The [public notebook](../notebooks/cached_kd_public.ipynb) connects the conversion, cache,
selection and reload stages. `recovery_session.py` and `kd_cache_contract.py` provide
shared helpers. `preflight_consistency.py` validates a separate archived external runner;
its [contract](../research/campaigns/PREFLIGHT_CONTRACT.md) explains the required inputs.

## Reproduce focused comparisons

| Tool | Scope |
|---|---|
| [benchmark_reversible.py](benchmark_reversible.py) | CPU reversible-backward comparison against a git revision |
| [benchmark_counter_stability.py](benchmark_counter_stability.py) | Synthetic update-rule ablation |

Exploratory benchmarks and old scale-run recipes are in
[research/experiments/](../research/experiments/README.md).

For documentation maintenance, `python scripts/check_docs.py` checks local file links
in the current guides and navigation pages. CI runs the same check.
