# Research archive

[Memory Native](../README.md) / Research

Experiments, measured results and the project's development history live here.
For installation and everyday use, start with the [documentation](../docs/README.md).

## Find a result

| Collection | Contents |
|---|---|
| [Results](results/README.md) | Measurement reports, raw logs and negative results |
| [Artifact manifest](results/ARTIFACT_MANIFEST.md) | What is public, external or missing |
| [Experiments](experiments/README.md) | Benchmarks, ablations and exploratory runners |
| [Preprint](paper/MEMORY_NATIVE_PREPRINT.md) | Method, assumptions and open questions |
| [Design notes](notes/README.md) | Earlier architecture proposals and development history |
| [Notebooks](notebooks/README.md) | Historical Colab and cloud-run recipes |
| [Campaigns](campaigns/PREFLIGHT_CONTRACT.md) | Archived external-runner configuration and preflight contracts |
| [Kaggle runs](kaggle/) | Historical kernel entry points and outputs |
| [Figures](figures/) | Charts accompanying the reports and article |

## Reading historical evidence

Reports describe the commits and environments in which they were produced. A historical
script is not necessarily a recommended current workflow. Some runs need external datasets,
checkpoints, hardware or a runner bundle that is not included here.

The layout migration preserves raw JSON, CSV, logs and figures without rewriting their
measurements or recorded source hashes. Paths embedded in those records refer to their
original checkout. Use [layout-map.json](layout-map.json) to map old paths to current ones;
the original files also remain available in git history. Source changes after a recorded
run do not retroactively validate it.

For a new experiment, record its commit, command, environment, data identity, seed and raw
outputs. Keep unsuccessful runs. Follow the [reproducibility guide](../docs/reproducibility.md).
