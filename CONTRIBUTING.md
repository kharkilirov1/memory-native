# Contributing

Thanks for helping improve Memory Native. Focused fixes, reproducible experiments and
counterexamples are welcome.

## Development setup

```bash
git clone https://github.com/kharkilirov1/memory-native.git
cd memory-native
python -m pip install -e ".[dev,donor,campaign]"
python examples/quickstart.py
python scripts/check_docs.py
python -m pytest -q
```

CI exercises Python 3.10 and 3.12. Hardware-specific tests may skip when CUDA, Triton,
MLX or distributed support is unavailable; a skip is not hardware validation.
See [reproducibility](docs/reproducibility.md) for environment details.

## Where changes belong

| Change | Location |
|---|---|
| Library behavior | `src/` with focused coverage in `tests/` |
| A small user example | `examples/` |
| User-facing guidance | `docs/` |
| A conversion, evaluation or recovery tool | `scripts/` |
| An experiment, design note or measurement | `research/` |

Read [architecture](docs/ARCHITECTURE.md) before changing update order or reversible
execution. The [contributor context](CLAUDE.md) records implementation pitfalls.

## Before opening a pull request

- Explain the concrete problem and the resulting behavior.
- Keep the change focused and preserve existing execution guards.
- Run relevant checks and report unavailable hardware or failed gates.
- Link raw evidence for numerical claims, including the workload, seed and baseline.
- Label synthetic results and separate persistent storage from peak training memory.
- Preserve negative results and historical raw artifacts.

Use commit messages such as `fix(counter): preserve carry saturation` or
`docs: simplify the quickstart`. The PR template covers rationale, validation and limits.

## Reports and discussion

Use the [issue templates](https://github.com/kharkilirov1/memory-native/issues/new/choose)
for bugs and reproductions. Scientific disagreement is welcome; explain the counterexample
or measurement. Follow [SECURITY.md](SECURITY.md) for private vulnerability reports.

Contributions are provided under the [MIT license](LICENSE). Substantial contributions
should be credited; discuss research authorship with the maintainer where appropriate.
