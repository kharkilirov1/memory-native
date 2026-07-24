# Security policy

## Supported versions

`memory-native` is pre-1.0 research software. Security fixes are applied to the current `main`
branch; older commits and experimental result snapshots are not supported release lines.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose user data, execute untrusted
code, corrupt checkpoints silently, or compromise a training host.

Use GitHub's **private vulnerability reporting** for this repository when available. If it is not
available, contact the maintainer at the address listed in `CITATION.cff` with subject
`memory-native security report`.

Include:

- affected commit or version;
- threat model and impact;
- minimal reproduction;
- whether untrusted model, dataset, checkpoint, or path input is required;
- any proposed mitigation.

Please allow a reasonable coordination window before public disclosure.

## Security scope

The project loads local datasets, model artifacts, and checkpoints through Python and optional
third-party ML libraries. Treat untrusted pickle-based PyTorch artifacts as executable content.
Prefer formats and loading modes that do not execute arbitrary Python objects.

The project does not claim hardened sandboxing for untrusted:

- Python packages or scripts;
- PyTorch checkpoints;
- model repositories with custom remote code;
- datasets;
- Triton or Metal kernels.

Dependency vulnerabilities in PyTorch, NumPy, Transformers, Triton, MLX, or CUDA should also be
reported upstream when appropriate.
