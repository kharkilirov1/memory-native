# Examples

[Memory Native](../README.md) / Examples

Run examples from the repository root after installing the package.

| Example | Command | Requirements |
|---|---|---|
| [Counter learning](quickstart.py) | `python examples/quickstart.py` | PyTorch; offline CPU by default |
| [MLX layer demo](mlx.py) | `python examples/mlx.py` | Optional MLX runtime; see the [port guide](../docs/MLX_PORT.md) |

The counter example prints error and storage separately. Use `--device cuda` with a
configured CUDA environment. These examples demonstrate operations on small inputs;
performance and quality claims require their own measurements.
