# Counter-BLAST capacity and optimization diagnostic

Eight predeclared CPU runs, seed 0, 1,000 steps, dimension 32, block size 8,
batch size 32. The same Gaussian inputs, 1,536 training rows and 512 held-out
rows are used throughout. The teacher is floating BLAST with factor rank 4
and global rank 16. Its weight SHA-256 is
`16f6a74a5aec196d11ec9a7e81a7f190ebf722ab3765dadb709605a1dc2182db`.

All floating factors use AdamW without weight decay. Every counter factor uses
the existing RMS counter update, `lr_scale=2e-4`, and no clipping. There is no
early stopping or LR selection. The table reports the final fixed-horizon
relative MSE, normalized by the corresponding split's mean squared target.

| Factors | Factor rank | C | LR | Coefficients | Train MSE | Held-out MSE | Initial → final visible rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Floating | 4 | — | 0.01 | 320 | 0.013364 | 0.013899 | 16 → 16 |
| Floating | 8 | — | 0.01 | 640 | 0.000291 | 0.000299 | 32 → 19 |
| All counter | 4 | 8 | 0.03 | 320 | 0.527142 | 0.537664 | 13 → 13 |
| All counter | 4 | 8 | 0.10 | 320 | 0.499493 | 0.511197 | 13 → 13 |
| All counter | 4 | 4 | 0.03 | 320 | 0.468374 | 0.482056 | 13 → 15 |
| All counter | 4 | 2 | 0.03 | 320 | 0.519823 | 0.533777 | 13 → 14 |
| All counter | 4 | 2 | 0.10 | 320 | 0.537388 | 0.550808 | 13 → 11 |
| All counter | 8 | 2 | 0.03 | 640 | 0.377658 | 0.389918 | 28 → 28 |

The global rank bounds are 16 and 32 for factor ranks 4 and 8 respectively.
Visible ranks use `torch.linalg.matrix_rank` with its default FP32 tolerance.
The rank-4 counter arms have exactly the same initial visible operator; changing
C changes the hidden accumulator, not the available visible ternary values.

The additional horizon, smaller C and larger counter LR do not close the gap
to floating factors. Counter rank-4 error fluctuates rather than converging:
the C=8, LR=0.03 baseline goes from 0.418183 at step 400 to 0.537664 at step
1,000. At step 400, both that value and floating rank-4 MSE 0.054472 exactly
reproduce the primary regression's seed-0 results.

Random ternary mixing matrices introduce a real initial bottleneck: rank-4
counter S factors begin with ranks `[3, 4, 3, 3]`, giving visible rank 13 of
the bound 16. This is insufficient to explain the whole gap. The expanded
counter arm reaches rank 28, above the teacher's rank 16, but still has MSE
0.389918. Its doubled coefficient count must be included in comparisons.
Training and held-out errors are close, so this failure appears in fitting
the training operator as well as in generalization.

Scale learning is a material limitation of the protocol: final counter scales
remain within 0.68–0.91% of their initialized values. These results mostly test
a nearly fixed-amplitude ternary factor family. They do not establish an
intrinsic impossibility of counter BLAST or separate representability limits
from optimization limits. Scale schedules, hybrid factors, more seeds and
other tasks remain untested. There is no GPU or language-model quality claim.

Each JSON contains the full curve, initial/final singular values and factor
ranks, flip/scale statistics and resource accounting. `manifest.json` records
the predeclared arms, source hashes, data hashes and source commit
`d9ca61709a96df10fd5f6440d3da2cd5b70dcec9`. The run used Python 3.12.14,
PyTorch 2.14.1+cpu, NumPy 2.5.3, one thread and CPU affinity `[3]`.

From the repository root, reproduce into a fresh directory:

```bash
taskset -c 3 env PYTHONPATH=src python scripts/new_math_capacity_ablation.py \
  --output results/new_math/capacity_ablation_reproduction \
  --steps 1000 --seed 0 --teacher blast
```

Use an available CPU number if CPU 3 is absent; numerical results should remain
deterministic on the same software stack, while timings may differ. Existing
output directories are refused to preserve the recorded evidence.
