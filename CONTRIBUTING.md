# Contributing to memory-native

Thanks for being interested. This is a research project — an open preprint draft, not a finished product — so contributions look a little different than a typical library. The biggest open question is scientific, not engineering, and **feedback is welcomed as much as code.**

## The one thing this project most needs

> **Does the counter synapse reach convergence parity at 7B+ scale on real corpora?**

At d=512 it edges out AdamW. At 1.21B it fits and trains on a single T4. But scale-and-data convergence is marked `OPEN` in the preprint, and that is the single most valuable thing a contributor could help settle — with compute, with a sharper analysis, or with a falsifying run.

The mechanistic hypothesis (in `docs/ARTICLE.md` and queued as a follow-up): the bounded counter loses gradient pressure when weights saturate against the boundary, and that saturation rate may grow with model width. That is measurable directly on existing witnesses — no new training required.

## Ways to contribute

### 1. Read the preprint and tell me what is wrong
The fastest, highest-leverage contribution. Open [`paper/MEMORY_NATIVE_PREPRINT.md`](paper/MEMORY_NATIVE_PREPRINT.md), find the weakest claim, and tell me — either in an issue or by email. Especially:

- Claims that overstate what the T4 runs prove
- Missing related work (BitNet, QAT, sigma-delta modulation, error feedback, memory-efficient optimizers)
- Anything that makes a reviewer reject this in the first paragraph

### 2. Reproduce a result and report
Pick any row from `results/` (e.g. `SHOOTOUT.md`, `SCALE_1B.md`, `KERNEL.md`), reproduce it on your hardware, and open an issue with:
- Your hardware (GPU/CPU, memory)
- The numbers you got vs the ones in the repo
- The exact command you ran

Negative results count. If a number does not reproduce, that is a real finding.

### 3. Code, kernels, tests
The codebase is pure PyTorch + Triton. Areas that need work:

- **Triton kernels** in `src/memory_native/triton_counter.py` and `fused_update.py` — correctness, new shapes, performance on newer GPUs (H100/Blackwell)
- **Tests** — CUDA-only paths are currently skipped on CPU; better CPU fallbacks welcome
- **Instrumentation** for the saturation analysis (described in the article)
- **Docs** — typos, clarity, examples

### 4. The open research question
If you have compute (one T4 / A10g / L4 is enough), instrumenting the counter layer to report saturation rate per step and sweeping model width is a self-contained experiment. See the follow-up section in `docs/ARTICLE.md`.

## Development setup

```bash
git clone https://github.com/kharkilirov1/memory-native.git
cd memory-native
pip install -e ".[dev]"
pytest                          # 139 passed, 12 skipped (CUDA-only)
```

- Python ≥ 3.9
- `torch>=2.1` and `numpy>=1.21`
- Optional: CUDA GPU for the GPU-only tests (the 12 skipped ones)
- Optional: `triton` for the fused kernels (falls back to torch path if absent)

## Before you open a PR

- **Tests pass.** `pytest` locally. If you add a feature, add a test.
- **No fabricated numbers.** Every empirical claim should point at a witness in `results/` or at a log you are adding. If a result is toy/synthetic, mark it as such.
- **Honest about what is proven vs open.** If your change touches a claim, update the preprint's `OPEN` markers rather than overstating.
- **One thing per PR.** Easier to review, easier to revert.

## Commit and PR style

- Commit messages: `type(scope): subject` — e.g. `feat(counter): add C=22 variant`, `fix(triton): correct grad_x stride`, `docs(preprint): clarify saturation hypothesis`.
- PR description: what changed, why, what was tested, what is left open.

## Discussion and conduct

- Be honest and direct. This project is improved by disagreement, not by politeness that hides problems.
- If you find a real bug in the method — not just the code — say so plainly. I would rather know today than after publication.
- Credit is given. If you contribute a non-trivial experiment or fix, you go in the preprint's acknowledgments (or as co-author for substantial research contributions, on request).

## Licensing contributions

All contributions are submitted under the project's [MIT License](LICENSE). By contributing you confirm your changes are yours to license that way.

---

**Repo:** [github.com/kharkilirov1/memory-native](https://github.com/kharkilirov1/memory-native)
**Preprint:** [`paper/MEMORY_NATIVE_PREPRINT.md`](paper/MEMORY_NATIVE_PREPRINT.md)
**Issues:** [github.com/kharkilirov1/memory-native/issues](https://github.com/kharkilirov1/memory-native/issues)
