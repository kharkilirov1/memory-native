# Review-fix validation (2026-10-02)

The combined review-fix working tree passed **434 tests, with 21 skips and 14
warnings**, in 75.15 seconds. The skips cover unavailable CUDA/Triton/Metal and
other optional hardware paths; the warnings are existing small-calibration MoE
rank-deficiency notices. [Raw pytest output](review_fixes_cpu_tests.log) is included.
No GPU or Metal performance/quality result is inferred from this CPU run.

Environment: Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.5.3, pytest 9.1.1,
Transformers 5.18.0, safetensors 0.8.0, accelerate 1.15.0, PyYAML 6.0.3,
MLX 0.32.3 CPU, tokenizers 0.23.2. CUDA was unavailable. The test command used
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`.

The [public notebook](../notebooks/cached_kd_public.ipynb) was also executed cell by
cell in the installed environment, skipping only its redundant pip-install line.
[Raw smoke output](review_fixes_public_smoke.log) includes conversion, teacher
cache, four actual backward passes, mandatory warm selection, and selected-model
reload. It used a random two-layer Qwen2 model (width 64, vocabulary 128), synthetic
uint32 bins, batch 1, sequence 16, group 32 and top-K 32. The selected model remained
the warm model because the small changes did not exceed the 0.001 acceptance
margin. Reload reproduced its mean log perplexity, 4.881365966796875.

These random smoke metrics validate the artifact path, not language quality or
convergence. Source files were uncommitted during execution: the printed source
HEAD is the integration parent and `run_manifest.json` recorded a dirty tree. The
review-fix commit contains the tested source; CI checks that committed tree.

Tests cover learned-scale latent expectations, six-bit bounds, hash-SR seed
serialization, classic/cascaded conversion resume, content/provenance and corruption
checks, tokenizer semantics, FP32-normalized KD/CE values and gradients, warm
fallback, accepted slim-checkpoint reload, and fail-closed external preflight.
