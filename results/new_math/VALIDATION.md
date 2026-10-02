# Validation and evidence inventory

- Full local suite: **525 passed, 21 skipped**, 85.03 seconds. The 14 warnings
  are existing small-budget MoE calibration witnesses. Raw output:
  [validation_cpu.log](validation_cpu.log).
- New operator/protocol tests with warnings treated as errors: **91 passed**.
  Raw output: [validation_focused.log](validation_focused.log).
- Primary run coverage: **24/24** completed main runs, **24/24** LR-tuning
  runs, **42/42** two-teacher regression runs.
- Additional diagnostics: **8/8** rank/C runs and **8/8** paired
  covariance/scale runs, each with a fixed 1,000-step horizon.
- Corpus SHA-256, fixed evaluation-window hashes, source hashes and selected
  LRs are retained. All recorded research source hashes were compared with
  the final files and matched. The capacity study exactly reproduces the
  primary floating/counter seed-0 regression at step 400.

Runtime was Python 3.12.14, PyTorch 2.14.1+cpu and NumPy 2.5.3. CUDA was not
available. Primary families used three simultaneous one-thread processes
pinned to separate cores; their command capture and execution description are
in [CAMPAIGN.json](CAMPAIGN.json). Plot generation uses the optional research
dependency, matplotlib. CPU wall times include shared-host uncertainty.

The Git base of this research branch is
`df41120c8ea8e4eb2937071a2b89367869633cd8` (merged fixes, PR #3).
The primary operator/protocol implementation is committed as
`d9ca61709a96df10fd5f6440d3da2cd5b70dcec9`. Diagnostic manifests identify the
Git HEAD present at execution and separately hash research additions that had
not yet been committed. Those exact additional source files are included in
this branch; a manifest's Git HEAD alone is not a complete diagnostic snapshot.

The new tests check scalar finite differences away from PAM boundaries,
independent forward/backward reconstruction, surrogate oracles, counter
pre-update gradient ordering, checkpoint semantics, batch-stream independence,
parameter budgets and local-metric learning signals. They do not establish
GPU or reduced-precision behavior. CUDA/Metal-specific existing tests were
skipped on this host.

Interpretation: [INTERPRETATION.md](INTERPRETATION.md).
Primary raw tables/plots: [REPORT.md](REPORT.md), [SUMMARY.json](SUMMARY.json).
Additional raw evidence: [capacity_ablation](capacity_ablation/README.md),
[covariance_ablation](covariance_ablation/README.md).
