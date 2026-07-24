# Maintainer application packet

This is working copy for open-source support applications. Verify all repository metrics and
program terms immediately before submitting. Do not replace adoption evidence with benchmark
claims: they answer different questions.

## Canonical project

- Repository: https://github.com/kharkilirov1/memory-native
- Role: primary maintainer
- License: MIT
- Primary artifact: `memory-native`, the reference PyTorch implementation
- Supporting project: MotifCL, a separate Vulkan-first native runtime

## One-sentence description

`memory-native` is an MIT-licensed research implementation of six-bit finite-state synapses and
reversible activations for reducing persistent and activation memory during neural-network
training without FP master weights or Adam moments.

## Evidence to link

- Method and limitations: `paper/MEMORY_NATIVE_PREPRINT.md`
- Current status and claims boundary: `PROJECT_STATUS.md`
- Reproduction protocol: `REPRODUCIBILITY.md`
- Hardware witnesses: `results/SCALE_1B.md`, `results/SHOOTOUT.md`, `results/POOLS.md`,
  `results/KERNEL.md`, `results/ACCELERATION.md`, `results/group_kernel_opt_stage01.md`
- Contribution workflow: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Citation metadata: `CITATION.cff`

## OpenAI Codex for Open Source

The public form currently limits each narrative field to 500 characters. Recount characters
after editing.

### Why does this repository qualify?

> I am the primary maintainer of memory-native, an active MIT-licensed research project exploring six-bit finite-state synapses and reversible training. The repository includes a PyTorch/MLX implementation, tests, a preprint, raw hardware witnesses, reproducibility instructions, and explicit negative/open results. Its ecosystem value is a falsifiable, low-memory training method for researchers working without modern datacenter GPUs.

### How will you use API credits?

> I will use Codex API credits for reproducibility automation, regression triage, review of external reproduction reports, test generation across PyTorch/MLX/Vulkan boundaries, security review, and release preparation. The highest-value work is auditing empirical claims against raw logs and maintaining CPU gates plus hardware-specific witnesses without presenting skipped tests or modeled estimates as measured results.

### Anything else?

> memory-native is the primary project; MotifCL is a supporting Vulkan runtime used to test compact-state execution on legacy AMD hardware. I maintain both, but keep their evidence separate. The project is early and does not claim broad adoption. Support would primarily reduce the maintenance burden of turning research scripts and hardware runs into reviewable, reproducible OSS artifacts.

## Anthropic Claude for Open Source

As of the last verification, the published quantitative routes include dependent repositories or
packages/downloads, recognized foundation roles, 100 merged external pull requests, 20 external
contributors, or an OpenSSF criticality score threshold. This repository is early-stage and does
not currently have evidence that it meets those thresholds.

Use the program's discretionary route only with explicit honesty:

> I maintain memory-native, an early MIT-licensed research implementation of finite-state optimizer-in-weight training. It does not yet meet the program's published adoption thresholds, so I am applying under the “doesn't quite fit” route. The repository's value is an unusually evidence-heavy, falsifiable implementation for low-memory training, with raw T4 witnesses, CPU tests, a preprint, explicit open questions, and a separate Vulkan deployment project.

Do not claim ecosystem dependence, download volume, external contributors, or criticality without
a current primary-source measurement.

## Pre-submission checklist

- [ ] Public GitHub profile and canonical repository are visible.
- [ ] Strongest local changes intended for the application are reviewed and pushed.
- [ ] Default branch CI is green.
- [ ] Repository description and topics match the README.
- [ ] Private vulnerability reporting is enabled in GitHub settings.
- [ ] At least one tagged release has reproducible release notes.
- [ ] Application metrics are refreshed from GitHub/registry/OpenSSF primary sources.
- [ ] Every submitted number links to a public witness.
- [ ] OpenAI Organization ID is available if API credits are requested.
- [ ] No application is submitted automatically; the maintainer reviews the final text.
