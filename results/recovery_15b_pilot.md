# Qwen2.5-1.5B counter-recovery — A100 Colab pilot (2026-07-13)

End-to-end witness on a serious donor: warm-start Qwen2.5-1.5B into 6-bit counter synapses
(`counter_packed`, fused Triton update WITH the row clip), recover by KD-distillation on a
pretraining-like EN/RU/code/math mix (12M-token pilot corpus). A100 40 GB, bf16.

Config: kind=counter_packed, cache_mode=int8, thr=0.5, counter lr=0.008, local_grad_clip=1.0,
fp lr=3e-4 (AdamW on embeddings/norm/tied lm_head), KD(T=2) + 0.3*CE, B4xT512, MAX_HOURS=0.8.
Corpus: FineWeb-Edu 55% / FineWeb-2 rus_Cyrl 25% / codeparrot-clean 15% / OpenWebMath 5%.

## Warm-start swap
196 nn.Linear -> counter_packed, 1,310,195,712 counter coeffs (skipped=1 tied lm_head).
GPU mem after build: 5.5 GiB. epoch = 5865 steps.

## Per-domain PPL: fp teacher -> counter warm-start -> distilled

| domain | fp teacher | counter warm-start | pilot recovered (step 4000, ~8.4M tok) |
|---|---:|---:|---:|
| EN   | 10.8 |    576,003 | 143.4 |
| RU   |  8.5 | 20,670,061 |  76.8 |
| code |  2.6 |    454,347 |  50.8 |
| math |  6.0 |    369,763 | 123.3 |

Ternarizing a full-precision 1.5B donor degrades it ~5e4-2e6x per domain; ~8.4M tokens of
distillation into the counter format recover it to 50-140 PPL -- a ~3e3-3e5x recovery, all
domains alive (RU/code/math did NOT collapse; the mixed corpus preserves every ability).

## Recovery curve (eval every 400 steps)
- step  400: EN 306.9  RU 297.0  code 517.2  math 576.6
- step 3600: EN 139.6  RU  75.5  code  51.0  math 120.6
- step 4000: EN 143.4  RU  76.8  code  50.8  math 123.3   <- plateau on the 12M pilot corpus
- step 4123: [deadline] wall budget (0.8h) reached; Colab then dropped the session before the
  final report/generations printed (ephemeral /content checkpoint lost -- irrelevant for a
  calibration pilot).

## Calibration for the main run
- **0.62 s/step at B4xT512** on A100 (~3300 tok/s), ~20% MFU -> headroom to raise the batch.
- B8xT1024 OOMs a 40 GB A100 (full-vocab KD logits [8,1024,151936] + backward); B4xT512 fits
  with large headroom (build 5.5 GiB). counter layers are eager-only so gradient checkpointing
  is unavailable -- the batch is the only activation-memory lever.
- The pilot plateaued at ~50-140 PPL on 12M tokens; the main run uses a 150M-token corpus
  (EN 82.5M / RU 37.5M / code 22.5M / math 7.5M) -> expect low-tens/single-digit PPL.
- **Colab drops the session well before 10.5h** -> the main run MUST checkpoint to Drive and
  resume (the deterministic domain sampler reads exactly the untouched remainder on resume);
  a 10h run is a series of reconnect+resume segments, not one continuous session.

Artifacts: scripts/build_mix_corpus.py, scripts/recovery_session.py, scripts/run_colab.py,
notebooks/qwen15_recovery_colab.ipynb; corpora as GitHub release assets pilot-corpus / main-corpus.
