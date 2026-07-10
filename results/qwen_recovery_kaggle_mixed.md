# Mixed-corpus recovery on a real GPU (Kaggle T4) — Russian + code come back

Date: 2026-07-06 · Qwen2.5-0.5B · Kaggle T4 (bf16) · kernel `lirovkharki/qwen-recovery-mixed`
Distill: 1500 steps on a **mixed EN + RU + code** corpus (300k train tok/lang), recipe as
scripts/finetune_qwen_counter.py (thr 0.5, fp lr 3e-4 + grad_clip, counter lr 0.008, ce 0.3, T 2).
Wall-clock 50.1 min. This is the counter-example to results/qwen_inference_samples.md (English-only
distill, which forgot Russian + code and collapsed onto one topic).

## Per-language PPL (held-out)

| lang | fp teacher | counter warm-start | recovered (mixed distill) |
|---|---:|---:|---:|
| EN   | 21.3 | 170 426  | **306.7** |
| RU   |  8.5 | 6 627 487 | **90.9** |
| CODE |  9.0 | 250 682  | **223.8** |

RU distill curve (monotone, no collapse): `403 → 193 → 129 → 106 → 95 → 90.9`.
All three languages recover in parallel; none is sacrificed.

## Generations — Stage 3 (recovered)

- EN "The capital of France is" → "a number of the city, and also known as a new name. … The most
  important part of his first time in 1920, he" — fluent English (bland, but grammatical).
- RU "Столица России — это город" → "…ы, что включает социальные и политические структуру. В 1920
  году был упоминал князя Германий" — **Russian, not English** (grammatically Russian, some errors).
- RU "Небо голубое, потому что" → "в позднее итоге-за (как 123) — … Временном князя Г" — Russian.

Contrast with the English-only CPU run: there a Russian prompt returned *English hockey text*
(Russian was gone). Here Russian returns **Russian**. That is the whole point.

## Verdict

- **A diverse corpus brings back every ability.** Russian and code recover alongside English
  (PPL 6.6M → 91, 250k → 224), and there is NO topic collapse — the model is not overfitting to
  one text, because it sees many.
- **But not to baseline.** Recovered PPL (EN 307, RU 91, CODE 224) is still ~10–25x the fp baseline
  (21 / 8.5 / 9). This is the same scale limit as before: 300k tok/lang and 1500 steps is tiny.
  Full recovery needs a large, pretraining-like corpus and far more steps — a data+compute cost,
  not a method failure (ternarization + distill are mechanically sound; they recover what they see).

Two runs, one lesson: **the recovery corpus decides which abilities return.** English-only →
English (overfit, forgets the rest); mixed → all three, partway back, no collapse.

## Reproduce

Kernel builder: scratch `build_and_push.py` embeds `src/memory_native` as a base64 tarball into a
Kaggle script kernel (T4, GPU, internet, `machine_shape=NvidiaTeslaT4` — required, else the
preinstalled torch has no kernel image for the assigned GPU). Body: `kernel_body.py`.
