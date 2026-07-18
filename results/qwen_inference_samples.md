# Inference witness — what the model actually outputs (3 stages)

Date: 2026-07-06 · Qwen2.5-0.5B · CPU · greedy, 40 new tokens, rep-penalty 1.3
Recovery: 120 distill steps on **English WikiText-2 only** (the same recipe as
results/qwen_recovery_cpu.md). Run time 38.4 min.

## Stage 1 — fp ORIGINAL (teacher)

- EN "The capital of France is" → "Paris. It was founded in 843 by Charles Martel…" (coherent; fact wrong — 0.5B is small)
- RU "Столица России — это город" → "…который не может быть без зданий. Всего в Москве насчитывается 120 зданий…" (coherent Russian)
- CODE "def fibonacci(n):" → "if n == 0: return [1] … elif (n==2 or n<=3): return [1,1] … else:" (Python-ish)

→ Original does EN + RU + code.

## Stage 2 — counter WARM-START (ternarized, no recovery)

- EN → "P  N  C  0  A" — garbage
- RU → "C WaC AAW, Hary ( B C cr S C…" — garbage
- CODE → "Sock -[Viewese: slib … 视 t" — garbage

→ Ternarization destroys the model wholesale, every language at once.

## Stage 3 — counter RECOVERED (after English-only distill)

- EN → "the Blue Jackets, and was a 2013. … The team had been injured in an NHL game … Columbus won…"
- RU "Столица России" → "the Blue Jackets was a 2013 … NHL … Columbus were unable to play…" (ENGLISH, not Russian)
- CODE "def fibonacci(n):" → "The Blue Jackets was the team in a 2013 … NHL … Columbus had been injured…" (English prose, not code)

## Verdict — the metric lied

English grammar came back (garbage → fluent sentences), BUT:
1. **Overfit to the tiny corpus.** WikiText-2 is full of ice-hockey articles (Columbus Blue
   Jackets). With only 40 cycled batches, the model collapsed onto that topic — it now says
   "Blue Jackets / NHL / Columbus / 2013" for *every* prompt.
2. **Russian is gone.** A Russian prompt gets English hockey text back.
3. **Code is gone.** `def fibonacci` gets English hockey text back.

So the "99.4% PPL gap closed" on WikiText was **misleading**: PPL fell because the model
memorized *that* text, not because it recovered general ability. Recovery on a narrow, single-
language corpus = catastrophic forgetting of everything else + topic collapse.

## Implication for a real recovery

The distill corpus must be **diverse and close to the donor's pretraining mix** (multilingual
text + code + varied domains), and large enough not to overfit. A single small English set
recovers English fluency at the cost of Russian, code, and topical breadth. This is a data
problem, not a method problem — but it means "recovered" must be judged by broad generation,
not by PPL on the very set you trained on.
