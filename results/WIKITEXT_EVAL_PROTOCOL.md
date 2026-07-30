# Standardized WikiText-2 eval — protocol, first baseline, comparison plan

Date: 2026-07-30. Harness: `scripts/eval_wikitext_ppl.py`.

## Why

Every PPL number in results/ so far lives on custom val slices — perfect for internal
ladders (relative, same-slice), unusable next to published GPTQ/AWQ/AQLM/QuIP# tables.
This harness is the standard protocol those tables use (GPTQ-paper convention):
wikitext-2-raw-v1 TEST, docs joined with "\n\n", one token stream, non-overlapping
seq-2048 windows, PPL = exp(mean NLL over all next-token predictions).

## First baseline (this box, CPU)

| model | arm | seq | windows | scored tokens | ppl |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | fp32 | 2048 | 20 of 138 | 40,940 | **12.70** |

Sanity: published full-set numbers for Qwen2.5-0.5B sit ~13 — the harness agrees.
Bounded runs MUST report the window count (this one: 20); full-set is one command on a
faster box (`MAX_WINDOWS=0`).

## How counter models plug in

- streamed conversion: `STATE_DIR=<out_dir>` (restore_counter_structure + eval);
- recovery checkpoint: `CKPT=<ckpt.pt>`;
- always strict `ALPHA=0` (the only deployable setting — project rule).

The 0.5B streamed state produced in this session was calibrated on RANDOM ids (restore-
machinery witness, results/RESTORE_WITNESS.md) — its ppl (856 @ seq 512) is a plumbing
number, NOT the solver's. A comparability-grade counter number needs a real-calibration
conversion at the deploy config (SALIENT_FIRST=0.02, asym s=0.15, 2 passes).

## The comparison the record still needs (the blind spot)

1. Convert Qwen2.5-1.5B (or 0.5B) with the deploy solver config on REAL calibration data
   and run this harness full-set → the first counter number on the standard protocol.
2. Baselines at matched bpw (~3.1-3.6 with 2% salient): GPTQ 3-bit g128 and AWQ 3-bit via
   their reference implementations, same protocol, same seq. AQLM/QuIP# 2-3 bit numbers
   can be quoted from their papers once our number is on the same protocol.
3. Position honestly: if the counter format is within noise of tuned 3-bit PTQ, the
   method's claim stays where it is strongest — the TRAINING pools (optim 0, grad 0,
   reversible activations, now a one-launch group-local update) — with PTQ quality as
   "competitive", not the headline.
