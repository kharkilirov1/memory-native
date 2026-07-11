"""Warm-start Qwen2.5-0.5B into the counter format, then recover its quality by distillation.

The runnable T4 artifact for the finetune-pretrained initiative (design:
docs/superpowers/specs/2026-07-06-qwen-counter-recovery-design.md). It produces the end-to-end
witness -- a PPL-recovery curve on held-out WikiText:

  PPL(fp Qwen)              baseline (the donor's own quality)
  PPL(counter, warm-start)  degraded  -- ternarization is lossy for a full-precision donor
  PPL(counter, distilled)   recovered -- pulled back toward the baseline by the distill finetune

Recovery is a NETWORK-level effect (composed layers + distill loss), not a per-layer self-fix
(see CLAUDE.md). The counter body self-updates inside backward (no Adam moments, no fp master, no
full grad buffer); only the fp slice (embeddings / final norm / tied lm_head / preserved q,k,v
bias) is trained by AdamW -- exactly the pools the method is designed NOT to zero.

Single-GPU by design: the 0.5B fp teacher (~1 GiB in bf16) sits resident beside the counter
student on a 16 GiB T4. For larger donors, replace ResidentTeacher with an offline top-k logit
cache (the TeacherSource seam is already there).

Run:  python scripts/finetune_qwen_counter.py
Deps: pip install -e ".[donor]" datasets   (transformers, safetensors, accelerate, datasets)
"""
import math
import sys

import torch

# ---------------- config ----------------
MODEL = "Qwen/Qwen2.5-0.5B"          # dense, open weights (not gated); base = meaningful PPL
DATASET = ("wikitext", "wikitext-2-raw-v1")
SEQ = 512                             # context length for train + eval batches
BATCH = 8
TRAIN_TOKENS = 400_000               # rolling-buffer budget for the distill corpus
VAL_TOKENS = 120_000                 # held-out slice for the PPL witness
ROUNDS, STEPS_PER_ROUND = 8, 50      # distill schedule; PPL is measured after each round
KD_ALPHA, CE_ALPHA, TEMPERATURE = 1.0, 0.3, 2.0
# Recovering from a heavily degraded warm-start (ternarizing a full-precision 0.5B donor sends PPL
# to ~1e5) DIVERGES at a naive lr with no clipping -- verified on this box: PPL exploded to 5e13.
# This recipe is stable and cut PPL ~100x in 40 CPU steps (186k -> ~1.5k). Full recovery to the fp
# baseline is a GPU-scale run (thousands of steps).
LR = 3e-4                           # AdamW lr for the fp params (embed/norm/lm_head/bias)
COUNTER_LR = 0.008                  # per-counter-layer update lr (override of the 0.04 default)
LOCAL_GRAD_CLIP = 1.0               # per-counter-layer row-grad clip
GRAD_CLIP = 1.0                     # fp-param global grad-norm clip before each AdamW step
THRESHOLD_RATIO = 0.5               # TWN keep-threshold: 0.5 gave the least-bad warm-start here
SEED = 0
# --- perf knobs (verified on CPU: sdpa+cache = 1.79x/step; see results/perf_audit_cpu.md) ---
ATTN_IMPL = "sdpa"                  # sdpa calls each projection once -> counter guard holds
TEACHER_TOPK = 128                  # cache teacher top-k logits; epoch 2+ skips the teacher
                                    # forward entirely (0 = off, exact full-vocab KD every step)
KIND = "counter_rms"                # "counter_packed" adds 6-bit storage + a fused Triton
                                    # update on CUDA, BUT the fused kernel requires
                                    # local_grad_clip=0 -- a separate T4 stability experiment.


def _pip(*pkgs):
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def load_corpus(tokenizer, device):
    """WikiText-2 raw -> Qwen BPE -> contiguous [B, SEQ] batches for train and held-out eval."""
    try:
        from datasets import load_dataset
    except Exception:
        _pip("datasets"); from datasets import load_dataset

    name, cfg = DATASET
    def token_stream(split, budget):
        ds = load_dataset(name, cfg, split=split)
        buf, out = [], []
        need = budget
        for row in ds:
            text = row["text"]
            if not text:
                continue
            buf.extend(tokenizer(text)["input_ids"])
            while len(buf) >= SEQ * BATCH:
                chunk = buf[: SEQ * BATCH]; buf = buf[SEQ * BATCH:]
                out.append(torch.tensor(chunk, dtype=torch.long, device=device).view(BATCH, SEQ))
                need -= SEQ * BATCH
                if need <= 0:
                    return out
        return out

    train = token_stream("train", TRAIN_TOKENS)
    val = token_stream("test", VAL_TOKENS)
    return train, val


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        _pip("transformers", "safetensors", "accelerate")
        from transformers import AutoModelForCausalLM, AutoTokenizer

    from memory_native.donor.qwen import qwen_to_counter
    from memory_native.eval import perplexity
    from memory_native.recovery import ResidentTeacher, TopKLogitCache, distill_finetune

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # teacher: the untouched fp donor, resident and frozen
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation=ATTN_IMPL
    ).to(device).eval()

    # student: a second copy, warm-started in place into the counter format
    # (qwen_to_counter defaults cache_mode="fp16": forward reads the derived T-cache
    # instead of re-decoding the packed state every call)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation=ATTN_IMPL
    ).to(device)
    report = qwen_to_counter(student, kind=KIND, threshold_ratio=THRESHOLD_RATIO,
                             lr=COUNTER_LR, local_grad_clip=LOCAL_GRAD_CLIP)
    print(report)

    train_batches, val_batches = load_corpus(tokenizer, device)
    print(f"corpus: {len(train_batches)} train batches, {len(val_batches)} val batches "
          f"({BATCH}x{SEQ})")

    ppl_fp = perplexity(teacher, val_batches)
    ppl_warm = perplexity(student, val_batches)
    print(f"\nPPL fp teacher       : {ppl_fp:8.3f}")
    print(f"PPL counter warm-start: {ppl_warm:8.3f}   (degraded by {ppl_warm - ppl_fp:+.3f})")

    # one cache OUTSIDE the round loop: batches repeat every ~97 steps, so rounds 2+ replay
    # cached top-k logits instead of running the 0.5B teacher forward each step.
    teacher_src = ResidentTeacher(teacher)
    if TEACHER_TOPK:
        teacher_src = TopKLogitCache(teacher_src, k=TEACHER_TOPK)
    curve = [ppl_warm]
    for r in range(ROUNDS):
        distill_finetune(
            student, teacher_src, train_batches,
            steps=STEPS_PER_ROUND, kd_alpha=KD_ALPHA, ce_alpha=CE_ALPHA,
            temperature=TEMPERATURE, lr=LR, grad_clip=GRAD_CLIP,
        )
        ppl = perplexity(student, val_batches)
        curve.append(ppl)
        gap = (ppl - ppl_fp) / max(ppl_warm - ppl_fp, 1e-9)
        print(f"round {r+1:2d}/{ROUNDS}  PPL {ppl:8.3f}   residual gap {gap*100:5.1f}%")

    if TEACHER_TOPK:
        print(f"teacher top-{TEACHER_TOPK} cache: {teacher_src.hits} hits / "
              f"{teacher_src.misses} misses (hits skip the teacher forward)")
    recovered = 1.0 - (curve[-1] - ppl_fp) / max(ppl_warm - ppl_fp, 1e-9)
    print(f"\nrecovery: {curve[0]:.3f} -> {curve[-1]:.3f}  (closed {recovered*100:.1f}% of the "
          f"warm-start gap to the fp baseline {ppl_fp:.3f})")
    print("curve:", "  ".join(f"{p:.2f}" for p in curve))


if __name__ == "__main__":
    main()
