"""Build the recovery-distillation mix corpus as per-domain token bins.

The 0.5B runs proved the corpus decides which abilities come back (English-only WikiText
collapsed RU/code; the EN+RU+code mix restored them). This builds a pretraining-like mix
for the 1.5B recovery: FineWeb-Edu (EN) + FineWeb-2 (RU) + codeparrot-clean (Python) +
OpenWebMath, tokenized with the donor's own BPE, EOS-joined, written as uint32 streams --
one train bin per domain plus a held-out val bin (val documents never enter train).

Interleaving happens at load time (the notebook samples a domain per sequence by weight),
so proportions stay tunable without rebuilding.

Usage:
  python scripts/build_mix_corpus.py --out data/mix_pilot --train-tokens 12_000_000
  python scripts/build_mix_corpus.py --out data/mix_full  --train-tokens 150_000_000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")  # tokenizer donor; corpus bins are donor-BPE-specific
# (domain, train share, dataset, load kwargs, text field)
# v2 mix: domain coverage IS ability coverage (measured). FineMath-4+ replaces OpenWebMath
# (stronger per-token, HF SmolLM2 ablations); peS2o adds the science register; smoltalk
# (field "messages" -> rendered through the donor's chat template) keeps instruct-tuned
# donors in distribution -- KD on raw web text alone drifts an instruct model's register.
# v3 rebalance: ru 0.20 -> 0.30 (the 1.5B campaign measured RU as the widest teacher gap;
# the share feeds BOTH the KD sampling weights and the solver's Hessian calibration).
SOURCES = [
    ("en",       0.40, "HuggingFaceFW/fineweb-edu",   dict(name="sample-10BT", split="train"),      "text"),
    ("ru",       0.30, "HuggingFaceFW/fineweb-2",     dict(name="rus_Cyrl", split="train"),         "text"),
    ("code",     0.12, "codeparrot/codeparrot-clean", dict(split="train"),                          "content"),
    ("math",     0.08, "HuggingFaceTB/finemath",      dict(name="finemath-4plus", split="train"),   "text"),
    ("science",  0.05, "HuggingFaceTB/smollm-corpus", dict(name="cosmopedia-v2", split="train"),    "text"),
    ("instruct", 0.05, "HuggingFaceTB/smoltalk",      dict(name="all", split="train"),              "messages"),
]
BATCH_DOCS = 64          # tokenizer batch (fast tokenizer parallelizes inside)
MAX_DOC_CHARS = 60_000   # clip pathological documents; keeps the stream diverse


def build_domain(tokenizer, eos: int, name: str, dataset: str, load_kw: dict, field: str,
                 train_budget: int, val_budget: int, out_dir: str) -> dict:
    from datasets import load_dataset
    ds = load_dataset(dataset, streaming=True, **load_kw)
    it = iter(ds)

    def extract(row) -> str:
        if field == "messages":                  # chat data: render through the donor template
            try:
                return tokenizer.apply_chat_template(row["messages"], tokenize=False)
            except Exception:
                # base checkpoints may ship no chat template (e.g. gemma-4 base):
                # fall back to a plain role-prefixed rendering instead of dropping the domain
                try:
                    return "\n\n".join(
                        f"{m['role']}: {m['content']}" for m in row["messages"]
                    )
                except Exception:
                    return ""
        return row.get(field) or ""

    def token_batches():
        buf = []
        for row in it:
            text = extract(row)
            if len(text) < 200:                  # skip near-empty docs
                continue
            buf.append(text[:MAX_DOC_CHARS])
            if len(buf) >= BATCH_DOCS:
                for ids in tokenizer(buf)["input_ids"]:
                    yield ids
                buf.clear()

    t0 = time.perf_counter()
    counts = {}
    gen = token_batches()
    for split, budget in (("val", val_budget), ("train", train_budget)):
        path = os.path.join(out_dir, f"{split}_{name}.bin")
        got = 0
        with open(path, "wb") as f:
            for ids in gen:
                arr = np.asarray(ids + [eos], dtype=np.uint32)
                f.write(arr.tobytes())
                got += len(arr)
                if got >= budget:
                    break
        counts[split] = got
        print(f"  [{name}] {split}: {got/1e6:.2f}M tokens -> {path}", flush=True)
    print(f"  [{name}] done in {time.perf_counter()-t0:.0f}s", flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-tokens", type=int, required=True)
    ap.add_argument("--val-tokens", type=int, default=300_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    eos = tokenizer.eos_token_id
    assert eos is not None

    manifest = {"tokenizer": MODEL, "eos": eos, "dtype": "uint32",
                "train_tokens_target": args.train_tokens, "domains": {}}
    for name, share, dataset, load_kw, field in SOURCES:
        budget = int(args.train_tokens * share)
        print(f"[{name}] share={share} target={budget/1e6:.1f}M from {dataset}", flush=True)
        counts = build_domain(tokenizer, eos, name, dataset, load_kw, field,
                              budget, args.val_tokens, args.out)
        manifest["domains"][name] = {"share": share, "dataset": dataset, **counts}

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("manifest:", json.dumps(manifest["domains"]))


if __name__ == "__main__":
    main()
