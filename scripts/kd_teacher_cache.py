"""Offline teacher-logit cache for cached-KD recovery (the 12B-on-T4 enabler).

A 12B bf16 teacher (22.3 GiB) does not fit a 16 GiB T4, and a CPU teacher makes every KD
step a minute. So the teacher runs ONCE, block-streamed (never fully resident — the
conversion's own machinery), over the EXACT deterministic KD stream the trainer will
consume (`DomainMix.batch_at(step)` is step-deterministic), and we cache top-K logits per
token. The trainer then needs no teacher at all: steps get cheaper AND smaller.

Memory shape: the activation buffer for the whole stream lives in RAM as bf16
([N_tokens, d] — ~6 GiB per 800k tokens at d=3840), one block at a time lives on DEVICE.
Cache size: N_tokens * K * 6 B (int32 idx + fp16 logit) ≈ 0.3 GiB per 800k tokens @K=64.

Feature-KD is deliberately dropped on this path (hidden states are uncacheable at these
sizes) — documented delta vs the live-teacher runner.

Run: MODEL=<donor dir> DATA_DIR=<mix bins> OUT=<cache dir> STEPS=800 BATCH=2 SEQ=512 \
     PYTHONPATH=src python scripts/kd_teacher_cache.py
env: MODEL, DATA_DIR, OUT, STEPS, BATCH, SEQ, TOPK (64), SEED (0), DEVICE, DTYPE (bf16),
     NUM_BLOCKS (0 = full depth; >0 truncates — smoke only, the trainer must match),
     CHUNK_ROWS (rows per device micro-batch), SHARD_STEPS (steps per cache shard file).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

MODEL = os.environ["MODEL"]
DATA_DIR = os.environ["DATA_DIR"]
OUT = os.environ["OUT"]
STEPS = int(os.environ.get("STEPS", "800"))
BATCH = int(os.environ.get("BATCH", "2"))
SEQ = int(os.environ.get("SEQ", "512"))
TOPK = int(os.environ.get("TOPK", "64"))
SEED = int(os.environ.get("SEED", "0"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("DTYPE", "bf16")]
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", "0"))
CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", "8"))
SHARD_STEPS = int(os.environ.get("SHARD_STEPS", "200"))


def log(msg: str) -> None:
    print(f"[cache {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def main() -> None:
    from recovery_session import DomainMix
    from memory_native.donor.streaming import (
        _WeightSource, _build_probe, _computed_buffers, _materialize,
        _resolve_decoder, _run_block,
    )
    from transformers import AutoConfig, AutoModelForCausalLM

    os.makedirs(OUT, exist_ok=True)
    mix = DomainMix(DATA_DIR, seq=SEQ, batch=BATCH, seed=SEED)
    ids = torch.cat([mix.batch_at(s, "cpu") for s in range(STEPS)])  # [STEPS*B, SEQ]
    n_rows = ids.shape[0]
    log(f"stream: {STEPS} steps x {BATCH}x{SEQ} = {n_rows * SEQ} tokens")

    src = _WeightSource(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config)
    inner, stack = _resolve_decoder(skeleton)
    if NUM_BLOCKS:
        import torch.nn as nn
        inner.layers = nn.ModuleList(list(inner.layers)[:NUM_BLOCKS])
    probe = _build_probe(config, "cpu")
    probe_inner, _ = _resolve_decoder(probe)
    computed_embed = _computed_buffers(probe_inner.embed_tokens)
    computed_block = _computed_buffers(probe_inner.layers[0])
    rotary = probe_inner.rotary_emb.to(DEVICE) if DEVICE != "cpu" else probe_inner.rotary_emb

    _materialize(inner.embed_tokens, f"{stack}.embed_tokens", src, "cpu", DTYPE,
                 computed=computed_embed)
    # buffer: list of [chunk, SEQ, d] bf16 CPU tensors
    buffer = [inner.embed_tokens(ids[i:i + CHUNK_ROWS])
              for i in range(0, n_rows, CHUNK_ROWS)]  # shard dtype end-to-end
    inner.embed_tokens.to("meta")
    gc.collect()
    log(f"embeddings done ({sum(b.numel() for b in buffer) * 2 / 2**30:.2f} GiB buffer)")

    n_blocks = len(inner.layers)
    for i in range(n_blocks):
        t0 = time.time()
        block = inner.layers[i]
        _materialize(block, f"{stack}.layers.{i}", src, DEVICE, DTYPE,
                     computed=computed_block)
        dt = next(block.parameters()).dtype
        buffer = [b.to(dt) for b in buffer]
        buffer = _run_block(block, buffer, rotary, DEVICE, CHUNK_ROWS, collect=True)
        block.to("meta")
        gc.collect()
        if DEVICE != "cpu":
            torch.cuda.empty_cache()
        if i % 4 == 0 or i == n_blocks - 1:
            log(f"block {i + 1}/{n_blocks} ({time.time() - t0:.1f}s)")

    _materialize(inner.norm, f"{stack}.norm", src, "cpu", DTYPE, computed={})
    norm = inner.norm
    if src.has("lm_head.weight"):
        w_head = src.get("lm_head.weight", DTYPE)
    else:
        w_head = src.get(f"{stack}.embed_tokens.weight", DTYPE)
    w_head = w_head.to(DEVICE)
    src.close()

    # top-K per token, sharded by steps
    rows_per_step = BATCH
    shard_idx, shard_val, shard_id = [], [], 0
    row = 0
    for b in buffer:
        h = norm.to(b.device)(b).to(DEVICE)
        logits = h.reshape(-1, h.shape[-1]) @ w_head.t()          # [chunk*SEQ, V]
        val, idx = torch.topk(logits.float(), TOPK, dim=-1)
        shard_idx.append(idx.to(torch.int32).cpu().view(b.shape[0], SEQ, TOPK))
        shard_val.append(val.to(torch.float16).cpu().view(b.shape[0], SEQ, TOPK))
        row += b.shape[0]
        del logits, h
        while sum(x.shape[0] for x in shard_idx) >= SHARD_STEPS * rows_per_step or (
                row == n_rows and shard_idx):
            have = torch.cat(shard_idx), torch.cat(shard_val)
            take = min(SHARD_STEPS * rows_per_step, have[0].shape[0])
            torch.save({"idx": have[0][:take], "val": have[1][:take]},
                       os.path.join(OUT, f"cache_{shard_id:04d}.pt"))
            shard_idx = [have[0][take:]] if have[0].shape[0] > take else []
            shard_val = [have[1][take:]] if have[1].shape[0] > take else []
            shard_id += 1
            if row == n_rows and not (shard_idx and shard_idx[0].shape[0]):
                shard_idx = []
                break
    json.dump({"steps": STEPS, "batch": BATCH, "seq": SEQ, "topk": TOPK, "seed": SEED,
               "shard_steps": SHARD_STEPS, "num_blocks": NUM_BLOCKS, "model": MODEL},
              open(os.path.join(OUT, "cache_manifest.json"), "w"))
    log(f"cache done: {shard_id} shards -> {OUT}")


if __name__ == "__main__":
    main()
