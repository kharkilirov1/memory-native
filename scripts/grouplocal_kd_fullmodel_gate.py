"""FULL-MODEL KD gate: row vs group vs group+dec4 on Qwen2.5-0.5B (GPU, ~30-40 min).

The 2-block CPU pre-gate (results/GROUPLOCAL_KD_PREGATE.md) showed group scope winning
-29.7% vs -15.9% held-out MSE. This is the model-scale follow-up on GPU: all 24 blocks
converted (classic PTQ start, shared by every arm), KD against the fp teacher, and the
headline metric is strict WikiText-2 PPL of each arm after the same step budget.

Arms (identical data order, identical start, fp slice frozen):
  row        -- production update path (dense kernels on CUDA), lr=LR
  group      -- group-local statistics (fused one-launch kernel on CUDA), lr=LR
  group+dec4 -- decimation=4 round-robin, lr=LR*4 (the measured recipe: same integrated
                signal, 1/4 update FLOPs, results/DECIMATION_WITNESS.md)

Run: PYTHONPATH=src python scripts/grouplocal_kd_fullmodel_gate.py
env: MODEL, STEPS, BATCH, SEQ, CALIB_BATCHES, CALIB_SEQ, LR, EVAL_EVERY, EVAL_WINDOWS,
     PPL_WINDOWS, DEVICE, ARMS (comma list among row,group,dec4).
"""
from __future__ import annotations

import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native.donor.ptq import ptq_warm_start
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
STEPS = int(os.environ.get("STEPS", "300"))
BATCH = int(os.environ.get("BATCH", "2"))
SEQ = int(os.environ.get("SEQ", "512"))
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "16"))
CALIB_SEQ = int(os.environ.get("CALIB_SEQ", "512"))
LR = float(os.environ.get("LR", "0.002"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "50"))
EVAL_WINDOWS = int(os.environ.get("EVAL_WINDOWS", "8"))
PPL_WINDOWS = int(os.environ.get("PPL_WINDOWS", "16"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ARMS = os.environ.get("ARMS", "row,group,dec4").split(",")
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", "0"))  # 0 = full model; >0 = smoke truncation


def wikitext_ids(tok, split):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]


def windows(ids, offset, n, seq, batch=1):
    out, pos = [], offset
    for _ in range(n):
        out.append(torch.stack([ids[pos + b * seq: pos + (b + 1) * seq]
                                for b in range(batch)]))
        pos += batch * seq
    return out


def counter_layers(m):
    return [l for l in m.modules() if isinstance(l, PackedGroupScaleCounterLinear)]


def clone_arm(src_model: nn.Module, stats_scope: str, decimation: int, lr: float):
    m2 = copy.deepcopy(src_model)
    for _, parent in m2.named_modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, PackedGroupScaleCounterLinear):
                nl = PackedGroupScaleCounterLinear(
                    child.in_features, child.out_features, group=child.group, C=child.C,
                    lr=lr, lr_scale=child.lr_scale, rms_beta=child.rms_beta,
                    rms_eps=child.rms_eps, local_grad_clip=child.local_grad_clip,
                    residual_alpha=child.residual_alpha, kernel_mode=child.kernel_mode,
                    strict_update=child.strict_update, stats_scope=stats_scope,
                    decimation=decimation,
                ).to(child.state.device)
                nl.state.copy_(child.state)
                nl.scale.copy_(child.scale)
                nl.perm.copy_(child.perm)
                nl.salient_idx = child.salient_idx.clone()
                nl.salient_val = child.salient_val.clone()
                nl._rebuild_salient_runtime()
                setattr(parent, name, nl)
    return m2


@torch.no_grad()
def hidden_mse(student, teacher, eval_batches):
    student.eval()
    total, count = 0.0, 0
    for ids in eval_batches:
        h_t = teacher.model(ids).last_hidden_state
        h_s = student.model(ids).last_hidden_state
        total += ((h_s - h_t) ** 2).mean().item() * ids.numel()
        count += ids.numel()
    return total / count


@torch.no_grad()
def wikitext_ppl(model, val_ids, n_windows, seq=1024):
    model.eval()
    total_nll, total_tok = 0.0, 0
    for w in range(n_windows):
        window = val_ids[w * seq:(w + 1) * seq].unsqueeze(0).to(DEVICE)
        logits = model(window).logits[0].float()
        logp = F.log_softmax(logits[:-1], dim=-1)
        total_nll += -logp.gather(1, window[0, 1:].unsqueeze(1)).sum().item()
        total_tok += seq - 1
    return float(torch.tensor(total_nll / total_tok).exp())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    fp = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if NUM_BLOCKS:
        fp.model.layers = nn.ModuleList(list(fp.model.layers)[:NUM_BLOCKS])
        fp.config.num_hidden_layers = NUM_BLOCKS
    fp = fp.to(DEVICE)
    fp.eval()
    teacher = copy.deepcopy(fp)
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_ids = wikitext_ids(tok, "train")
    val_ids = wikitext_ids(tok, "validation")
    calib = [w.to(DEVICE) for w in windows(train_ids, 0, CALIB_BATCHES, CALIB_SEQ)]
    kd_stream = [w.to(DEVICE) for w in windows(
        train_ids, CALIB_BATCHES * CALIB_SEQ + 1000, STEPS, SEQ, BATCH)]
    eval_batches = [w.to(DEVICE) for w in windows(val_ids, 0, EVAL_WINDOWS, SEQ)]
    print(f"model={MODEL} dev={DEVICE} steps={STEPS} batch={BATCH}x{SEQ} lr={LR} "
          f"arms={ARMS}", flush=True)

    fp_ppl = wikitext_ppl(teacher, val_ids, PPL_WINDOWS)
    print(f"fp teacher wikitext ppl={fp_ppl:.3f} ({PPL_WINDOWS}x1024)", flush=True)

    base = fp  # converted in place: the shared PTQ start (row scope)
    t0 = time.time()
    report = ptq_warm_start(
        base, calib, mode="gptq_group", kind="counter_packed",
        group=128, refine_iters=1, scale_refit="align", act_order=True,
        lr=LR, local_grad_clip=1.0, stats_scope="row", progress=False,
    )
    for p in base.parameters():
        p.requires_grad_(False)
    print(f"PTQ start: {report} in {(time.time() - t0) / 60:.1f} min", flush=True)
    warm_ppl = wikitext_ppl(base, val_ids, PPL_WINDOWS)
    warm_mse = hidden_mse(base, teacher, eval_batches)
    print(f"warm: ppl={warm_ppl:.3f} hidden_mse={warm_mse:.6f}", flush=True)

    arm_defs = {
        "row": dict(stats_scope="row", decimation=1, lr=LR),
        "group": dict(stats_scope="group", decimation=1, lr=LR),
        "dec4": dict(stats_scope="group", decimation=4, lr=LR * 4),
    }
    results = {}
    for arm in ARMS:
        # "name" uses the default lr of arm_defs; "name:0.001" overrides it — the
        # group denominator is finer than row's, so the effective step differs at
        # equal lr and a single-lr comparison is apples-to-oranges.
        arm = arm.strip()
        if ":" in arm:
            name, lr_override = arm.split(":")
            cfg = dict(arm_defs[name], lr=float(lr_override))
        else:
            cfg = arm_defs[arm]
        student = clone_arm(base, cfg["stats_scope"], cfg["decimation"], cfg["lr"])
        scopes = {l.stats_scope for l in counter_layers(student)}
        assert scopes == {cfg["stats_scope"]}, scopes
        t0 = time.time()
        for step, ids in enumerate(kd_stream, start=1):
            student.train()
            with torch.no_grad():
                target = teacher.model(ids).last_hidden_state
            loss = ((student.model(ids).last_hidden_state - target) ** 2).mean()
            loss.backward()
            if step % EVAL_EVERY == 0 or step == STEPS:
                m = hidden_mse(student, teacher, eval_batches)
                print(f"[{arm}] step {step:4d} train={loss.item():.6f} eval={m:.6f} "
                      f"({(time.time() - t0) / step:.2f}s/step)", flush=True)
        ppl = wikitext_ppl(student, val_ids, PPL_WINDOWS)
        mse = hidden_mse(student, teacher, eval_batches)
        results[arm] = (ppl, mse)
        print(f"[{arm}] FINAL ppl={ppl:.3f} hidden_mse={mse:.6f}", flush=True)
        del student
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nfp ppl={fp_ppl:.3f}  warm ppl={warm_ppl:.3f} mse={warm_mse:.6f}")
    for arm, (ppl, mse) in results.items():
        print(f"{arm:>6}: ppl={ppl:.3f} ({ppl / warm_ppl:.3f}x warm)  "
              f"hidden_mse={mse:.6f} ({mse / warm_mse:.3f}x warm)")


if __name__ == "__main__":
    main()
