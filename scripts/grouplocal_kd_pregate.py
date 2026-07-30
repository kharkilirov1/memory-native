"""Row-vs-group stats_scope KD pre-gate on real donor blocks (CPU, ~1-2 h).

The group-local update (results/GROUP_LOCAL_UPDATE.md) changes the optimizer statistics
(RMS denom + clip over the 128-group instead of the row). Before spending a GPU session on
the full KD-parity gate, this witness answers the cheap version of the question on a REAL
donor: from an IDENTICAL PTQ start, do a few hundred KD steps under row scope vs group
scope land in the same place, or does one scope lose?

Protocol (mirrors the CPU network witnesses of the solver campaign):
  * Qwen2.5-0.5B fp32, truncated to the first NUM_BLOCKS transformer blocks (+ final norm);
  * teacher = the fp truncation (frozen deepcopy); student = same truncation, body linears
    PTQ-converted to `counter_packed` (cheap classic config -- both arms share the start,
    solver fanciness is irrelevant for a relative gate);
  * the group arm is a buffer-exact clone of the row student with stats_scope="group"
    (state/scale/perm/salient copied; v starts zero in BOTH scopes after conversion);
  * KD: MSE between student and teacher last_hidden_state on a WikiText-2 train stream,
    identical data order and seeds in both arms; fp params FROZEN so the only difference
    is the counter update rule; counter lr constant-low (recipe: from a good PTQ start,
    start LOW), clip=1.0;
  * metric: held-out (validation) hidden-state MSE at the warm start and during training.

Single-domain data is fine HERE: this is a relative optimizer gate from a shared start,
not a recovery-quality claim (the mixed-corpus rule applies to full recovery runs).

Run: PYTHONPATH=src python scripts/grouplocal_kd_pregate.py
Env: MODEL, NUM_BLOCKS, STEPS, BATCH, SEQ, CALIB_BATCHES, CALIB_SEQ, LR, EVAL_EVERY.
"""
from __future__ import annotations

import copy
import os
import time

import torch
import torch.nn as nn

from memory_native.donor.ptq import ptq_warm_start
from memory_native.group_scale_packed import PackedGroupScaleCounterLinear

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", "2"))
STEPS = int(os.environ.get("STEPS", "200"))
BATCH = int(os.environ.get("BATCH", "2"))
SEQ = int(os.environ.get("SEQ", "512"))
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "16"))
CALIB_SEQ = int(os.environ.get("CALIB_SEQ", "512"))
LR = float(os.environ.get("LR", "0.002"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "25"))
EVAL_WINDOWS = int(os.environ.get("EVAL_WINDOWS", "8"))


def load_truncated(model_name: str, n_blocks: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.model.layers = nn.ModuleList(list(model.model.layers)[:n_blocks])
    model.config.num_hidden_layers = n_blocks
    model.eval()
    return tok, model


def wikitext_ids(tok, split: str) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    return tok(text, return_tensors="pt").input_ids[0]


def windows(ids: torch.Tensor, offset: int, n: int, seq: int, batch: int = 1):
    out = []
    pos = offset
    for _ in range(n):
        chunk = torch.stack([ids[pos + b * seq: pos + (b + 1) * seq] for b in range(batch)])
        out.append(chunk)
        pos += batch * seq
    return out


def to_group_scope(model: nn.Module) -> nn.Module:
    m2 = copy.deepcopy(model)
    swapped = 0
    for _, parent in m2.named_modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, PackedGroupScaleCounterLinear):
                nl = PackedGroupScaleCounterLinear(
                    child.in_features, child.out_features, group=child.group, C=child.C,
                    lr=child.lr, lr_scale=child.lr_scale, rms_beta=child.rms_beta,
                    rms_eps=child.rms_eps, local_grad_clip=child.local_grad_clip,
                    residual_alpha=child.residual_alpha, kernel_mode=child.kernel_mode,
                    strict_update=child.strict_update, stats_scope="group",
                )
                nl.state.copy_(child.state)
                nl.scale.copy_(child.scale)
                nl.perm.copy_(child.perm)
                nl.salient_idx = child.salient_idx.clone()
                nl.salient_val = child.salient_val.clone()
                nl._rebuild_salient_runtime()
                setattr(parent, name, nl)
                swapped += 1
    assert swapped, "no packed counter layers found to re-scope"
    return m2


@torch.no_grad()
def eval_mse(student: nn.Module, teacher: nn.Module, eval_batches) -> float:
    student.eval()
    total, count = 0.0, 0
    for ids in eval_batches:
        h_t = teacher.model(ids).last_hidden_state
        h_s = student.model(ids).last_hidden_state
        total += ((h_s - h_t) ** 2).mean().item() * ids.numel()
        count += ids.numel()
    return total / count


def kd_arm(name: str, student: nn.Module, teacher: nn.Module, train_stream,
           eval_batches) -> list[tuple[int, float]]:
    curve = [(0, eval_mse(student, teacher, eval_batches))]
    print(f"[{name}] warm eval mse={curve[0][1]:.6f}", flush=True)
    t0 = time.time()
    for step, ids in enumerate(train_stream, start=1):
        student.train()
        with torch.no_grad():
            target = teacher.model(ids).last_hidden_state
        out = student.model(ids).last_hidden_state
        loss = ((out - target) ** 2).mean()
        loss.backward()
        if step % EVAL_EVERY == 0 or step == len(train_stream):
            m = eval_mse(student, teacher, eval_batches)
            curve.append((step, m))
            print(f"[{name}] step {step:4d} train={loss.item():.6f} "
                  f"eval={m:.6f} ({(time.time() - t0) / step:.1f}s/step)", flush=True)
    return curve


def main() -> None:
    torch.manual_seed(0)
    tok, fp = load_truncated(MODEL, NUM_BLOCKS)
    teacher = copy.deepcopy(fp)
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_ids = wikitext_ids(tok, "train")
    val_ids = wikitext_ids(tok, "validation")
    calib = windows(train_ids, 0, CALIB_BATCHES, CALIB_SEQ)
    kd_stream = windows(train_ids, CALIB_BATCHES * CALIB_SEQ + 1000, STEPS, SEQ, BATCH)
    eval_batches = windows(val_ids, 0, EVAL_WINDOWS, SEQ)
    print(f"model={MODEL} blocks={NUM_BLOCKS} steps={STEPS} batch={BATCH}x{SEQ} "
          f"calib={CALIB_BATCHES}x{CALIB_SEQ} lr={LR}")

    student_row = fp  # converted in place below
    report = ptq_warm_start(
        student_row, calib, mode="gptq_group", kind="counter_packed",
        group=128, refine_iters=1, scale_refit="align", act_order=True,
        lr=LR, local_grad_clip=1.0, stats_scope="row", progress=True,
    )
    print(f"converted: {report}")
    for p in student_row.parameters():
        p.requires_grad_(False)
    scopes = {l.stats_scope for l in student_row.modules()
              if isinstance(l, PackedGroupScaleCounterLinear)}
    assert scopes == {"row"}, scopes

    student_group = to_group_scope(student_row)
    scopes = {l.stats_scope for l in student_group.modules()
              if isinstance(l, PackedGroupScaleCounterLinear)}
    assert scopes == {"group"}, scopes

    curve_row = kd_arm("row", student_row, teacher, kd_stream, eval_batches)
    curve_group = kd_arm("group", student_group, teacher, kd_stream, eval_batches)

    print("\nstep  row_eval  group_eval")
    gd = dict(curve_group)
    for step, m in curve_row:
        print(f"{step:5d}  {m:.6f}  {gd.get(step, float('nan')):.6f}")
    warm = curve_row[0][1]
    fr, fg = curve_row[-1][1], curve_group[-1][1]
    print(f"\nwarm={warm:.6f}  final row={fr:.6f} ({fr / warm:.3f}x warm)  "
          f"final group={fg:.6f} ({fg / warm:.3f}x warm)  group/row={fg / fr:.3f}")


if __name__ == "__main__":
    main()
