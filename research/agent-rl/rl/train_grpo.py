"""GRPO on the desktop tasks, with the mx6 latent grid (optimizer-free, update inside backward) or an AdamW baseline.

    python rl/train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --optim lg --steps 50 --B 2 --G 4 \
        --kinds write_save append_line --ad-bin /path/to/agent-desktop --out runs/lg1

One step: B task seeds x G episodes of the SAME task (same seed -> same instruction and files); reward = env reward at
episode end; advantage = R - mean_group(R) (Dr.GRPO, no std); groups with equal rewards are dropped; loss =
-sum_turns A * sum logp(completion | prompt) / total completion tokens; ONE update per step (on-policy: no ratio, no
clip, no reference model).

mx6 arm: every decoder nn.Linear -> LatentGridLinear (rl/convert.py), momentum 0.9 fp8 accumulated over EXACTLY
--n-micro micro-batches (mx6/lg_accum.set_momentum(n=...) + lg_accumulate). Each micro-batch is ONE padded forward and
ONE backward, so every lg layer sees exactly n backwards (empty micro-batches run a zero-loss 1-token pass).
--moe: fused experts become per-expert lg layers (rl/convert_moe.py) and the momentum is sparse (a layer may get 0..n
backwards, ONE update at the exit of lg_accumulate; empty micro-batches are skipped instead of the zero-loss pass).
Embeddings and lm_head are frozen; biases (Qwen q/k/v) and norms train with a persistent AdamW.

Verified 2026-11-01 on a real Xvfb desktop (CPU, Qwen2.5-0.5B-Instruct): rollouts + update run end to end
(~25 s per episode); the 0.5B model gets reward 0 on write_save, so no group had a learning signal.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0] / "mxfp4-native-no-state" / "mxfp4-native"))

from deskenv.env import DesktopEnv          # noqa: E402
from deskenv.session import DesktopSession  # noqa: E402
from deskenv.tasks import make_task         # noqa: E402
from rl.convert import convert_to_lg        # noqa: E402
from rl.convert_moe import convert_moe_experts_to_lg, moe_skip  # noqa: E402
from rl.policy import LMPolicy              # noqa: E402


def run_group(session, policy, kind: str, seed: int, g: int, max_turns: int) -> list[dict]:
    """G episodes of one task. Each turn keeps (prompt_ids, completion_ids) for the update."""
    episodes = []
    for _ in range(g):
        workdir = session.new_workdir()
        task = make_task(kind, random.Random(seed), workdir)
        task.setup(workdir)
        proc = session.launch(task.launch_cmd, wait_app=task.app, workdir=workdir)
        try:
            env = DesktopEnv(session.ad, task, workdir=workdir, max_steps=max_turns)
            obs = env.reset()
            turns, history, done, reward, info = [], [], False, 0.0, {}
            while not done:
                text, pids, cids = policy.act(obs, history)
                obs, reward, done, info = env.step(text)
                last = obs.rsplit("Last action:", 1)[-1].strip()[:120] if "Last action:" in obs else ""
                history.append((text.splitlines()[-1][:160] if text else "", last))
                turns.append({"prompt_ids": pids.cpu(), "completion_ids": cids.cpu(), "text": text})
            episodes.append({"reward": float(reward), "turns": turns, "info": info, "kind": kind, "seed": seed})
        finally:
            session.kill(proc)
    return episodes


def pack(turns: list[dict], n: int) -> list[list[dict]]:
    """EXACTLY n micro-batches, greedy-balanced by total length (some may be empty)."""
    out, load = [[] for _ in range(n)], [0] * n
    for t in sorted(turns, key=lambda t: -(len(t["prompt_ids"]) + len(t["completion_ids"]))):
        i = min(range(n), key=lambda j: load[j])
        out[i].append(t)
        load[i] += len(t["prompt_ids"]) + len(t["completion_ids"])
    return out


def micro_loss(model, batch: list[dict], pad_id: int, n_tokens_total: int, device) -> torch.Tensor:
    """ONE padded forward over the micro-batch; -sum A * logp(completion) / n_tokens_total (grad-carrying)."""
    if not batch:                                     # zero-loss pass: every lg layer still gets its one backward
        x = torch.full((1, 1), pad_id, dtype=torch.long, device=device)
        return model(input_ids=x).logits.sum() * 0.0
    seqs = [torch.cat([t["prompt_ids"], t["completion_ids"]]) for t in batch]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    tgt_mask = torch.zeros((len(seqs), L), dtype=torch.float32)
    adv = torch.tensor([t["advantage"] for t in batch], dtype=torch.float32)
    for i, (s, t) in enumerate(zip(seqs, batch)):
        ids[i, :len(s)] = s
        att[i, :len(s)] = 1
        tgt_mask[i, len(t["prompt_ids"]):len(s)] = 1.0          # completion tokens only
    ids, att, tgt_mask, adv = ids.to(device), att.to(device), tgt_mask.to(device), adv.to(device)
    # lm_head ONLY on completion positions: full-sequence fp32 logits at a 262k vocab (Gemma 4) are tens of GB
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    m = tgt_mask[:, 1:]
    sel = m.bool()
    hs, tg = h[sel], ids[:, 1:][sel]
    w_adv = adv[:, None].expand_as(m)[sel]
    head = model.get_output_embeddings()
    cfg = getattr(model.config, "text_config", model.config)
    cap = getattr(cfg, "final_logit_softcapping", None)
    logp_parts = []
    for j in range(0, hs.shape[0], 512):
        lg = head(hs[j:j + 512]).float()
        if cap:
            lg = torch.tanh(lg / cap) * cap
        logp_parts.append(lg.gather(-1, tg[j:j + 512, None]).squeeze(-1) - torch.logsumexp(lg, -1))
    logp = torch.cat(logp_parts) if logp_parts else hs.sum(-1) * 0.0
    return -(w_adv * logp).sum() / max(n_tokens_total, 1)


def update(model, turns, n_micro, optim_kind, adamw, pad_id, device) -> float:
    n_tok = sum(len(t["completion_ids"]) for t in turns)
    model.train()
    total = 0.0
    batches = pack(turns, n_micro)
    if optim_kind == "lg":
        from mx6.lg_accum import get_accum, lg_accumulate
        sparse = bool(getattr(get_accum(model), "sparse", False))
        with lg_accumulate(model, n_micro) as acc:
            for i in acc.micro_batches():
                if sparse and not batches[i]:         # sparse (MoE): a layer may skip micro-batches; no zero-loss pass
                    continue
                loss = micro_loss(model, batches[i], pad_id, n_tok, device)
                loss.backward()                       # the lg update happens here (on the last micro-batch)
                total += float(loss.detach())
    else:
        for b in batches:
            loss = micro_loss(model, b, pad_id, n_tok, device)
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([p for g in adamw.param_groups for p in g["params"]], 1.0)
    if adamw is not None:
        adamw.step()
        adamw.zero_grad(set_to_none=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--optim", choices=("lg", "adamw"), default="lg")
    ap.add_argument("--lr", type=float, default=3e-3, help="lg: counter lr; adamw: lr of all trainable params")
    ap.add_argument("--lr-small", type=float, default=1e-4, help="lg arm: AdamW lr of biases/norms")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--moe", action="store_true",
                    help="MoE model: fused experts -> per-expert lg layers (rl/convert_moe.py) + sparse momentum")
    ap.add_argument("--n-micro", type=int, default=4)
    ap.add_argument("--kinds", nargs="*", default=["write_save", "append_line"])
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ad-bin", default="agent-desktop")
    ap.add_argument("--display", type=int, default=99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/grpo")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", encoding="utf-8")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    if args.optim == "lg":
        backend = "torch" if device == "cpu" else "auto"
        if args.moe:
            mrep = convert_moe_experts_to_lg(model, lr=args.lr, backend=backend)
            print(f"moe: {mrep['n_modules']} expert modules, {mrep['n_experts']} experts -> {mrep['n_layers']} lg layers",
                  flush=True)
            rep = convert_to_lg(model, lr=args.lr, momentum=0.0, backend=backend, skip=moe_skip())
            rep["n_layers"] += mrep["n_layers"]
        else:
            rep = convert_to_lg(model, lr=args.lr, momentum=0.0, backend=backend)
        from mx6.lg_accum import set_momentum
        set_momentum(model, 0.9, "fp8", n=args.n_micro, sparse=args.moe)
        small = [p for n_, p in model.named_parameters() if n_.endswith("bias") or "norm" in n_]
        for p in small:
            p.requires_grad_(True)
        adamw = torch.optim.AdamW(small, lr=args.lr_small, weight_decay=0.0) if small else None
        print(f"lg: {rep['n_layers']} layers converted, {len(small)} small AdamW tensors", flush=True)
    else:
        train = [p for n_, p in model.named_parameters() if "embed" not in n_ and "lm_head" not in n_]
        for p in train:
            p.requires_grad_(True)
        adamw = torch.optim.AdamW(train, lr=args.lr, weight_decay=0.0)
        print(f"adamw: {sum(p.numel() for p in train) / 1e6:.1f}M trainable params", flush=True)

    policy = LMPolicy(model, tok, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
    session = DesktopSession(display=args.display, ad_bin=args.ad_bin)
    session.start()
    try:
        for step in range(args.steps):
            t0 = time.time()
            eps = []
            for b in range(args.B):
                # round-robin over kinds so every kind shows up (random choice repeated one kind)
                kind, seed = args.kinds[(step * args.B + b) % len(args.kinds)], rng.randrange(10 ** 6)
                eps += run_group(session, policy, kind, seed, args.G, args.max_turns)
            t_roll = time.time() - t0
            turns = []
            for i in range(0, len(eps), args.G):
                grp = eps[i:i + args.G]
                rs = [e["reward"] for e in grp]
                if max(rs) - min(rs) < 1e-9:
                    continue
                mean = sum(rs) / len(rs)
                for e in grp:
                    for t in e["turns"]:
                        t["advantage"] = e["reward"] - mean
                        turns.append(t)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"  rollout done: {len(turns)} update turns, "
                      f"{sum(len(t['prompt_ids']) + len(t['completion_ids']) for t in turns)} tokens, "
                      f"gpu {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB", flush=True)
            loss = update(model, turns, args.n_micro, args.optim, adamw, pad_id, device) if turns else 0.0
            rs = [e["reward"] for e in eps]
            rec = {"step": step, "mean_reward": sum(rs) / len(rs), "success": sum(r >= 0.99 for r in rs) / len(rs),
                   "mean_turns": sum(len(e["turns"]) for e in eps) / len(eps), "n_update_turns": len(turns),
                   "loss": loss, "t_rollout": round(t_roll, 1), "t_step": round(time.time() - t0, 1),
                   "kinds": [e["kind"] for e in eps[::args.G]], "rewards": rs,
                   "sample": eps[0]["turns"][0]["text"][-200:] if eps and eps[0]["turns"] else "",
                   # first episode of every group: its reward and the tail of every reply (why it failed)
                   "traces": [{"kind": e["kind"], "reward": e["reward"],
                               "acts": [t["text"][-160:] for t in e["turns"]]} for e in eps[::args.G]],
                   "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1) if torch.cuda.is_available() else 0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step}: R={rec['mean_reward']:.3f} succ={rec['success']:.2f} turns={rec['mean_turns']:.1f} "
                  f"upd_turns={len(turns)} loss={loss:.4f} t={rec['t_step']}s", flush=True)
    finally:
        session.stop()
        log.close()


if __name__ == "__main__":
    main()
