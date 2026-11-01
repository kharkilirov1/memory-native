#!/usr/bin/env python3
"""CLI for GRPO training of a HF LM on desktop RL tasks, with mx6 latent-grid or AdamW."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# MX6 / repo path setup
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
_MX6 = _REPO.parent / "research" / "mxfp4-native-no-state" / "mxfp4-native"
if str(_MX6) not in sys.path:
    sys.path.insert(0, str(_MX6))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deskenv.ad import AgentDesktop, FakeAD, _ScriptedStep
from deskenv.env import DesktopEnv
from deskenv.session import DesktopSession
from deskenv.tasks import make_task
from rl.convert import convert_to_lg, LGLinearBias
from rl.grpo import grpo_step, compute_logprobs
from rl.policy import LMPolicy


def _build_eval_set(seed: int, kinds: list[str], n_per_kind: int = 8):
    """Build a fixed set of task seeds for evaluation."""
    rng = random.Random(seed)
    tasks = []
    for kind in kinds:
        for _ in range(n_per_kind):
            tasks.append((kind, rng.randint(0, 2 ** 31)))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO desktop RL training")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--optim", choices=("lg", "adamw"), default="lg")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--n-micro", type=int, default=4)
    ap.add_argument("--kinds", nargs="*", default=["write_save", "uppercase_save"])
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--ad-bin", default="agent-desktop")
    ap.add_argument("--display", type=int, default=99)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--dry-env", action="store_true",
                    help="Use FakeAD-based scripted env for testing on Windows")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── device ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── model ───────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.train()

    # ── convert to lg or prepare AdamW ──────────────────────────────
    adamw_params: list[torch.nn.Parameter] = []
    if args.optim == "lg":
        report = convert_to_lg(
            model, k=6, headroom=2.0, lr=args.lr,
            momentum=0.9, mom_dtype="fp8", n_micro=args.n_micro,
            skip=("lm_head", "embed"),
            backend="torch" if device == "cpu" else "auto",
        )
        print(f"Converted {report['n_layers']} layers to LG")
        for l in report["report_lines"][:10]:
            print(l)
        # Collect AdamW params (LGLinearBias biases, norms, etc.)
        for name, mod in model.named_parameters():
            if any(s in name for s in ("bias", "norm", "layernorm", "ln_")):
                if mod.requires_grad:
                    adamw_params.append(mod)
        print(f"AdamW params (bias/norm): {len(adamw_params)}")
    else:
        print("Using plain AdamW (all params)")

    # ── env runner ──────────────────────────────────────────────────
    session = None
    ad: AgentDesktop | FakeAD
    if args.dry_env:
        ad = FakeAD()
        print("Dry-env mode: using FakeAD")
    else:
        session = DesktopSession(display=args.display, ad_bin=args.ad_bin)
        session.start()
        ad = session.ad

    def run_env(seed: int, policy) -> list[dict]:
        """Run episodes for one task seed."""
        rng = random.Random(seed)
        kind = random.Random(seed).choice(args.kinds) if args.dry_env else args.kinds[seed % len(args.kinds)]
        from deskenv.runner import run_episode
        # We can't easily use run_episode here because we need per-turn logprobs.
        # Simplified: one dummy episode
        workdir = session.new_workdir() if session else Path("C:\\tmp")
        task = make_task(kind, rng, workdir)
        task.setup(workdir)
        proc = None
        if session:
            proc = session.launch(task.launch_cmd, wait_app=task.app, workdir=workdir)
        env = DesktopEnv(ad, task, workdir=workdir, max_steps=args.max_turns)
        obs = env.reset()
        turns = []
        done = False
        while not done:
            action, pids, cids = policy(obs)
            obs, reward, done, info = env.step(action)
            turns.append({"action_text": action, "prompt_ids": pids, "completion_ids": cids})
        if proc:
            session.kill(proc)
        return [{"reward": reward, "turns": turns, "info": info}]

    policy = LMPolicy(model, tokenizer, temperature=1.0, device=device)

    # ── training loop ───────────────────────────────────────────────
    log: list[dict] = []
    try:
        for step in range(args.steps):
            result = grpo_step(
                model, run_env,
                b=args.B, g=args.G,
                optim=args.optim, lr=args.lr,
                n_micro=args.n_micro,
                adamw_params=adamw_params if args.optim == "lg" else None,
                max_turns=args.max_turns,
                device=device,
            )
            log.append({"step": step, **result})
            print(f"step {step:3d}: mean_r={result['mean_reward']:.4f} "
                  f"max_r={result['max_reward']:.4f} succ={result['success_rate']:.3f} "
                  f"tok={result['n_tokens']} t={result['step_time_s']:.2f}s")

            if args.eval_every > 0 and step % args.eval_every == 0 and step > 0:
                print(f"  eval at step {step}...")

    finally:
        if session:
            session.stop()

    # ── output ──────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Log saved to {out_path}")


if __name__ == "__main__":
    main()
