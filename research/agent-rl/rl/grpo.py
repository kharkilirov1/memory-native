"""GRPO training step for the desktop RL environment."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

_MX6 = Path(__file__).resolve().parents[3] / "research" / "mxfp4-native-no-state" / "mxfp4-native"
if str(_MX6) not in sys.path:
    sys.path.insert(0, str(_MX6))

from mx6 import lg_accum
from mx6.latent_grid import lg_layers


Policy = Callable[[str, list[tuple[str, str]] | None],
                  tuple[str, torch.Tensor, torch.Tensor]]
"""policy(obs, history) -> (action_text, prompt_ids, completion_ids)."""

EnvRunner = Callable[[int, Policy], list[dict[str, Any]]]
"""run_episodes(seed, policy) -> list of episode dicts with keys:
   reward, turns (list of dicts with prompt_ids, completion_ids, action_text), info."""


def compute_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    prompt_len: int | None = None,
) -> torch.Tensor:
    """Compute log probabilities of all tokens in *input_ids*.

    Returns (B, L) logprobs tensor.
    """
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out[0]
    logprobs = logits.log_softmax(-1)
    # Gather logprob of each token
    ids = input_ids.unsqueeze(-1)
    token_lp = logprobs.gather(-1, ids).squeeze(-1)
    if prompt_len is not None:
        token_lp = token_lp[:, prompt_len:]
    return token_lp


def _mean_group_rewards(rewards: list[float]) -> float:
    return sum(rewards) / len(rewards) if rewards else 0.0


def grpo_step(
    model: nn.Module,
    run_env: EnvRunner,
    b: int = 2,
    g: int = 4,
    optim: str = "lg",
    lr: float = 1e-4,
    n_micro: int = 4,
    adamw_params: list[nn.Parameter] | None = None,
    max_turns: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    """One GRPO step: sample episodes, compute loss, update model.

    Parameters
    ----------
    model:
        The LM (with or without LatentGridLinear layers).
    run_env:
        Callable taking (seed, policy) and returning episode records.
    b:
        Number of task seeds (groups).
    g:
        Episodes per seed (group size).
    optim:
        ``"lg"`` (mx6 update inside backward, no AdamW) or ``"adamw"``.
    lr:
        Learning rate (for lg layers or AdamW).
    n_micro:
        Micro-batches for ``lg_accumulate``.
    adamw_params:
        List of FP parameters for the AdamW optimizer (biases, norms, etc.).
    max_turns:
        Max turns per episode.

    Returns
    -------
    dict with keys: mean_reward, max_reward, success_rate, mean_turns,
    malformed_rate, n_tokens, step_time_s, report_lines.
    """
    model.train()
    t0 = time.time()
    report: list[str] = []

    # ── sample episodes ────────────────────────────────────────────────
    all_episodes: list[dict[str, Any]] = []
    import random as _random
    for seed in range(b):
        eps = run_env(seed, lambda obs, hist=None: "ACTION: {\"op\": \"done\"}")
        all_episodes.extend(eps)

    total_rewards = [ep["reward"] for ep in all_episodes]
    mean_r = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    max_r = max(total_rewards) if total_rewards else 0.0
    success = sum(1.0 for r in total_rewards if r >= 0.99)
    success_rate = success / len(total_rewards) if total_rewards else 0.0
    mean_turns = sum(len(ep.get("turns", [])) for ep in all_episodes) / len(all_episodes) if all_episodes else 0.0

    # ── compute advantages (Dr.GRPO) ───────────────────────────────────
    all_turns: list[dict] = []
    group_idx = 0
    for seed in range(b):
        group_eps = all_episodes[group_idx:group_idx + g]
        group_idx += g
        rewards = [ep["reward"] for ep in group_eps if ep.get("reward") is not None]
        if not rewards:
            continue
        # Skip groups with all-equal R
        if max(rewards) == min(rewards):
            continue
        mean_rg = _mean_group_rewards(rewards)
        for ep in group_eps:
            adv = ep["reward"] - mean_rg
            for turn in ep.get("turns", []):
                turn["advantage"] = adv
                all_turns.append(turn)

    if not all_turns:
        return {
            "mean_reward": mean_r, "max_reward": max_r,
            "success_rate": success_rate, "mean_turns": mean_turns,
            "malformed_rate": 0.0, "n_tokens": 0,
            "step_time_s": time.time() - t0, "report_lines": report,
        }

    total_tokens = sum(t["completion_ids"].shape[0] for t in all_turns if "completion_ids" in t)

    # ── compute logprobs (skip if pre-computed) ────────────────────────
    for turn in all_turns:
        if "logprobs" in turn and turn["logprobs"] is not None:
            continue
        ids = turn.get("completion_ids")
        if ids is not None and ids.numel() > 0:
            with torch.no_grad():
                lp = compute_logprobs(model, ids.unsqueeze(0).to(device),
                                      prompt_len=0)
            turn["logprobs"] = lp[0]

    # ── build per-turn loss ────────────────────────────────────────────
    losses: list[torch.Tensor] = []
    for turn in all_turns:
        lp = turn.get("logprobs")
        if lp is None or lp.numel() == 0:
            continue
        adv = torch.tensor(turn["advantage"], device=device)
        loss_turn = -adv * lp.sum()  # sum of logprobs
        losses.append(loss_turn)

    if not losses:
        return {
            "mean_reward": mean_r, "max_reward": max_r,
            "success_rate": success_rate, "mean_turns": mean_turns,
            "malformed_rate": 0.0, "n_tokens": 0,
            "step_time_s": time.time() - t0, "report_lines": report,
        }

    # Token-level normalisation
    n_total = sum(t.get("completion_ids", torch.zeros(0)).shape[0] for t in all_turns)
    loss_total = sum(losses) / max(n_total, 1)

    # ── backward ───────────────────────────────────────────────────────
    if not loss_total.requires_grad and loss_total.grad_fn is None:
        report.append("  (no grad — skipping backward)")
    elif optim == "lg":
        st = lg_accum.get_accum(model)
        if st is None or st.n <= 1:
            # No accumulation — just backward
            loss_total.backward()
            if adamw_params:
                opt = torch.optim.AdamW(adamw_params, lr=lr)
                opt.step()
                opt.zero_grad()
        else:
            # Pack into N micro-batches (balanced by token count)
            n = st.n
            micro_turns = _pack_into_micro(all_turns, n)
            with lg_accum.lg_accumulate(model, n) as acc:
                for i, batch in zip(acc.micro_batches(), micro_turns):
                    # Build micro-batch loss
                    micro_loss = torch.tensor(0.0, device=device)
                    for turn in batch:
                        lp = turn.get("logprobs")
                        if lp is None or lp.numel() == 0:
                            continue
                        adv = torch.tensor(turn["advantage"], device=device)
                        micro_loss = micro_loss + (-adv * lp.sum())
                    # Standard practice: divide by n_micro
                    (micro_loss / n).backward()
            # AdamW step for bias/norm params
            if adamw_params:
                opt = torch.optim.AdamW(adamw_params, lr=lr)
                opt.step()
                opt.zero_grad()
    else:
        # AdamW arm
        loss_total.backward()
        if adamw_params:
            nn.utils.clip_grad_norm_(adamw_params, 1.0)
            opt = torch.optim.AdamW(adamw_params, lr=lr)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
        opt.step()
        opt.zero_grad()

    report.append(f"  loss: {loss_total.item():.6f}, advantages: {[t['advantage'] for t in all_turns[:6]]}")

    return {
        "mean_reward": mean_r,
        "max_reward": max_r,
        "success_rate": success_rate,
        "mean_turns": mean_turns,
        "malformed_rate": 0.0,
        "n_tokens": total_tokens,
        "step_time_s": time.time() - t0,
        "report_lines": report,
    }


def _pack_into_micro(
    turns: list[dict], n: int
) -> list[list[dict]]:
    """Pack *turns* into n micro-batches, roughly balanced by token count.

    If fewer turns than n, pad with empty batches.
    """
    if len(turns) <= n:
        batches = [[t] for t in turns]
        while len(batches) < n:
            batches.append([])
        return batches

    # Sort by token count ascending
    sorted_t = sorted(turns, key=lambda t: t.get("completion_ids", torch.zeros(0)).shape[0])
    batches = [[] for _ in range(n)]
    token_counts = [0] * n
    for t in sorted_t:
        ntok = t.get("completion_ids", torch.zeros(0)).shape[0]
        idx = min(range(n), key=lambda i: token_counts[i])
        batches[idx].append(t)
        token_counts[idx] += ntok
    return batches
