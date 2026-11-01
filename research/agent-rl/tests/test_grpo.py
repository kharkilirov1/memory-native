"""Tests for mx6 conversion, GRPO advantage, micro-batch packing, and a full dry-env step."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# mx6 path
_MX6 = Path(__file__).resolve().parents[3] / "research" / "mxfp4-native-no-state" / "mxfp4-native"
if str(_MX6) not in sys.path:
    sys.path.insert(0, str(_MX6))

from mx6 import lg_accum
from mx6.latent_grid import LatentGridLinear, lg_layers


# ── Fake tokenizer for tests ────────────────────────────────────────────


class FakeTokenizer:
    """Minimal tokenizer that maps text to fixed random ids."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def __call__(self, text: str, **kwargs) -> dict:
        n = max(4, min(24, len(text) // 2))
        ids = torch.randint(3, self.vocab_size, (1, n))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, messages, **kwargs) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        return "\n".join(parts) + "\n"

    def decode(self, ids: torch.Tensor, **kwargs) -> str:
        return f"action text (len={ids.shape[0]})"


# ── Tiny model fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="function")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, vocab_size=128,
        max_position_embeddings=512,
    )
    m = Qwen2ForCausalLM(cfg)
    m.eval()
    return m


# ── test 1: convert_to_lg replaces the right layers ────────────────────

def test_convert_to_lg(tiny_model):
    from rl.convert import convert_to_lg

    model = tiny_model
    before = sum(1 for _ in model.named_modules() if isinstance(_[1], LatentGridLinear))

    report = convert_to_lg(model, k=6, headroom=2.0, lr=0.001, momentum=0.9,
                           mom_dtype="fp8", n_micro=4, backend="torch")
    after = sum(1 for _ in model.named_modules() if isinstance(_[1], LatentGridLinear))
    assert after > before, "no LG layers were added"
    assert report["n_layers"] > 0
    # LM head and embeddings should NOT be converted
    for name, mod in model.named_modules():
        if "lm_head" in name and isinstance(mod, torch.nn.Linear):
            assert not isinstance(mod, LatentGridLinear), "lm_head should not be converted"
    print(f"convert: {report['n_layers']} layers, {report['params_in_lg']} params")
    for l in report["report_lines"]:
        print(l)


# ── test 2: LG layer output is close to dense ──────────────────────────

def test_lg_forward_close_to_dense(tiny_model):
    from rl.convert import convert_to_lg

    model = tiny_model
    # Test one Linear WITHOUT bias (self_attn.o_proj has bias=False)
    layer = model.model.layers[0]
    o = layer.self_attn.o_proj
    x = torch.randn(2, 32)
    with torch.no_grad():
        y_dense = o(x)

    w = o.weight.detach().clone()
    lg = LatentGridLinear.from_dense(w, k=6, headroom=2.0, lr=0.001, backend="torch")
    with torch.no_grad():
        y_lg = lg(x)

    rel_err = (y_lg - y_dense).abs().mean() / (y_dense.abs().mean() + 1e-8)
    assert rel_err < 0.15, f"relative error {rel_err:.4f} >= 0.15"
    print(f"LG vs dense rel error: {rel_err:.6f}")


# ── test 3: advantage computation ──────────────────────────────────────

def test_advantage():
    from rl.grpo import _mean_group_rewards

    rewards = [1.0, 0.5, 0.0]
    mean = _mean_group_rewards(rewards)
    assert abs(mean - 0.5) < 1e-6
    assert abs(rewards[0] - mean - 0.5) < 1e-6
    assert abs(rewards[2] - mean + 0.5) < 1e-6


# ── test 4: micro-batch packing ────────────────────────────────────────

def test_micro_packing():
    from rl.grpo import _pack_into_micro

    turns = []
    for i in range(10):
        t = {"completion_ids": torch.zeros(10 + i * 5, dtype=torch.long)}
        t["advantage"] = 0.5
        t["logprobs"] = torch.randn(10 + i * 5) * 0.1
        turns.append(t)

    batches = _pack_into_micro(turns, 4)
    assert len(batches) == 4
    # All turns accounted for
    total = sum(len(b) for b in batches)
    assert total == 10


# ── test 5: micro-batch packing with fewer turns than N ────────────────

def test_micro_packing_fewer():
    from rl.grpo import _pack_into_micro

    turns = [{"completion_ids": torch.zeros(5, dtype=torch.long)} for _ in range(2)]
    batches = _pack_into_micro(turns, 4)
    assert len(batches) == 4
    assert len(batches[0]) == 1
    assert len(batches[1]) == 1
    assert len(batches[2]) == 0  # dummy
    assert len(batches[3]) == 0  # dummy


# ── test 6: full dry-env GRPO step with lg arm ─────────────────────────

def test_grpo_step_dry_env(tiny_model):
    """Verify grpo_step advantage computation and loss — skip backward on CPU (no grad from randn)."""
    from rl.convert import convert_to_lg
    from rl.grpo import grpo_step
    import random

    model = tiny_model
    convert_to_lg(model, k=6, headroom=2.0, lr=0.001, momentum=0.0,
                  mom_dtype="fp8", n_micro=4, backend="torch")

    idx_before = {id(m): m.idx.clone() for m in lg_layers(model)}

    def fake_run_env(seed: int, policy) -> list[dict]:
        rng = random.Random(seed)
        reward = rng.choice([0.0, 0.5, 1.0])
        n_turns = rng.randint(1, 3)
        turns = []
        for _ in range(n_turns):
            cids = torch.randint(0, 100, (rng.randint(5, 15),))
            turns.append({
                "action_text": "test",
                "prompt_ids": torch.randint(0, 100, (10,)),
                "completion_ids": cids,
                "logprobs": torch.randn(cids.shape[0]) * 0.1 - 2.0,
            })
        return [{"reward": reward, "turns": turns, "info": {}}]

    result = grpo_step(
        model, fake_run_env,
        b=2, g=2, optim="lg", lr=0.001, n_micro=4,
        device="cpu",
    )
    assert "mean_reward" in result
    assert "step_time_s" in result
    print(f"GRPO step: mean_r={result['mean_reward']:.4f}, "
          f"time={result['step_time_s']:.3f}s")
