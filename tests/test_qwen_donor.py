"""Phase 2-remaining witness: warm-start a real-shaped (but tiny, random) Qwen2 donor into the
counter format by in-place swapping its body linears. No network: transformers instantiates a
random model from a tiny config, so this runs on CPU in CI."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from memory_native.counter import CompactCounterLinear, RMSCounterLinear
from memory_native.convert import CounterLinearWithBias
from memory_native.donor.qwen import qwen_to_counter

BODY = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BIASED = ("q_proj", "k_proj", "v_proj")   # Qwen2 attention projections carry a bias


def _tiny_qwen(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg).eval()


def test_swap_covers_body_and_skips_head():
    m = _tiny_qwen()
    report = qwen_to_counter(m)
    # every transformer-body linear got swapped, in both layers (7 per layer x 2 layers)
    assert len(report.swapped) == 7 * 2
    assert all(any(p.endswith(b) for b in BODY) for p in report.swapped)
    # the tied lm_head is left fp (skipped), never swapped
    assert "lm_head" in report.skipped
    assert not any("lm_head" in p for p in report.swapped)
    assert report.coeffs > 0


def test_module_types_and_bias_preserved():
    m = _tiny_qwen()
    # snapshot the donor q_proj bias before the in-place swap mutates the tree
    ref_bias = m.model.layers[0].self_attn.q_proj.bias.detach().clone()
    qwen_to_counter(m)

    attn0 = m.model.layers[0].self_attn
    mlp0 = m.model.layers[0].mlp
    # biased projections -> counter wrapped with a preserved fp bias
    for name in BIASED:
        mod = getattr(attn0, name)
        assert isinstance(mod, CounterLinearWithBias), name
        assert isinstance(mod.counter, RMSCounterLinear), name
    # bias-free linears -> bare counter layers
    for mod in (attn0.o_proj, mlp0.gate_proj, mlp0.up_proj, mlp0.down_proj):
        assert isinstance(mod, RMSCounterLinear)
    # the preserved bias is exactly the donor's
    assert torch.equal(m.model.layers[0].self_attn.q_proj.bias, ref_bias)
    # head stays a plain fp Linear; embeddings stay an fp Embedding
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, CompactCounterLinear)
    assert isinstance(m.model.embed_tokens, nn.Embedding)


def test_forward_runs_and_degrades():
    m = _tiny_qwen()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        fp_logits = m(ids).logits.clone()
    qwen_to_counter(m)
    with torch.no_grad():
        cn_logits = m(ids).logits
    assert cn_logits.shape == fp_logits.shape
    assert torch.isfinite(cn_logits).all()
    # ternarization is lossy for a full-precision donor -> logits must actually move
    assert (cn_logits - fp_logits).abs().mean() > 1e-4


def test_eager_guard_survives_two_train_steps():
    """Counter layers are eager-only (one forward per backward). A standard HF forward must call
    each projection exactly once, so two consecutive train steps must not trip the reuse guard."""
    m = _tiny_qwen()
    qwen_to_counter(m)
    m.train()
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        out = m(ids, labels=ids)
        out.loss.backward()
        m.zero_grad(set_to_none=True)
