"""Residual-stream rotation witnesses.

The load-bearing claim is computation invariance: after fold+rotate the model must produce
the SAME logits (float tolerance). If that holds, using the rotated model for PTQ is honest
-- the rotation is free at runtime. The second claim is the point of the exercise: the
rotated weights quantize BETTER (incoherence spreads outliers)."""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM

from memory_native.donor.rotate import random_orthogonal, rotate_residual_stream


def _tiny(seed=0, vocab=64):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(cfg).eval()


def test_random_orthogonal_is_orthogonal():
    R = random_orthogonal(48, seed=3)
    assert torch.allclose(R @ R.t(), torch.eye(48, dtype=R.dtype), atol=1e-10)


def test_rotation_preserves_logits():
    m = _tiny(0)
    ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        ref = m(ids).logits
    rotate_residual_stream(m, seed=1)
    # gammas folded away, head untied
    assert torch.allclose(m.model.norm.weight, torch.ones_like(m.model.norm.weight))
    assert m.get_output_embeddings().weight is not m.get_input_embeddings().weight
    with torch.no_grad():
        got = m(ids).logits
    assert torch.allclose(got, ref, atol=1e-4), float((got - ref).abs().max())


def test_rotation_hurts_group_ternary_fit_documented_negative():
    """MEASURED NEGATIVE (design ingredient A3 retired for ternary): incoherence rotations
    -- the strongest lever for uniform 2-bit grids (QuIP/QuaRot) -- HURT the group-ternary
    grid. Ternary+group-scale THRIVES on concentrated outliers (an outlier group gets its
    own large scale); rotation gaussianizes the weights, the worst case for a 3-level
    alphabet. Pinned so nobody re-tries it silently; the outlier-PRESERVING path (BiLLM-style
    salient residuals, A4) is the correct ternary direction."""
    from memory_native.donor.ptq import gptq_group_ternary
    torch.manual_seed(5)
    d, out = 64, 48
    w = torch.randn(out, d) * 0.02
    w[:, :3] *= 25.0                                   # channel outliers (the LLM pathology)
    X = torch.randn(512, d) * 0.1
    R = random_orthogonal(d, seed=7).to(torch.float32)

    def err(wm, Xm):
        w_hat, _, _ = gptq_group_ternary(wm, Xm.t() @ Xm, group=32, act_order=True)
        return float(((Xm @ (wm - w_hat).t()) ** 2).sum())

    e_plain = err(w, X)
    e_rot = err(w @ R, X @ R)
    assert e_rot > e_plain, (e_rot, e_plain)           # if this flips, revisit A3 for ternary
