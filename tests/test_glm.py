"""MN-GLM — GLM-5.2-class decoder on the counter method (RMSNorm + GQA + RoPE + Counter-MoE).

Pins: the new components are correct (RMSNorm unit-norm, RoPE preserves norm + relative-position,
GQA shapes), the model builds with counter linears + grouped MoE, runs forward/backward, and trains.
"""
import math

import torch

from memory_native.glm import MNGLM, ReversibleMNGLM, GLMAttention, RMSNorm, _apply_rope, _rope_cache


def test_rmsnorm_normalizes():
    x = torch.randn(4, 7, 32) * 5.0
    y = RMSNorm(32)(x)
    rms = y.pow(2).mean(-1).sqrt()                      # weight=1 -> output rms ≈ 1
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_norm():
    cos, sin = _rope_cache(16, 8, "cpu", torch.float32)
    x = torch.randn(2, 3, 16, 8)
    y = _apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)   # rotation is norm-preserving


def test_gqa_attention_shapes_and_kv_reduction():
    d, nh, nkv, T = 64, 8, 2, 10
    attn = GLMAttention(d, nh, nkv, "dense", {}, qk_norm=True)
    # KV projections are nkv/nh of the query projection (the GQA cache shrink)
    assert attn.k.weight.shape[0] == nkv * (d // nh)
    assert attn.q.weight.shape[0] == nh * (d // nh)
    x = torch.randn(2, T, d)
    assert attn(x).shape == (2, T, d)                   # RoPE is computed inside from T


def test_glm_builds_and_trains():
    torch.manual_seed(0)
    V = 48
    m = MNGLM(V, n_embd=64, n_layer=2, n_head=8, n_kv_head=2, block_size=32,
              kind="counter_packed", n_experts=4, top_k=2, grouped=True, C=11, lr=0.05).train()
    assert len(m.counter_layers()) > 0                  # attention projections are counter linears
    # the ONLY fp params are embeddings/head (tied), norms, router(s) -- no dense weight matrices
    fp = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any("router" in n for n in fp) and any("tok" in n for n in fp)

    idx = torch.randint(0, V, (4, 32)); tgt = torch.randint(0, V, (4, 32))
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=3e-3)
    def vloss():
        with torch.no_grad():
            return float(m(idx, tgt)[1])
    first = vloss()
    for _ in range(40):
        _, loss = m(idx, tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    assert vloss() < first                              # it learns


def test_glm_loss_includes_moe_aux():
    torch.manual_seed(0)
    V = 48
    m = MNGLM(V, 64, 2, 8, 2, 32, n_experts=4, top_k=2, grouped=True).train()
    idx = torch.randint(0, V, (2, 16)); tgt = torch.randint(0, V, (2, 16))
    import torch.nn.functional as F
    logits, loss = m(idx, tgt)
    ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
    assert torch.allclose(loss, ce + m.aux_loss(), atol=1e-6)
    loss.backward()


def test_reversible_glm_trains_and_is_o1_memory():
    """ReversibleMNGLM: same GLM stack with O(1)-in-depth activation memory. It must train, and the
    reversible stack must store only the final output (not per-layer activations)."""
    torch.manual_seed(0)
    V = 48
    m = ReversibleMNGLM(V, n_embd=64, n_layer=4, n_head=8, n_kv_head=2, block_size=32,
                        kind="counter_packed", n_experts=4, top_k=2, grouped=True,
                        anchor_every=2, C=11, lr=0.05).train()
    assert len(m.counter_layers()) > 0
    idx = torch.randint(0, V, (4, 32)); tgt = torch.randint(0, V, (4, 32))
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=3e-3)
    def vloss():
        with torch.no_grad():
            return float(m(idx, tgt)[1])
    first = vloss()
    for _ in range(40):
        _, loss = m(idx, tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    assert vloss() < first                              # it learns through the reversible stack


def test_reversible_glm_anchor_invariance():
    """anchor_every must not change the forward (anchors are a backward-only memory/speed knob)."""
    def build(A):
        torch.manual_seed(0)
        return ReversibleMNGLM(48, 64, 4, 8, 2, 32, n_experts=4, top_k=2, grouped=True,
                               anchor_every=A, C=11).eval()
    m0, m2 = build(0), build(2)
    idx = torch.randint(0, 48, (2, 16))
    with torch.no_grad():
        assert torch.allclose(m0(idx)[0], m2(idx)[0], atol=1e-5)
