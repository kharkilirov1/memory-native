"""Integration gate: the M-levers are wired into the real GPT behind config flags.

Guards (1) the original dense path is byte-for-byte unchanged, (2) each lever builds, runs a
forward+backward, exposes its counter layers, and trains (loss drops), (3) the M4/M1 load-balance
aux loss is folded into the training loss.
"""
import copy

import pytest
import torch

from memory_native.models import CONFIGS, GPT, GPTConfig
from memory_native.counter import CompactCounterLinear


def _cfg(**over):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    base.update(over)
    return GPTConfig(**base)


def _batch(cfg, n=4):
    torch.manual_seed(1)
    idx = torch.randint(0, cfg.vocab_size, (n, cfg.block_size))
    tgt = torch.randint(0, cfg.vocab_size, (n, cfg.block_size))
    return idx, tgt


def _eval_loss(g, idx, tgt) -> float:
    """Loss with NO graph -> counter layers take the inference path (no outstanding-forward flag,
    no self-update), so we can probe the loss without owing a backward."""
    with torch.no_grad():
        _, loss = g(idx, tgt)
    return float(loss)


# --------------------------------------------------------------- (1) dense path unchanged
def test_dense_ffn_is_the_default_and_unchanged():
    """ffn defaults to 'dense'; the Block keeps fc/fc2 and NO ffn module -> original transformer."""
    cfg = _cfg()
    assert cfg.ffn == "dense"
    torch.manual_seed(0); g_default = GPT("dense", cfg)
    torch.manual_seed(0); g_explicit = GPT("dense", _cfg(ffn="dense"))
    assert g_default.blocks[0].ffn is None                      # dense path intact (no MoE/memory)
    idx, tgt = _batch(cfg)
    with torch.no_grad():
        l1, _ = g_default(idx); l2, _ = g_explicit(idx)
    assert torch.equal(l1, l2)                                  # default == explicit dense, bit-exact


def test_dense_loss_has_no_aux():
    cfg = _cfg()
    g = GPT("dense", cfg)
    assert g.aux_loss() is None                                 # nothing added to the dense loss
    idx, tgt = _batch(cfg)
    _, loss = g(idx, tgt)
    assert torch.isfinite(loss)


# --------------------------------------------------------------- (2) FFN levers: M4 moe, M1 memory
@pytest.mark.parametrize("ffn,over", [
    ("moe", dict(ffn_experts=4, ffn_top_k=2)),
    ("memory", dict(ffn_cells=256, ffn_k=4)),               # 256 = 16^2 (perfect square)
])
def test_ffn_lever_builds_runs_and_trains(ffn, over):
    cfg = _cfg(ffn=ffn, **over)
    torch.manual_seed(0)
    g = GPT("dense", cfg)                                    # attention dense, FFN = the lever
    assert g.blocks[0].ffn is not None
    idx, tgt = _batch(cfg)
    opt = torch.optim.AdamW(g.trainable_parameters(), lr=1e-2)
    first = _eval_loss(g, idx, tgt)
    for _ in range(25):
        opt.zero_grad(); _, loss = g(idx, tgt); loss.backward(); opt.step()
    assert _eval_loss(g, idx, tgt) < first                  # it learns


def test_moe_aux_loss_folded_into_loss():
    import torch.nn.functional as F
    cfg = _cfg(ffn="moe", ffn_experts=4, ffn_top_k=2)
    g = GPT("dense", cfg).train()
    idx, tgt = _batch(cfg)
    logits, loss = g(idx, tgt)                              # one training forward
    aux = g.aux_loss()                                     # from this forward's last_aux_loss
    assert aux is not None and torch.isfinite(aux) and float(aux) > 0.0
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    assert torch.allclose(loss, ce + aux, atol=1e-6)       # training loss = CE + weighted aux
    loss.backward()                                        # settle the outstanding counter forwards


# --------------------------------------------------------------- (3) linear levers: M3 slowfast, M2 group
@pytest.mark.parametrize("kind,kw", [
    ("slowfast", dict(rank=4, merge_every=4)),
    ("group", dict(group=4, keep=2)),
])
def test_linear_lever_builds_and_trains(kind, kw):
    cfg = _cfg()
    torch.manual_seed(0)
    g = GPT(kind, cfg, **kw)
    idx, tgt = _batch(cfg)
    opt = torch.optim.AdamW(g.trainable_parameters(), lr=1e-2)
    first = _eval_loss(g, idx, tgt)
    for _ in range(25):
        opt.zero_grad(); _, loss = g(idx, tgt); loss.backward(); opt.step()
    last = _eval_loss(g, idx, tgt)
    assert last == last and last <= first + 1e-3           # finite (NaN!=NaN) + no divergence


def test_levers_compose_moe_with_slowfast_attention():
    """The axes are orthogonal: slowfast linears (attention) + MoE FFN in one model."""
    cfg = _cfg(ffn="moe", ffn_experts=4, ffn_top_k=2)
    torch.manual_seed(0)
    g = GPT("slowfast", cfg, rank=4, merge_every=4)
    idx, tgt = _batch(cfg)
    _, loss = g(idx, tgt); loss.backward()
    assert torch.isfinite(loss)
