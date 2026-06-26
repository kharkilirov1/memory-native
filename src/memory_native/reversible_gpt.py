"""The FULL method as one model: a reversible-counter transformer.

Both memory levers active at once:
  * every linear (q,k,v,proj,fc,fc2) is a finite-state counter layer (counter_packed +
    act_save_bits) -> weights/optimizer/gradients in <1 byte/weight, no Adam moments;
  * the transformer stack is wrapped in ReversibleSequence -> O(1)-in-depth activation memory
    (the whole chain stores only its final output and reconstructs in backward).

Reformer-style coupling: the residual stream is duplicated into two width-d halves; each block
is y1 = x1 + F(x2), y2 = x2 + G(y1) with F = attention sub-block, G = MLP sub-block. Only
embeddings, LayerNorms and the tied head are conventional Parameters (an AdamW trains those);
the counter layers self-update inside the reversible backward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import make_linear
from .counter import CompactCounterLinear
from .models import GPTConfig
from .reversible import ReversibleCouplingBlock, ReversibleSequence

__all__ = ["ReversibleGPT"]


class _AttnSub(nn.Module):
    """F: LayerNorm -> causal multi-head attention with counter q/k/v/proj. [B,T,d] -> [B,T,d]."""

    def __init__(self, d: int, n_head: int, kind: str, counter_kw: dict) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.q = make_linear(kind, d, d, 1.0, **counter_kw)
        self.k = make_linear(kind, d, d, 1.0, **counter_kw)
        self.v = make_linear(kind, d, d, 1.0, **counter_kw)
        self.proj = make_linear(kind, d, d, 1.0, **counter_kw)
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.ln(x)
        q = self.q(h).view(b, t, self.n_head, -1).transpose(1, 2)
        k = self.k(h).view(b, t, self.n_head, -1).transpose(1, 2)
        v = self.v(h).view(b, t, self.n_head, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # deterministic (no dropout)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(a)


class _MLPSub(nn.Module):
    """G: LayerNorm -> counter MLP (fc, gelu, fc2). [B,T,d] -> [B,T,d]."""

    def __init__(self, d: int, kind: str, counter_kw: dict) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc = make_linear(kind, d, 4 * d, 1.0, **counter_kw)
        self.fc2 = make_linear(kind, 4 * d, d, 1.0, **counter_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc(self.ln(x))))


class ReversibleGPT(nn.Module):
    """Full-method GPT: counter linears + O(1) reversible activation memory.

    counter_kw are forwarded to every counter layer (lr, lr_scale, C, act_save_bits, ...).
    kind defaults to 'counter_packed'; pass 'dense' to get a reversible *dense* baseline.
    """

    def __init__(self, cfg: GPTConfig, kind: str = "counter_packed", **counter_kw) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.n_embd
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.block_size, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        blocks = [ReversibleCouplingBlock(2 * d,
                                          F=_AttnSub(d, cfg.n_head, kind, counter_kw),
                                          G=_MLPSub(d, kind, counter_kw))
                  for _ in range(cfg.n_layer)]
        self.rev = ReversibleSequence(blocks)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tie

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(t, device=idx.device)
        e = self.tok(idx) + self.pos(pos)[None]
        x = torch.cat([e, e], dim=-1)          # duplicate into the two reversible streams
        x = self.rev(x)
        d = self.cfg.n_embd
        h = self.lnf(x[..., :d] + x[..., d:])  # recombine the two streams
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def counter_layers(self) -> list[CompactCounterLinear]:
        return [m for m in self.modules() if isinstance(m, CompactCounterLinear)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
