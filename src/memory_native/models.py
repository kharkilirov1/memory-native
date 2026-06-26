"""A small GPT harness whose linear layers are swappable (dense / qat / counter / counter_rms),
so the same model is the parity comparison. Pure PyTorch, runs on CPU/CUDA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import make_linear
from .counter import CompactCounterLinear

__all__ = ["GPTConfig", "CONFIGS", "GPT"]


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int


CONFIGS = {
    "micro": GPTConfig(32, 16, 2, 2, 32),
    "tiny": GPTConfig(2048, 128, 4, 4, 256),
    "s512": GPTConfig(8192, 256, 8, 8, 512),
    "small": GPTConfig(8192, 256, 12, 12, 768),
}


class Block(nn.Module):
    def __init__(self, kind: str, cfg: GPTConfig, **kw) -> None:
        super().__init__()
        d, nl = cfg.n_embd, cfg.n_layer
        rg = 1.0 / math.sqrt(2.0 * nl)
        self.ln1 = nn.LayerNorm(d)
        self.q = make_linear(kind, d, d, 1.0, **kw)
        self.k = make_linear(kind, d, d, 1.0, **kw)
        self.v = make_linear(kind, d, d, 1.0, **kw)
        self.proj = make_linear(kind, d, d, rg, **kw)
        self.ln2 = nn.LayerNorm(d)
        self.fc = make_linear(kind, d, 4 * d, 1.0, **kw)
        self.fc2 = make_linear(kind, 4 * d, d, rg, **kw)
        self.n_head = cfg.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, t, self.n_head, -1).transpose(1, 2)
        k = self.k(h).view(b, t, self.n_head, -1).transpose(1, 2)
        v = self.v(h).view(b, t, self.n_head, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    """kind in {dense, qat, counter, counter_rms}. counter_kw are forwarded to the counter
    layers (lr, lr_scale, C, tile_rows, pulse_mode, ...)."""

    def __init__(self, kind: str, cfg: GPTConfig, **counter_kw) -> None:
        super().__init__()
        self.cfg = cfg
        self.kind = kind
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(kind, cfg, **counter_kw) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tie input/output embedding

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def counter_layers(self) -> list[CompactCounterLinear]:
        return [m for m in self.modules() if isinstance(m, CompactCounterLinear)]

    def trainable_parameters(self):
        """Conventional Parameters (embeddings, norms, head) — counter layers self-update
        and expose no parameters(), so this is exactly what an AdamW should own."""
        return [p for p in self.parameters() if p.requires_grad]
