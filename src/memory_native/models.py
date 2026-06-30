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
    # --- architecture levers (default = the original dense-FFN transformer, unchanged) ---
    # ffn: the FFN sub-block. "dense" = fc->gelu->fc2 (original). "moe" = Counter-MoE (M4, the
    #   validated win: capacity in experts at ~dense active compute). "memory" = product-key
    #   CounterMemoryFFN (M1, a compute-saver; lower quality than dense, opt-in).
    ffn: str = "dense"
    ffn_experts: int = 8      # M4: E total experts (capacity, not active compute)
    ffn_top_k: int = 2        # M4: experts visited per token (active compute)
    ffn_cells: int = 16384    # M1: number of memory cells (perfect square)
    ffn_k: int = 8            # M1: cells retrieved per token


CONFIGS = {
    "micro": GPTConfig(32, 16, 2, 2, 32),
    "tiny": GPTConfig(2048, 128, 4, 4, 256),
    "s512": GPTConfig(8192, 256, 8, 8, 512),
    "small": GPTConfig(8192, 256, 12, 12, 768),
}


def _counter_numeric(kw: dict) -> dict:
    """The counter knobs the FFN-level modules (M1/M4) accept -- they take only C/lr/lr_scale,
    not the linear-level kw (tile_rows, rms_mode, cache_mode, ...). Filter so an arm built with a
    full counter_kw can still select ffn='moe'/'memory' without passing unexpected kwargs."""
    return {k: kw[k] for k in ("C", "lr", "lr_scale") if k in kw}


class Block(nn.Module):
    def __init__(self, kind: str, cfg: GPTConfig, **kw) -> None:
        super().__init__()
        d, nl = cfg.n_embd, cfg.n_layer
        rg = 1.0 / math.sqrt(2.0 * nl)
        self.n_embd = d                     # expose for MoD / dim inference
        self.ln1 = nn.LayerNorm(d)
        self.q = make_linear(kind, d, d, 1.0, **kw)
        self.k = make_linear(kind, d, d, 1.0, **kw)
        self.v = make_linear(kind, d, d, 1.0, **kw)
        self.proj = make_linear(kind, d, d, rg, **kw)
        self.ln2 = nn.LayerNorm(d)
        self.n_head = cfg.n_head
        # FFN sub-block: dense (original) or a counter architecture lever (M4 moe / M1 memory).
        self.ffn_kind = cfg.ffn
        if cfg.ffn == "dense":
            self.fc = make_linear(kind, d, 4 * d, 1.0, **kw)
            self.fc2 = make_linear(kind, 4 * d, d, rg, **kw)
            self.ffn = None
        elif cfg.ffn == "moe":
            from .moe_ffn import CounterMoEFFN
            self.ffn = CounterMoEFFN(d, n_experts=cfg.ffn_experts, top_k=cfg.ffn_top_k,
                                     **_counter_numeric(kw))
        elif cfg.ffn == "memory":
            from .memory_ffn import CounterMemoryFFN
            self.ffn = CounterMemoryFFN(d, n_cells=cfg.ffn_cells, k=cfg.ffn_k,
                                        **_counter_numeric(kw))
        else:
            raise ValueError(f"unknown ffn kind: {cfg.ffn!r} (dense|moe|memory)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, t, self.n_head, -1).transpose(1, 2)
        k = self.k(h).view(b, t, self.n_head, -1).transpose(1, 2)
        v = self.v(h).view(b, t, self.n_head, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        h2 = self.ln2(x)
        x = x + (self.fc2(F.gelu(self.fc(h2))) if self.ffn is None else self.ffn(h2))
        return x


class GPT(nn.Module):
    """kind in {dense, qat, counter, counter_rms, counter_packed, counter_triton, slowfast (M3),
    group (M2)} selects the LINEAR type; cfg.ffn in {dense, moe (M4), memory (M1)} selects the FFN
    sub-block. counter_kw are forwarded to the counter layers (lr, lr_scale, C, tile_rows,
    rms_mode, ...); the FFN-level levers take only the C/lr/lr_scale subset."""

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
            aux = self.aux_loss()           # M1/M4 load-balance term (None if no such FFN)
            if aux is not None:
                loss = loss + aux
        return logits, loss

    def aux_loss(self) -> torch.Tensor | None:
        """Sum the FFN load-balance aux losses (M4 Counter-MoE / M1 memory) weighted by each
        module's own aux_loss_weight, from the last forward. None when no FFN exposes one (e.g.
        the dense FFN) -- so the training loss is unchanged for the original transformer."""
        total = None
        for m in self.modules():
            w = float(getattr(m, "aux_loss_weight", 0.0) or 0.0)
            if w > 0.0 and hasattr(m, "last_aux_loss"):
                term = w * m.last_aux_loss
                total = term if total is None else total + term
        return total

    def counter_layers(self) -> list[CompactCounterLinear]:
        return [m for m in self.modules() if isinstance(m, CompactCounterLinear)]

    def trainable_parameters(self):
        """Conventional Parameters (embeddings, norms, head) — counter layers self-update
        and expose no parameters(), so this is exactly what an AdamW should own."""
        return [p for p in self.parameters() if p.requires_grad]
