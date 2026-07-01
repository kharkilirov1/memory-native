"""MN-GLM — a GLM-5.2-class decoder skeleton on the counter-synapse method.

Modern attention stack (RMSNorm + GQA + RoPE + optional QK-norm) with a sparse Counter-MoE FFN
(plan M4, grouped/stacked kernels). Every linear is a counter layer (ternary + 6-bit counter,
self-updating), so the GLM body inherits the method's sub-byte weights + in-state optimizer.

What this adds over the repo's GPT (the GLM build-list): RMSNorm (was LayerNorm), GQA (was full
MHA), RoPE (was learned positional embeddings), QK-norm. The MoE FFN reuses CounterMoEFFN; SwiGLU
experts are a drop-in expert-MLP variant (the current experts are fc->gelu->fc2, kept so the fast
grouped kernel applies -- the MoE quality win is activation-agnostic). Reversible wrapping (the
activation-memory lever) composes on top and is left to ReversibleSequence as in ReversibleGPT.

Pure PyTorch, CPU/CUDA.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import make_linear
from .counter import CompactCounterLinear
from .moe_ffn import CounterMoEFFN
from .reversible import ReversibleCouplingBlock, ReversibleSequence

__all__ = ["RMSNorm", "GLMAttention", "GLMBlock", "MNGLM", "ReversibleMNGLM"]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (GLM/Llama style): no mean-subtraction, one scale per channel."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def _rope_cache(T: int, head_dim: int, device, dtype, base: float = 10000.0):
    """cos/sin tables [T, head_dim] for rotary position embedding (NeoX/Llama 'rotate_half' form)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)   # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                             # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [B, H, T, head_dim]; cos/sin [T, head_dim]
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class GLMAttention(nn.Module):
    """Grouped-query attention with RoPE and optional QK-norm; all projections are counter linears.

    n_head query heads, n_kv_head key/value heads (GQA: n_head % n_kv_head == 0). The KV heads are
    repeated to match the query heads before SDPA, shrinking the KV projection / cache by the ratio.
    """

    def __init__(self, d: int, n_head: int, n_kv_head: int, kind: str, ckw: dict,
                 qk_norm: bool = True) -> None:
        super().__init__()
        if d % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head (GQA)")
        self.nh, self.nkv = n_head, n_kv_head
        self.hd = d // n_head
        if self.hd % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.d = d
        self.q = make_linear(kind, d, n_head * self.hd, 1.0, **ckw)
        self.k = make_linear(kind, d, n_kv_head * self.hd, 1.0, **ckw)
        self.v = make_linear(kind, d, n_kv_head * self.hd, 1.0, **ckw)
        self.o = make_linear(kind, d, d, 1.0, **ckw)
        self.qn = RMSNorm(self.hd) if qk_norm else None
        self.kn = RMSNorm(self.hd) if qk_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        cos, sin = _rope_cache(T, self.hd, x.device, x.dtype)          # computed internally (from T)
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)     # [B, nh, T, hd]
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)    # [B, nkv, T, hd]
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.qn is not None:
            q = self.qn(q); k = self.kn(k)                             # QK-norm (per-head-dim RMS)
        q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)                        # GQA: broadcast KV to all q heads
            v = v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, self.d)
        return self.o(a)


def _counter_numeric(kw: dict) -> dict:
    return {k: kw[k] for k in ("C", "lr", "lr_scale") if k in kw}


class GLMBlock(nn.Module):
    """RMSNorm-pre-norm GLM block: attention sublayer + Counter-MoE FFN sublayer, both residual."""

    def __init__(self, d: int, n_head: int, n_kv_head: int, kind: str, ckw: dict,
                 n_experts: int, top_k: int, qk_norm: bool, grouped: bool,
                 aux_loss_weight: float = 1e-2) -> None:
        super().__init__()
        self.n_embd = d
        self.n1 = RMSNorm(d)
        self.attn = GLMAttention(d, n_head, n_kv_head, kind, ckw, qk_norm)
        self.n2 = RMSNorm(d)
        self.ffn = CounterMoEFFN(d, n_experts=n_experts, top_k=top_k, grouped=grouped,
                                 aux_loss_weight=aux_loss_weight, **_counter_numeric(ckw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class MNGLM(nn.Module):
    """GLM-5.2-class decoder on the counter method: RMSNorm + GQA + RoPE + Counter-MoE, tied head."""

    def __init__(self, vocab_size: int, n_embd: int, n_layer: int, n_head: int, n_kv_head: int,
                 block_size: int, *, kind: str = "counter_packed", n_experts: int = 8, top_k: int = 2,
                 qk_norm: bool = True, grouped: bool = True, aux_loss_weight: float = 1e-2,
                 **counter_kw) -> None:
        super().__init__()
        self.block_size = block_size
        self.d = n_embd
        self.n_head = n_head
        self.tok = nn.Embedding(vocab_size, n_embd)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.blocks = nn.ModuleList(
            GLMBlock(n_embd, n_head, n_kv_head, kind, counter_kw, n_experts, top_k,
                     qk_norm, grouped, aux_loss_weight)
            for _ in range(n_layer))
        self.nf = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok.weight                             # tie

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.block_size}")
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.nf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            aux = self.aux_loss()
            if aux is not None:
                loss = loss + aux
        return logits, loss

    def aux_loss(self) -> torch.Tensor | None:
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
        return [p for p in self.parameters() if p.requires_grad]


class _GLMAttnSub(nn.Module):
    """F for the reversible coupling: RMSNorm -> GQA(+RoPE, QK-norm) -> out-proj, maps [.,d]->[.,d]."""

    def __init__(self, d, n_head, n_kv_head, kind, ckw, qk_norm):
        super().__init__()
        self.d = d
        self.n = RMSNorm(d)
        self.attn = GLMAttention(d, n_head, n_kv_head, kind, ckw, qk_norm)

    def forward(self, x):
        return self.attn(self.n(x))


class _GLMFFNSub(nn.Module):
    """G for the reversible coupling: RMSNorm -> Counter-MoE FFN, maps [.,d]->[.,d]."""

    def __init__(self, d, kind, ckw, n_experts, top_k, grouped, aux_loss_weight):
        super().__init__()
        self.d = d
        self.n = RMSNorm(d)
        self.ffn = CounterMoEFFN(d, n_experts=n_experts, top_k=top_k, grouped=grouped,
                                 aux_loss_weight=aux_loss_weight, **_counter_numeric(ckw))

    def forward(self, x):
        return self.ffn(self.n(x))


class ReversibleMNGLM(nn.Module):
    """MN-GLM with O(1)-in-depth activation memory: the GLM blocks are wrapped in a reversible
    coupling stack (F = attention sublayer, G = MoE FFN sublayer). Same GLM-5.2 components + tuned
    counter knobs as MNGLM; adds the activation-memory lever (anchor_every is the speed/memory knob,
    2-4 recommended per PERF_ANATOMY). The two reversible streams are the duplicated embedding."""

    def __init__(self, vocab_size: int, n_embd: int, n_layer: int, n_head: int, n_kv_head: int,
                 block_size: int, *, kind: str = "counter_packed", n_experts: int = 8, top_k: int = 2,
                 qk_norm: bool = True, grouped: bool = True, aux_loss_weight: float = 1e-2,
                 anchor_every: int = 0, **counter_kw) -> None:
        super().__init__()
        self.block_size = block_size
        self.d = n_embd
        self.tok = nn.Embedding(vocab_size, n_embd)
        nn.init.normal_(self.tok.weight, std=0.02)
        blocks = [ReversibleCouplingBlock(
                      2 * n_embd,
                      F=_GLMAttnSub(n_embd, n_head, n_kv_head, kind, counter_kw, qk_norm),
                      G=_GLMFFNSub(n_embd, kind, counter_kw, n_experts, top_k, grouped, aux_loss_weight))
                  for _ in range(n_layer)]
        self.rev = ReversibleSequence(blocks, anchor_every=anchor_every)
        self.nf = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.block_size}")
        e = self.tok(idx)
        x = torch.cat([e, e], dim=-1)                                  # two reversible streams
        x = self.rev(x)
        h = self.nf(x[..., :self.d] + x[..., self.d:])                 # recombine
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            aux = self.aux_loss()
            if aux is not None:
                loss = loss + aux
        return logits, loss

    def aux_loss(self) -> torch.Tensor | None:
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
        return [p for p in self.parameters() if p.requires_grad]
