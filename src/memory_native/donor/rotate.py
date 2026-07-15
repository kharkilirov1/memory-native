"""QuaRot-style residual-stream rotation (R1), folded entirely into the weights.

Transformers are computation-invariant to an orthogonal rotation R of the residual stream
once RMSNorm gammas are folded into the adjacent linears: reading layers take W <- W (g R),
writing layers take W <- R^T W, embeddings take E <- E R, the final gamma folds into the
LM head (untied first if tied). RMSNorm without a per-channel weight commutes with R because
an orthogonal map preserves the row norm. Everything is folded -- the runtime graph is
unchanged, so this is deployment-honest (QuaRot/SpinQuant establish the same invariance).

Why it is here: incoherence processing is the single strongest known PTQ lever at extreme
bit-widths -- the rotation spreads weight/activation outliers across channels so a group
ternary grid fits far better (mn-solver v2 design, ingredient A3).

Scope: R1 only (hidden-size rotation). The down_proj INPUT rotation (QuaRot's online
Hadamard on the MLP intermediate) is intentionally omitted -- it needs a runtime op.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["random_orthogonal", "rotate_residual_stream"]


def random_orthogonal(d: int, seed: int = 0) -> torch.Tensor:
    """Haar-ish random orthogonal via QR of a gaussian (fp64), sign-fixed for determinism."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R))


@torch.no_grad()
def rotate_residual_stream(model: nn.Module, seed: int = 0) -> None:
    """Fold gammas + rotate the residual stream of a Qwen2-style causal LM, in place.

    After this call the model computes the SAME function (up to float rounding) -- verified
    by the logits-equality test -- but its weight matrices are incoherent, which is what the
    low-bit solver needs."""
    d = model.config.hidden_size
    emb = model.get_input_embeddings()
    R = random_orthogonal(d, seed).to(emb.weight.device)

    def mul(w: torch.Tensor, *, right: torch.Tensor | None = None,
            left: torch.Tensor | None = None) -> None:
        W = w.to(torch.float64)
        if right is not None:
            W = W @ right
        if left is not None:
            W = left @ W
        w.copy_(W.to(w.dtype))

    # untie the head so the final-norm gamma can fold into it independently of the embedding
    head = model.get_output_embeddings()
    if head.weight is emb.weight:
        head.weight = nn.Parameter(emb.weight.detach().clone())

    final_norm = model.model.norm
    mul(head.weight, right=torch.diag(final_norm.weight.to(torch.float64)) @ R)
    final_norm.weight.fill_(1.0)

    mul(emb.weight, right=R)

    for blk in model.model.layers:
        ln1, ln2 = blk.input_layernorm, blk.post_attention_layernorm
        g1 = torch.diag(ln1.weight.to(torch.float64)) @ R
        g2 = torch.diag(ln2.weight.to(torch.float64)) @ R
        for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj):
            mul(lin.weight, right=g1)                      # readers: W <- W (g R); bias untouched
        ln1.weight.fill_(1.0)
        mul(blk.self_attn.o_proj.weight, left=R.t())       # writers: W <- R^T W (no bias in Qwen2)
        for lin in (blk.mlp.gate_proj, blk.mlp.up_proj):
            mul(lin.weight, right=g2)
        ln2.weight.fill_(1.0)
        mul(blk.mlp.down_proj.weight, left=R.t())
