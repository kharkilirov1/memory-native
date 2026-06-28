"""M10 -- Mixture-of-Depths (MoD): a per-block token router for fewer FLOPs/token.

External basis: Raposo et al., "Mixture-of-Depths: Dynamically allocating compute in
transformer-based language models" (DeepMind, 2024).

Idea: not every token needs every block. A tiny fp router scores each token; the top
`capacity` fraction (e.g. 0.5) are passed through the wrapped block (attention+FFN);
the remaining tokens BYPASS the block entirely via the residual -- their compute is
skipped. The compute budget per block is therefore fixed at `capacity * tokens`.

So the router learns *which* tokens matter, we multiply the block's contribution for the
selected tokens by the router's sigmoid weight (a weighted residual / straight-through
style). Skipped tokens are returned bit-identically (pure residual identity).

CAUSAL CAVEAT (important): the standard MoD top-k selection is a NON-CAUSAL operation --
it ranks tokens within a fixed window, so token i's keep/skip decision can depend on
later tokens. For training a witness on a fixed block this is fine (and is exactly the
"expert-choice"-style routing DeepMind use during training). At inference / for a strictly
causal model you would instead train a per-token predictor of the routing decision (the
"router predictor" in the paper) so no future information leaks. We do top-k over the
batch*time tokens here for the witness; this is acceptable for measuring the FLOP/quality
tradeoff but is NOT a drop-in causal inference path.

Pure PyTorch, CPU/CUDA.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MoDBlock"]


class MoDBlock(nn.Module):
    """Wrap a block (any ``nn.Module`` mapping ``(B,T,D) -> (B,T,D)``) with token routing.

    Args:
        block: the inner block to (conditionally) apply. It MUST already be a residual block,
            i.e. ``block(x)`` returns ``x + delta`` (as the TinyGPT / models.py Block does).
            MoD then mixes ``block(x_sel)`` for selected tokens and passes the rest through
            unchanged.
        capacity: fraction in (0, 1] of tokens to route through the block. ``1.0`` means every
            token is processed (exact parity with the unwrapped block, modulo the router weight,
            see ``use_router_weight``).
        use_router_weight: if True (default, the MoD recipe), the block's *contribution* for a
            selected token is scaled by the router's sigmoid weight, so the router gets a
            gradient signal telling it which tokens benefit from the block. At capacity=1.0 this
            is disabled automatically so the wrapper is an exact identity over the block.

    The realized fraction of tokens actually processed is exposed via ``last_processed_fraction``
    after each forward (and via ``realized_fraction()``).
    """

    def __init__(self, block: nn.Module, capacity: float = 0.5,
                 use_router_weight: bool = True) -> None:
        super().__init__()
        if not (0.0 < capacity <= 1.0):
            raise ValueError(f"capacity must be in (0, 1], got {capacity}")
        self.block = block
        self.capacity = float(capacity)
        self.use_router_weight = bool(use_router_weight)
        self._d = self._infer_dim(block)
        # tiny fp router: one score per token. AdamW owns this Parameter.
        self.router = nn.Linear(self._d, 1)
        nn.init.zeros_(self.router.bias)
        nn.init.normal_(self.router.weight, std=0.02)
        self.last_processed_fraction: float = 0.0

    @staticmethod
    def _infer_dim(block: nn.Module) -> int:
        for attr in ("n_embd", "d", "dim"):
            if hasattr(block, attr) and isinstance(getattr(block, attr), int):
                return getattr(block, attr)
        # fall back to the first LayerNorm / Linear feature size we can find
        for m in block.modules():
            if isinstance(m, nn.LayerNorm):
                return m.normalized_shape[0]
            if isinstance(m, nn.Linear):
                return m.in_features
        raise ValueError("could not infer block embedding dim; pass a block exposing .n_embd/.d")

    def realized_fraction(self) -> float:
        """Fraction of tokens actually pushed through the block on the last forward."""
        return self.last_processed_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        n = b * t
        flat = x.reshape(n, d)

        # capacity=1.0 -> process everything; this is EXACT parity with self.block(x).
        if self.capacity >= 1.0:
            self.last_processed_fraction = 1.0
            return self.block(x)

        k = max(1, int(round(self.capacity * n)))
        k = min(k, n)

        scores = self.router(flat).squeeze(-1)        # (n,)
        # top-k tokens by router score get the block; the rest bypass via the residual.
        top_val, top_idx = torch.topk(scores, k, sorted=False)
        self.last_processed_fraction = k / n

        sel = flat.index_select(0, top_idx).view(1, k, d)   # (1,k,d) keeps block's (B,T,D) API
        block_out = self.block(sel).view(k, d)              # = sel + delta_sel (residual block)

        if self.use_router_weight:
            # weighted residual: scale the block's *delta* by the router's sigmoid weight so the
            # router receives gradient. out = sel + w * (block_out - sel) = sel + w*delta.
            w = torch.sigmoid(top_val).unsqueeze(-1)        # (k,1)
            new_sel = sel.view(k, d) + w * (block_out - sel.view(k, d))
        else:
            new_sel = block_out

        # start from the pure residual (every token passed through unchanged) ...
        out = flat.clone()
        # ... then overwrite the selected rows with the block's (weighted) output.
        out = out.index_copy(0, top_idx, new_sel)
        return out.view(b, t, d)

    def block_flop_fraction(self) -> float:
        """The fraction of the unwrapped block's FLOPs this wrapper spends (== capacity).

        The router adds a tiny d-per-token term on top, negligible vs the block's ~d^2.
        """
        return self.capacity
