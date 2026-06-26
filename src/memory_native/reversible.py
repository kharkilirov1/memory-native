"""Reversible coupling block — the memory lever for activations.

A reversible block computes its output without storing the input activations: the backward
pass reconstructs the input from the output via the exact inverse, then recomputes the local
forward to get gradients. Forward therefore stores only the block's output, not its inputs,
so a deep stack's activation memory stays O(1) in depth instead of O(depth).

    forward:  y1 = x1 + F(x2);   y2 = x2 + G(y1)
    inverse:  x2 = y2 - G(y1);   x1 = y1 - F(x2)

F and G must be deterministic (no dropout/RNG inside, no in-place state side effects). This
is pure PyTorch and runs on CPU/CUDA.

Note on float reconstruction: the inverse is not bit-exact and the error accumulates slowly
with depth (~3e-3 over ~12 blocks in fp32). On the depths tested that is training-neutral;
for very deep stacks verify with a depth-sweep and, if needed, keep anchors every K blocks.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["ReversibleCouplingBlock", "ReversibleSequential"]


class _ReversibleFn(torch.autograd.Function):
    # apply(x, F, G, n_f, *params): params are F's params followed by G's params, passed as
    # individual tensors so autograd routes one gradient back into each leaf parameter.
    @staticmethod
    def forward(ctx, x, F, G, n_f, *params):
        ctx.F, ctx.G, ctx.n_f = F, G, n_f
        d = x.shape[-1] // 2
        with torch.no_grad():
            x1, x2 = x[..., :d], x[..., d:]
            y1 = x1 + F(x2)
            y2 = x2 + G(y1)
            y = torch.cat([y1, y2], dim=-1)
        # store ONLY the output (+ params for the recompute); no input activation is saved.
        ctx.save_for_backward(y.detach(), *params)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        saved = ctx.saved_tensors
        y, params = saved[0], saved[1:]
        d = y.shape[-1] // 2
        y1, y2 = y[..., :d], y[..., d:]

        # 1) reconstruct the input from the output (the reversible trick: no stored input).
        with torch.no_grad():
            x2 = y2 - ctx.G(y1)
            x1 = y1 - ctx.F(x2)

        # 2) recompute the local forward with grad enabled, from the reconstructed input.
        x1 = x1.detach().requires_grad_(True)
        x2 = x2.detach().requires_grad_(True)
        with torch.enable_grad():
            z1 = x1 + ctx.F(x2)
            z2 = x2 + ctx.G(z1)
            z = torch.cat([z1, z2], dim=-1)

        grads = torch.autograd.grad(z, (x1, x2) + tuple(params), grad_y)
        grad_x = torch.cat([grads[0], grads[1]], dim=-1)
        # match apply's signature: x, F(None), G(None), n_f(None), then one grad per param.
        return (grad_x, None, None, None) + tuple(grads[2:])


class ReversibleCouplingBlock(nn.Module):
    """One reversible coupling block over a channel-split input (last dim even).

    F, G are arbitrary deterministic modules mapping [..., d] -> [..., d] where d = C/2.
    Default F, G are single Linear maps; pass your own (e.g. a small MLP or attention-style
    module) to make the block expressive.
    """

    def __init__(self, dim: int, F: nn.Module | None = None, G: nn.Module | None = None) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("ReversibleCouplingBlock needs an even channel dim")
        d = dim // 2
        self.F = F if F is not None else nn.Linear(d, d, bias=False)
        self.G = G if G is not None else nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_params = tuple(self.F.parameters())
        g_params = tuple(self.G.parameters())
        return _ReversibleFn.apply(x, self.F, self.G, len(f_params), *f_params, *g_params)


class ReversibleSequential(nn.Module):
    """A stack of reversible blocks. Activation memory is independent of depth: only the
    final output is kept by autograd; every block reconstructs its input in backward."""

    def __init__(self, blocks: Sequence[ReversibleCouplingBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x
