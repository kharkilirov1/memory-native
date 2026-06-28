"""Reversible coupling block — the memory lever for activations.

A reversible block computes its output without storing the input activations: the backward
pass reconstructs the input from the output via the exact inverse, then recomputes the local
forward to get gradients. Forward therefore stores only the block's output, not its inputs,
so a deep stack's activation memory stays O(1) in depth instead of O(depth).

    forward:  y1 = x1 + F(x2);   y2 = x2 + G(y1)
    inverse:  x2 = y2 - G(y1);   x1 = y1 - F(x2)

F and G must be deterministic (no dropout/RNG inside, no in-place state side effects). This
is pure PyTorch and runs on CPU/CUDA.

Note on float reconstruction: the inverse need not be bit-exact in principle. Empirically, for
these coupling maps the recomputed INPUT GRADIENTS match a stored-activation reference to ~0 across
a weight-scale sweep up to forward magnitudes ~1e9 (test_inverse_and_anchored_exact_across_weight
_scales) -- the reconstruction is exact in fp32 here, so the inverse path is training-neutral on the
regimes tested. Anchors (anchor_every=K) recompute-from-anchor instead of inverting; use them for
the activation-MEMORY tradeoff (and as a safety margin for unusually deep/ill-conditioned stacks).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["ReversibleCouplingBlock", "ReversibleSequential", "ReversibleSequence"]


class _ReversibleFn(torch.autograd.Function):
    # apply(x, F, G, tap, n_f, *params): params are F's params then G's params, passed as
    # individual tensors so autograd routes one gradient back into each leaf parameter. `tap`
    # is a scalar requiring grad so the output requires grad (and backward + any inner self-
    # updating counter layers run) even when the input and F/G carry no gradient.
    @staticmethod
    def forward(ctx, x, F, G, tap, n_f, *params):
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
        # match apply's signature: x, F(None), G(None), tap(None), n_f(None), then param grads.
        return (grad_x, None, None, None, None) + tuple(grads[2:])


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
        # tap forces the output to require grad so backward runs even when neither the input
        # nor F/G need a gradient (e.g. F/G are self-updating counter layers with no params).
        tap = (torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
               if torch.is_grad_enabled() else x.new_zeros(()))
        return _ReversibleFn.apply(x, self.F, self.G, tap, len(f_params), *f_params, *g_params)


class ReversibleSequential(nn.Module):
    """A stack of reversible blocks applied one at a time. Each block is its own autograd
    Function that stores ITS OWN output, so activation memory is O(depth) with a *small*
    constant (one [N,dim] per block). For the O(1)-in-depth ideal use ReversibleSequence."""

    def __init__(self, blocks: Sequence[ReversibleCouplingBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def _couple_fwd(block: "ReversibleCouplingBlock", x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 + block.F(x2)
    y2 = x2 + block.G(y1)
    return torch.cat([y1, y2], dim=-1)


def _couple_bwd(block: "ReversibleCouplingBlock", y: torch.Tensor, grad_y: torch.Tensor):
    d = y.shape[-1] // 2
    y1, y2 = y[..., :d], y[..., d:]
    with torch.no_grad():                       # reconstruct this block's input from its output
        x2 = y2 - block.G(y1)
        x1 = y1 - block.F(x2)
    x1 = x1.detach().requires_grad_(True)
    x2 = x2.detach().requires_grad_(True)
    with torch.enable_grad():                   # recompute to get grads (fires inner counters)
        z1 = x1 + block.F(x2)
        z2 = x2 + block.G(z1)
        z = torch.cat([z1, z2], dim=-1)
    params = tuple(block.F.parameters()) + tuple(block.G.parameters())
    grads = torch.autograd.grad(z, (x1, x2) + params, grad_y)
    grad_x = torch.cat([grads[0], grads[1]], dim=-1)
    x = torch.cat([x1.detach(), x2.detach()], dim=-1)
    return x, grad_x, grads[2:]


class _ReversibleSequenceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, blocks, meta, tap, *all_params):
        ctx.blocks = blocks
        ctx.meta = meta
        with torch.no_grad():                   # run the whole chain; keep NO intermediates
            for b in blocks:
                x = _couple_fwd(b, x)
        ctx.save_for_backward(x.detach(), *all_params)   # store ONLY the final output (+params)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        y = ctx.saved_tensors[0]
        n_params = len(ctx.saved_tensors) - 1
        grad_params = [None] * n_params
        offsets, off = [], 0
        for (nf, ng) in ctx.meta:
            offsets.append(off)
            off += nf + ng
        grad_y = grad_out
        for i in range(len(ctx.blocks) - 1, -1, -1):     # walk the chain backwards
            x, grad_x, gparams = _couple_bwd(ctx.blocks[i], y, grad_y)
            base = offsets[i]
            for k, g in enumerate(gparams):
                grad_params[base + k] = g
            y, grad_y = x, grad_x                          # earlier block's output = this input
        return (grad_y, None, None, None) + tuple(grad_params)


class _AnchoredReversibleSequenceFn(torch.autograd.Function):
    # Speed/memory knob (acceleration memo M7): store the activation at every A-th block (an
    # "anchor") and, in backward, recompute each chunk forward FROM its anchor with grad
    # (checkpoint-style) instead of reconstructing via the inverse. That drops the inverse pass
    # (~1 forward/block faster) and is exact (no float-inverse error), at O(L/A + A) activation
    # memory. A=L is one whole-chain checkpoint; A=1 stores every block. Counter contract is the
    # same as the reversible path: a no-grad forward walk, then an enable_grad recompute that
    # fires each inner counter exactly once.
    @staticmethod
    def forward(ctx, x, blocks, meta, anchor_every, tap, *all_params):
        ctx.blocks, ctx.meta, ctx.A = blocks, meta, anchor_every
        anchors = []
        with torch.no_grad():
            xi = x
            for i, b in enumerate(blocks):
                if i % anchor_every == 0:
                    anchors.append(xi)
                xi = _couple_fwd(b, xi)
        ctx.n_anchors = len(anchors)
        ctx.save_for_backward(*[a.detach() for a in anchors], *all_params)
        return xi

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        anchors, all_params = saved[:ctx.n_anchors], saved[ctx.n_anchors:]
        grad_params = [None] * len(all_params)
        offsets, off = [], 0
        for (nf, ng) in ctx.meta:
            offsets.append(off)
            off += nf + ng
        blocks, A, L = ctx.blocks, ctx.A, len(ctx.blocks)
        grad_y = grad_out
        for c in range((L + A - 1) // A - 1, -1, -1):       # chunks, last to first
            start, end = c * A, min((c + 1) * A, L)
            xin = anchors[c].detach().requires_grad_(True)
            with torch.enable_grad():                        # recompute the chunk forward (no inverse)
                h = xin
                for i in range(start, end):
                    h = _couple_fwd(blocks[i], h)
            chunk_params = []
            for i in range(start, end):
                chunk_params.extend(tuple(blocks[i].F.parameters()))
                chunk_params.extend(tuple(blocks[i].G.parameters()))
            grads = torch.autograd.grad(h, (xin,) + tuple(chunk_params), grad_y)
            grad_y = grads[0]
            k = 1
            for i in range(start, end):
                base, (nf, ng) = offsets[i], ctx.meta[i]
                for j in range(nf + ng):
                    grad_params[base + j] = grads[k]
                    k += 1
        return (grad_y, None, None, None, None) + tuple(grad_params)


class ReversibleSequence(nn.Module):
    """A reversible stack with **O(1)-in-depth** activation memory (default): the whole chain is a
    single autograd Function that stores ONLY the final output. Backward walks the chain in
    reverse, reconstructing each block's input from its output and recomputing locally (classic
    RevNet). Inner self-updating counter layers fire once per block during that walk.

    `anchor_every=A > 0` switches to anchored mode (acceleration memo M7): store the activation
    every A blocks and recompute each chunk forward from its anchor instead of inverting. This is
    the speed/memory knob — O(1) reversible is the minimum-memory extreme, not the fastest point;
    anchors spend O(L/A + A) memory to skip the inverse pass (and avoid float-inverse error).

    Note: the float inverse (anchor_every=0) accumulates a little reconstruction error with depth
    (~3e-3 over ~12 blocks); training-neutral at modest depth, but very deep stacks may want
    anchors, which reconstruct exactly."""

    def __init__(self, blocks: Sequence[ReversibleCouplingBlock], anchor_every: int = 0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.anchor_every = int(anchor_every)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        all_params, meta = [], []
        for b in self.blocks:
            fp = tuple(b.F.parameters())
            gp = tuple(b.G.parameters())
            meta.append((len(fp), len(gp)))
            all_params.extend(fp)
            all_params.extend(gp)
        tap = (torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
               if torch.is_grad_enabled() else x.new_zeros(()))
        if self.anchor_every > 0:
            return _AnchoredReversibleSequenceFn.apply(
                x, tuple(self.blocks), tuple(meta), self.anchor_every, tap, *all_params)
        return _ReversibleSequenceFn.apply(x, tuple(self.blocks), tuple(meta), tap, *all_params)
