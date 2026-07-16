"""Reversible coupling blocks on MLX — the activations lever.

    forward:  y1 = x1 + F(x2);   y2 = x2 + G(y1)
    inverse:  x2 = y2 - G(y1);   x1 = y1 - F(x2)

`ReversibleSequence` is the whole-chain form (the O(1)-in-depth ideal): ONE custom
function over the entire stack whose VJP walks the chain backwards, reconstructing each
block's input from its output and recomputing locally with `mx.vjp`. Only the chain input,
the final output and the block parameters are referenced by the step graph — no
per-block activation survives to backward, which is what bounds activation memory at
depth. Inner self-updating counter layers fire exactly once per block during that walk
(their custom VJP runs inside the local `mx.vjp`), same contract as the torch port.

F and G must be deterministic (no dropout/RNG inside). `anchor_every=A > 0` switches to
anchored/checkpoint mode: store the activation every A blocks and recompute each chunk
forward from its anchor instead of inverting — exact reconstruction, O(L/A + A) memory,
one forward per block cheaper than the inverse walk.
"""
from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

__all__ = ["ReversibleCouplingBlock", "ReversibleSequential", "ReversibleSequence"]


class ReversibleCouplingBlock(nn.Module):
    """One reversible coupling block over a channel-split input (last dim even).

    F, G are arbitrary deterministic modules mapping [..., d] -> [..., d] where d = dim/2.
    Defaults are single Linear maps without bias; pass your own (e.g. counter layers or a
    small MLP) to make the block expressive."""

    def __init__(self, dim: int, F: nn.Module | None = None, G: nn.Module | None = None) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("ReversibleCouplingBlock needs an even channel dim")
        d = dim // 2
        self.F = F if F is not None else nn.Linear(d, d, bias=False)
        self.G = G if G is not None else nn.Linear(d, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 + self.F(x2)
        y2 = x2 + self.G(y1)
        return mx.concatenate([y1, y2], axis=-1)

    def inverse(self, y: mx.array) -> mx.array:
        d = y.shape[-1] // 2
        y1, y2 = y[..., :d], y[..., d:]
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(x2)
        return mx.concatenate([x1, x2], axis=-1)


class ReversibleSequential(nn.Module):
    """A stack of reversible blocks applied one at a time, differentiated the ordinary way
    (every block's input stays referenced by the step graph). Use `ReversibleSequence` for
    the O(1)-in-depth activation memory; this class is the plain reference stack."""

    def __init__(self, blocks: Sequence[ReversibleCouplingBlock]) -> None:
        super().__init__()
        self.layers = list(blocks)

    def __call__(self, x: mx.array) -> mx.array:
        for b in self.layers:
            x = b(x)
        return x


def _block_params(block: ReversibleCouplingBlock) -> list[tuple[str, mx.array]]:
    """Flat (key, array) list of a block's trainable parameters, deterministic order."""
    return tree_flatten(block.trainable_parameters())


def _with_params(block: ReversibleCouplingBlock, keys: list[str], arrays: Sequence[mx.array]) -> None:
    """Swap the given arrays into the block's trainable parameters (by flattened key)."""
    if keys:
        block.update(tree_unflatten(list(zip(keys, arrays))))


def _couple_fwd(block: ReversibleCouplingBlock, x: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 + block.F(x2)
    y2 = x2 + block.G(y1)
    return mx.concatenate([y1, y2], axis=-1)


class ReversibleSequence(nn.Module):
    """A reversible stack with O(1)-in-depth activation memory (default), or anchored
    checkpointing with `anchor_every=A > 0` (store every A-th input, recompute chunks
    forward — exact, no float-inverse error, O(L/A + A) memory).

    Note: the float inverse accumulates a little reconstruction error with depth
    (training-neutral at the depths tested in the torch port, ~1e-3 over ~12 blocks);
    unusually deep stacks may want anchors."""

    def __init__(self, blocks: Sequence[ReversibleCouplingBlock], anchor_every: int = 0) -> None:
        super().__init__()
        self.layers = list(blocks)
        self.anchor_every = int(anchor_every)

    def __call__(self, x: mx.array) -> mx.array:
        blocks = self.layers
        specs = [_block_params(b) for b in blocks]
        keys = [[k for k, _ in spec] for spec in specs]
        offsets, off = [], 0
        for spec in specs:
            offsets.append(off)
            off += len(spec)
        flat: list[mx.array] = [a for spec in specs for _, a in spec]
        anchor = self.anchor_every
        anchors_store: list[mx.array] = []  # filled by forward when anchor > 0

        @mx.custom_function
        def rev_chain(x0: mx.array, *params: mx.array) -> mx.array:
            anchors_store.clear()
            h = x0
            for i, b in enumerate(blocks):
                if anchor > 0 and i % anchor == 0:
                    anchors_store.append(h)
                h = _couple_fwd(b, h)
            return h

        @rev_chain.vjp
        def rev_chain_vjp(primals, cotangents, outputs):
            x0 = primals[0]
            params = list(primals[1:])
            y = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            gy = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
            grad_params: list[mx.array] = [mx.zeros_like(p) for p in params]

            def local_fn_for(i):
                block, bkeys = blocks[i], keys[i]

                def local(x1, x2, *bp):
                    _with_params(block, bkeys, bp)
                    z1 = x1 + block.F(x2)
                    z2 = x2 + block.G(z1)
                    return [z1, z2]

                return local

            if anchor > 0:
                # anchored/checkpoint mode: recompute each chunk forward from its stored
                # anchor with mx.vjp (no inverse pass, exact reconstruction).
                anchors = list(anchors_store)
                L = len(blocks)
                n_chunks = (L + anchor - 1) // anchor
                for cidx in range(n_chunks - 1, -1, -1):
                    start, end = cidx * anchor, min((cidx + 1) * anchor, L)
                    chunk_arrays: list[mx.array] = []
                    for i in range(start, end):
                        chunk_arrays.extend(params[offsets[i]:offsets[i] + len(keys[i])])

                    def chunk_fn(xin, *bp, _start=start, _end=end):
                        pos = 0
                        h2 = xin
                        for i in range(_start, _end):
                            n = len(keys[i])
                            _with_params(blocks[i], keys[i], bp[pos:pos + n])
                            pos += n
                            h2 = _couple_fwd(blocks[i], h2)
                        return h2

                    _, grads = mx.vjp(chunk_fn, [anchors[cidx]] + chunk_arrays, [gy])
                    gy = grads[0]
                    pos = 1
                    for i in range(start, end):
                        n = len(keys[i])
                        for k in range(n):
                            grad_params[offsets[i] + k] = grads[pos + k]
                        pos += n
                return (gy, *grad_params)

            # pure reversible mode: walk backwards, inverting each block from its output.
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                d = y.shape[-1] // 2
                y1, y2 = y[..., :d], y[..., d:]
                # 1) reconstruct this block's input from its output (no VJP -> no update).
                x2 = mx.stop_gradient(y2 - block.G(y1))
                x1 = mx.stop_gradient(y1 - block.F(x2))
                # 2) recompute the local forward under mx.vjp: param grads + input
                #    cotangent; inner counter layers self-update exactly once, here.
                bparams = params[offsets[i]:offsets[i] + len(keys[i])]
                gy1, gy2 = gy[..., :d], gy[..., d:]
                _, grads = mx.vjp(local_fn_for(i), [x1, x2] + list(bparams), [gy1, gy2])
                gy = mx.concatenate([grads[0], grads[1]], axis=-1)
                for k in range(len(keys[i])):
                    grad_params[offsets[i] + k] = grads[2 + k]
                y = mx.concatenate([x1, x2], axis=-1)  # earlier block's output = this input
            return (gy, *grad_params)

        return rev_chain(x, *flat)
