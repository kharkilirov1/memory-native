"""Experimental BLAST operators executed directly from their learned factors.

For input block j and output block i, W_ij = U_i diag(s_ij) V_j.T.
The implementation projects each input block once, mixes the blocks separately
for each rank coordinate, then expands each output block once. It never builds
the full dense W or its dense weight gradient.
"""
from __future__ import annotations

import math
import operator

import torch
from torch import nn

from ..counter import RMSCounterLinear

__all__ = ["BLASTLinear", "CounterBLASTLinear", "LowRankLinear"]


def _positive_integer(name, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _blocks(width: int, block_size: int) -> tuple[int, ...]:
    return tuple(min(block_size, width - offset) for offset in range(0, width, block_size))


class _BLASTTopology(nn.Module):
    def __init__(self, in_features, out_features, block_size, rank, init_gain,
                 bias, *, counter_C=0):
        super().__init__()
        self.in_features = _positive_integer("in_features", in_features)
        self.out_features = _positive_integer("out_features", out_features)
        self.block_size = _positive_integer("block_size", block_size)
        self.rank = _positive_integer("rank", rank)
        self.init_gain = float(init_gain)
        if not math.isfinite(self.init_gain) or self.init_gain <= 0:
            raise ValueError("init_gain must be positive and finite")
        self.input_block_sizes = _blocks(self.in_features, self.block_size)
        self.output_block_sizes = _blocks(self.out_features, self.block_size)
        self.n_input_blocks = len(self.input_block_sizes)
        self.n_output_blocks = len(self.output_block_sizes)
        # Block size and C are mathematical state, even when buffer shapes happen
        # to match. Refuse a differently interpreted checkpoint before loading it.
        self.register_buffer("_structure", torch.tensor([
            self.in_features, self.out_features, self.block_size, self.rank, counter_C,
        ], dtype=torch.int64))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

    @property
    def coefficient_count(self) -> int:
        """Matrix-factor coefficients (optional bias is counted separately)."""
        return self.rank * (self.in_features + self.out_features
                            + self.n_input_blocks * self.n_output_blocks)

    @property
    def rank_bound(self) -> int:
        """Maximum possible global rank after the block projections/expansions."""
        return min(sum(min(width, self.rank) for width in self.input_block_sizes),
                   sum(min(width, self.rank) for width in self.output_block_sizes))

    def operation_counts(self, batch_tokens: int) -> dict[str, int]:
        """Arithmetic for all factor matmuls, including input and factor gradients.

        Counts are MACs, not measured speed. Forward costs B*P; backward costs
        2*B*P when both input and coefficient derivatives are computed. They omit
        bias adds, counter decode/update arithmetic, memory moves and allocation.
        """
        batch_tokens = _positive_integer("batch_tokens", batch_tokens)
        forward = batch_tokens * self.coefficient_count
        return {"forward_macs": forward, "backward_macs": 2 * forward}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"expected input with last dimension {self.in_features}")
        pieces = x.split(self.input_block_sizes, dim=-1)
        z = torch.stack([factor(piece) for factor, piece in zip(self.V, pieces)], dim=-2)
        u = torch.stack([factor(z[..., :, k]) for k, factor in enumerate(self.S)], dim=-1)
        result = torch.cat([factor(u[..., i, :]) for i, factor in enumerate(self.U)], dim=-1)
        return result if self.bias is None else result + self.bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "_structure")
        if saved is not None and (
                not isinstance(saved, torch.Tensor)
                or saved.shape != self._structure.shape
                or not torch.equal(saved.detach().cpu(), self._structure.detach().cpu())):
            raise RuntimeError("BLAST checkpoint topology or counter C mismatch")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class BLASTLinear(_BLASTTopology):
    """Floating-point BLAST factors with ordinary autograd/optimizer support.

    For independent zero-mean unit-variance input coordinates, initialization
    has E[Var(y_o)] = init_gain**2. V_j has weight variance 1/actual_block_width,
    S_k has variance init_gain**2/n_input_blocks, and U_i has variance 1/rank.
    These match the counter topology's initialization moments, including edge
    blocks. counter_lr/lr_scale/C are accepted for a shared experiment interface
    and have no effect on this floating-point model.
    """
    def __init__(self, in_features: int, out_features: int, *, block_size: int = 16,
                 rank: int = 8, init_gain: float = 1., bias: bool = False,
                 counter_lr: float = .003, lr_scale: float = 2e-4, C: int = 8):
        super().__init__(in_features, out_features, block_size, rank, init_gain, bias)
        self.V = nn.ModuleList(nn.Linear(width, self.rank, bias=False)
                               for width in self.input_block_sizes)
        self.S = nn.ModuleList(nn.Linear(self.n_input_blocks, self.n_output_blocks,
                                        bias=False) for _ in range(self.rank))
        self.U = nn.ModuleList(nn.Linear(self.rank, width, bias=False)
                               for width in self.output_block_sizes)
        with torch.no_grad():
            for factor in self.V:
                nn.init.normal_(factor.weight, std=1 / math.sqrt(factor.in_features))
            for factor in self.S:
                nn.init.normal_(factor.weight, std=self.init_gain / math.sqrt(self.n_input_blocks))
            for factor in self.U:
                nn.init.normal_(factor.weight, std=1 / math.sqrt(self.rank))


class CounterBLASTLinear(_BLASTTopology):
    """BLAST with all U/V/S factors trained by existing RMS counter synapses.

    These reference factors store uint8 synaptic codes plus per-row fp32 scale
    and RMS state; they are not a six-bit packed implementation. There are no
    floating master factors or Adam moments. Existing backward computes each
    factor's input derivative before updating that factor, so the complete
    chain's dinput uses the forward-time weights. Every counter sublayer runs
    exactly once per forward, preserving its eager-only mutation guard.
    """
    def __init__(self, in_features: int, out_features: int, *, block_size: int = 16,
                 rank: int = 8, init_gain: float = 1., bias: bool = False,
                 counter_lr: float = .003, lr_scale: float = 2e-4, C: int = 8):
        C = _positive_integer("C", C)
        if 3 * (2 * C - 1) > 256:
            raise ValueError("C is too large for uint8 counter state")
        for name, value in (("counter_lr", counter_lr), ("lr_scale", lr_scale)):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        super().__init__(in_features, out_features, block_size, rank, init_gain,
                         bias, counter_C=C)
        self.C = C

        def factor(fin, fout, gain):
            return RMSCounterLinear(fin, fout, C=C, lr=float(counter_lr),
                                    lr_scale=float(lr_scale), init_gain=gain,
                                    local_grad_clip=0.0)

        self.V = nn.ModuleList(factor(width, self.rank, 1.)
                               for width in self.input_block_sizes)
        self.S = nn.ModuleList(factor(self.n_input_blocks, self.n_output_blocks,
                                     self.init_gain) for _ in range(self.rank))
        self.U = nn.ModuleList(factor(self.rank, width, 1.)
                               for width in self.output_block_sizes)


class LowRankLinear(nn.Module):
    """Floating low-rank *replacement* of a linear operator for capacity controls.

    This is U(Vx), without a frozen dense base or LoRA residual. A rank matched
    to BLAST's global rank bound and a rank matched to its coefficient budget
    are different controls and should both be labeled explicitly.
    """
    def __init__(self, in_features: int, out_features: int, *, rank: int = 8,
                 init_gain: float = 1., bias: bool = False):
        super().__init__()
        self.in_features = _positive_integer("in_features", in_features)
        self.out_features = _positive_integer("out_features", out_features)
        self.rank = _positive_integer("rank", rank)
        self.init_gain = float(init_gain)
        if not math.isfinite(self.init_gain) or self.init_gain <= 0:
            raise ValueError("init_gain must be positive and finite")
        self.V = nn.Linear(self.in_features, self.rank, bias=False)
        self.U = nn.Linear(self.rank, self.out_features, bias=bias)
        with torch.no_grad():
            nn.init.normal_(self.V.weight, std=1 / math.sqrt(self.in_features))
            nn.init.normal_(self.U.weight, std=self.init_gain / math.sqrt(self.rank))
            if self.U.bias is not None:
                self.U.bias.zero_()

    @property
    def coefficient_count(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    @property
    def rank_bound(self) -> int:
        return min(self.rank, self.in_features, self.out_features)

    def operation_counts(self, batch_tokens: int) -> dict[str, int]:
        batch_tokens = _positive_integer("batch_tokens", batch_tokens)
        forward = batch_tokens * self.coefficient_count
        return {"forward_macs": forward, "backward_macs": 2 * forward}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.V(x))
