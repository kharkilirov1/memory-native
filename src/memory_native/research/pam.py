"""Piecewise-affine multiplication (PAM), a small PyTorch research reference.

Implements Sections 2.2 and 2.5 of Kosson and Jaggi (2023),
``Multiplication-Free Transformer Training via Piecewise Affine Operations``
https://arxiv.org/abs/2305.17190 . Only the linear-layer products are changed:
accumulation, bias, attention, normalization, losses and optimizers remain ordinary
PyTorch operations. ``frexp``/``ldexp`` are a numerical reference, not a custom
hardware implementation or evidence of an execution-time/energy improvement.

Inputs must be finite float32/float64. Subnormals are normalized rather than
flushed to zero. Underflow uses the dtype's normal rounding; nonfinite inputs,
outputs or requested gradients raise ValueError. Exact gradients at sign/octave
boundaries use the branch selected in forward (carry when mantissas sum to >=1).
At a zero operand, the exact mode defines BOTH operand gradients to be zero;
there is no unique limiting slope at zero in general. The surrogate mode uses
the paper's PAM(other_operand, upstream_gradient) rule, including at zero.
"""
from __future__ import annotations

import math
import numbers

import torch
from torch import nn

__all__ = ["pam", "PAMLinear"]

_DTYPES = (torch.float32, torch.float64)


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"PAM requires finite {name}; NaN/Inf and overflow are unsupported")


def _validate_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dtype not in _DTYPES or b.dtype not in _DTYPES:
        raise TypeError("PAM supports float32 and float64 tensors only")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("PAM operands must have the same dtype and device")
    _finite(a, "first operand")
    _finite(b, "second operand")


def _parts(value: torch.Tensor):
    fraction, exponent = torch.frexp(value.abs())
    # frexp normalizes subnormal inputs too. At zero the dummy mantissa is 0;
    # an explicit zero mask handles its value and derivative below.
    mantissa = torch.where(value == 0, 0.0, fraction + fraction - 1.0)
    return mantissa, exponent - 1, torch.signbit(value), value != 0


def _product_parts(a_parts, b_parts):
    ma, ea, negative_a, nonzero_a = a_parts
    mb, eb, negative_b, nonzero_b = b_parts
    total = ma + mb
    carry = (total >= 1).to(ea.dtype)
    magnitude = torch.ldexp(1.0 + (total - carry), ea + eb + carry)
    signed = torch.where(negative_a ^ negative_b, -magnitude, magnitude)
    return torch.where(nonzero_a & nonzero_b, signed, 0.0), carry


def _product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _product_parts(_parts(a), _parts(b))[0]


def _exact_derivative(delta, exponent, negative, nonzero):
    # Some PyTorch versions allocate ldexp's result with the first operand's
    # pre-broadcast shape, then warn while resizing it. Expanded views avoid
    # that deprecated path without materializing another product-size tensor.
    delta, exponent = torch.broadcast_tensors(delta, exponent)
    derivative = torch.ldexp(delta, exponent)
    derivative = torch.where(negative, -derivative, derivative)
    return torch.where(nonzero, derivative, 0.0)


class _PAMMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        _validate_pair(a, b)
        result = _product(a, b)
        _finite(result, "result")
        ctx.save_for_backward(a, b)
        return result

    @staticmethod
    def backward(ctx, delta):
        _finite(delta, "upstream gradient")
        a, b = ctx.saved_tensors
        pa, pb = _parts(a), _parts(b)
        carry = ((pa[0] + pb[0]) >= 1).to(pa[1].dtype)
        nonzero = pa[3] & pb[3]
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = _exact_derivative(delta, pb[1] + carry, pb[2], nonzero)
            grad_a = grad_a.sum_to_size(a.shape)
            _finite(grad_a, "first-operand gradient")
        if ctx.needs_input_grad[1]:
            grad_b = _exact_derivative(delta, pa[1] + carry, pa[2], nonzero)
            grad_b = grad_b.sum_to_size(b.shape)
            _finite(grad_b, "second-operand gradient")
        return grad_a, grad_b


def pam(a, b) -> torch.Tensor:
    """Broadcasting scalar PAM with the exact, piecewise-constant derivative.

    Python real scalars are accepted and use the other tensor's dtype/device,
    or the default floating dtype if both operands are scalars. Tensor operands
    must be float32/float64; differing floating dtypes are promoted. Unlike a
    stochastic approximation, PAM is deterministic and generally underestimates
    the magnitude of a multiplication by up to 1/9 in real arithmetic.
    """
    if not torch.is_tensor(a):
        if not isinstance(a, numbers.Real):
            raise TypeError("PAM operands must be tensors or real scalars")
        a = torch.as_tensor(a, dtype=b.dtype if torch.is_tensor(b) else torch.get_default_dtype(),
                            device=b.device if torch.is_tensor(b) else None)
    if not torch.is_tensor(b):
        if not isinstance(b, numbers.Real):
            raise TypeError("PAM operands must be tensors or real scalars")
        b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    if a.dtype not in _DTYPES or b.dtype not in _DTYPES:
        raise TypeError("PAM supports float32 and float64 tensors only")
    dtype = torch.promote_types(a.dtype, b.dtype)
    return _PAMMultiply.apply(a.to(dtype=dtype), b.to(dtype=dtype))


class _PAMLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, backward_mode, chunk_size):
        _validate_pair(x, weight)
        if bias is not None:
            _validate_pair(weight, bias)
        flat = x.reshape(-1, weight.shape[1])
        xp, wp = _parts(flat), _parts(weight)
        output = x.new_zeros((flat.shape[0], weight.shape[0]))
        # Only [tokens, out_features, chunk_size] product tiles are live, never
        # the full [tokens, out_features, in_features] product for large fan-in.
        for lo in range(0, weight.shape[1], chunk_size):
            hi = min(lo + chunk_size, weight.shape[1])
            a = tuple(part[:, None, lo:hi] for part in xp)
            b = tuple(part[None, :, lo:hi] for part in wp)
            products, _ = _product_parts(a, b)
            output.add_(products.sum(dim=-1))
        if bias is not None:
            output.add_(bias)
        _finite(output, "linear output")
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.backward_mode, ctx.chunk_size = backward_mode, chunk_size
        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, delta):
        _finite(delta, "upstream gradient")
        x, weight = ctx.saved_tensors
        flat = x.reshape(-1, weight.shape[1])
        delta = delta.reshape(-1, weight.shape[0])
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        grad_x = torch.zeros_like(flat) if need_x else None
        grad_w = torch.zeros_like(weight) if need_w else None
        if need_x or need_w:
            xp, wp = _parts(flat), _parts(weight)
            dp = _parts(delta[:, :, None]) if ctx.backward_mode == "surrogate" else None
            for lo in range(0, weight.shape[1], ctx.chunk_size):
                hi = min(lo + ctx.chunk_size, weight.shape[1])
                a = tuple(part[:, None, lo:hi] for part in xp)
                b = tuple(part[None, :, lo:hi] for part in wp)
                if ctx.backward_mode == "exact":
                    carry = ((a[0] + b[0]) >= 1).to(a[1].dtype)
                    nonzero = a[3] & b[3]
                    if need_x:
                        contribution = _exact_derivative(delta[:, :, None], b[1] + carry,
                                                         b[2], nonzero)
                        grad_x[:, lo:hi] = contribution.sum(dim=1)
                    if need_w:
                        contribution = _exact_derivative(delta[:, :, None], a[1] + carry,
                                                         a[2], nonzero)
                        grad_w[:, lo:hi] = contribution.sum(dim=0)
                else:
                    if need_x:
                        grad_x[:, lo:hi] = _product_parts(b, dp)[0].sum(dim=1)
                    if need_w:
                        grad_w[:, lo:hi] = _product_parts(a, dp)[0].sum(dim=0)
        grad_b = delta.sum(dim=0) if need_b and ctx.has_bias else None
        if grad_x is not None:
            _finite(grad_x, "input gradient")
            grad_x = grad_x.reshape_as(x)
        if grad_w is not None:
            _finite(grad_w, "weight gradient")
        if grad_b is not None:
            _finite(grad_b, "bias gradient")
        return grad_x, grad_w, grad_b, None, None


class PAMLinear(nn.Module):
    """A trainable linear layer using PAM products and ordinary accumulation.

    ``backward='exact'`` uses the derivative of the PAM forward; ``'surrogate'``
    uses PAM in place of multiplication in the ordinary linear backward. Weight
    layout and bias semantics match nn.Linear. Chunking changes reduction order,
    so float32 results across chunk sizes agree to rounding tolerance, not bits.
    This CPU-friendly reference is expected to be slower than optimized GEMM.
    """
    def __init__(self, in_features: int, out_features: int, *, backward: str = "exact",
                 init_gain: float = 1.0, bias: bool = False, chunk_size: int = 16):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features),
                            ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if backward not in ("exact", "surrogate"):
            raise ValueError("backward must be 'exact' or 'surrogate'")
        if not isinstance(init_gain, numbers.Real) or not math.isfinite(init_gain) or init_gain < 0:
            raise ValueError("init_gain must be finite and nonnegative")
        self.in_features, self.out_features = in_features, out_features
        self.backward_mode, self.chunk_size = backward, chunk_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("_pam_config", torch.tensor(
            [in_features, out_features, int(backward == "surrogate")], dtype=torch.int64))
        nn.init.normal_(self.weight, std=float(init_gain) / math.sqrt(in_features))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A weight-only reload must not silently change the learning rule. Check
        # structural metadata before super() can mutate this layer's parameters.
        # Chunk size is not structural: it changes reduction rounding only.
        config = state_dict.get(prefix + "_pam_config")
        expected = torch.tensor([self.in_features, self.out_features,
                                 int(self.backward_mode == "surrogate")], dtype=torch.int64)
        if (not torch.is_tensor(config) or config.dtype != torch.int64
                or tuple(config.shape) != (3,) or not torch.equal(config.detach().cpu(), expected)):
            raise RuntimeError(
                "PAM checkpoint configuration is missing or differs from this layer's "
                "dimensions/backward mode; copy weights explicitly for a deliberate ablation")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"PAMLinear expects inputs with last dimension {self.in_features}")
        return _PAMLinear.apply(x, self.weight, self.bias, self.backward_mode, self.chunk_size)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, backward={self.backward_mode!r}, "
                f"chunk_size={self.chunk_size}")
