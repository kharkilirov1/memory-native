"""A bias-free linear layer whose weights train in MXFP4 with no optimizer state.

Forward uses the E2M1 values only (strict readout); the 2-bit residual is invisible to
the forward pass and exists only to carry quantization error between steps. The update
runs inside the autograd Function's backward, so no gradient buffer or optimizer is kept.
"""
from __future__ import annotations

import torch
from torch import nn

from . import pack, update


class _MX6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "MX6Linear", token: torch.Tensor) -> torch.Tensor:
        weight = module.weight()
        ctx.save_for_backward(x, weight, module.codes, module.scale)
        ctx.module = module
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, codes_before, _ = ctx.saved_tensors
        module = ctx.module
        grad_x = grad_out @ weight
        grad_w = grad_out.t() @ x
        if not torch.equal(codes_before, module.codes):
            raise RuntimeError("weights changed between forward and backward")
        with torch.no_grad():
            module.apply_update_(grad_w)
        module._pending = False
        return grad_x, None, None


class MX6Linear(nn.Module):
    """MXFP4 values + 2-bit residual, one E8M0 scale per 32 input features."""

    def __init__(self, in_features: int, out_features: int, lr: float = 0.01,
                 rms_beta: float = 0.9, rms_eps: float = 1e-3, residual_mode: str = "rel",
                 feedback: bool = False, rounding: str = "sr", integrator: int = 0,
                 second_moment: str = "row", cautious: bool = False,
                 integrator_decay: float = 0.0, scale_headroom: float = 6.0,
                 scale_mode: str = "headroom", lr_mode: str = "abs",
                 compact_state: bool = False):
        super().__init__()
        blocks = pack.check_shape(in_features, out_features)
        self.in_features, self.out_features = in_features, out_features
        self.lr, self.rms_beta, self.rms_eps = lr, rms_beta, rms_eps
        self.residual_mode = residual_mode
        self.feedback = feedback
        self.rounding = rounding
        self.integrator = int(integrator)
        self.second_moment = second_moment
        self.cautious = cautious
        self.integrator_decay = integrator_decay
        self.scale_headroom = scale_headroom
        self.scale_mode = scale_mode
        self.lr_mode = lr_mode
        self.compact_state = bool(compact_state)
        if self.compact_state and self.integrator not in (0, 3):
            raise ValueError("compact_state supports integrator=0 (stateless) and integrator=3 "
                             "(signed 2-bit remainder); the code must fit in 4 bits")
        if self.compact_state and self.feedback:
            raise ValueError("compact_state does not support the separate 2-bit error-feedback plane")
        init = torch.empty(out_features, in_features).uniform_(
            -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5)
        codes, scale = pack.quantize(init, headroom=scale_headroom, mode=scale_mode)
        if self.compact_state:
            if codes.numel() % 4:
                raise ValueError("compact_state requires parameter count divisible by 4")
            self.register_buffer("_codes_packed", pack.pack_nibbles(codes))
            if self.integrator > 1:
                self.register_buffer("_acc_packed", pack.pack_2bit_signed(
                    torch.zeros(out_features, in_features, dtype=torch.int8)))
            else:
                self.register_buffer("_acc_packed", torch.empty(0, dtype=torch.uint8))
        else:
            self.register_buffer("_codes_dense", codes)
            if self.integrator > 1:
                self.register_buffer("_acc_dense", torch.zeros(out_features, in_features, dtype=torch.int8))
            else:
                self.register_buffer("_acc_dense", torch.empty(0, dtype=torch.int8))
        self.register_buffer("kappa", torch.zeros((out_features, in_features) if self.feedback else (0,), dtype=torch.uint8))
        self.register_buffer("scale", scale)
        if self.second_moment == "block":
            self.register_buffer("v_block", torch.full((out_features, blocks), 127, dtype=torch.uint8))
        elif self.second_moment == "col":
            self.register_buffer("v_row", torch.full((out_features, 1), 102, dtype=torch.uint8))
            self.register_buffer("v_col", torch.full((1, in_features), 102, dtype=torch.uint8))
        elif self.second_moment == "layer":
            self.register_buffer("v_layer", torch.full((1,), 102, dtype=torch.uint8))
        elif self.second_moment == "row":
            self.register_buffer("v_rms", torch.full((out_features, 1), rms_eps, dtype=torch.float32))
        else:
            raise ValueError(f"unknown second_moment {self.second_moment!r}")
        self._token = torch.zeros(0, requires_grad=True)
        self._pending = False

    @property
    def codes(self) -> torch.Tensor:
        if self.compact_state:
            return pack.unpack_nibbles(self._codes_packed, (self.out_features, self.in_features))
        return self._codes_dense

    @property
    def acc(self) -> torch.Tensor:
        if self.compact_state:
            if self._acc_packed.numel() == 0:
                return torch.empty(0, dtype=torch.int8)
            return pack.unpack_2bit_signed(self._acc_packed, (self.out_features, self.in_features))
        return self._acc_dense

    def _store_codes(self, codes: torch.Tensor) -> None:
        if self.compact_state:
            self._codes_packed.copy_(pack.pack_nibbles(codes))
        else:
            self._codes_dense.copy_(codes)

    def _store_acc(self, acc: torch.Tensor) -> None:
        if self.compact_state:
            if self._acc_packed.numel() == 0:
                return  # stateless integrator: no accumulator plane to write
            self._acc_packed.copy_(pack.pack_2bit_signed(acc))
        else:
            self._acc_dense.copy_(acc)

    def weight(self) -> torch.Tensor:
        """The visible weight (float); the residual is deliberately not included."""
        return pack.dequantize(self.codes, self.scale)

    def latent_weight(self) -> torch.Tensor:
        """Visible weight plus the hidden optimizer pressure, for analysis only."""
        s = pack.scale_floats(self.scale).repeat_interleave(pack.BLOCK, dim=1)
        d = update.delta(self.codes, self.residual_mode) * s
        if self.integrator > 1:
            C = (self.integrator + 1) // 2
            return self.weight() + self.acc.to(torch.float32) * d / C
        if not self.kappa.numel():
            return self.weight()
        kappa = update.KAPPA.to(s.device)[self.kappa.to(torch.long)]
        return self.weight() + kappa * d

    @classmethod
    def from_dense(cls, weight: torch.Tensor, **kwargs) -> "MX6Linear":
        layer = cls(weight.shape[1], weight.shape[0], **kwargs)
        with torch.no_grad():
            codes, scale = pack.quantize(
                weight.detach().float(),
                headroom=kwargs.get("scale_headroom", 6.0),
                mode=kwargs.get("scale_mode", "headroom"))
            layer._store_codes(codes)
            layer.scale.copy_(scale)
            layer.kappa.zero_()
        return layer

    def persistent_bytes(self) -> int:
        """Persistent tensor bytes, not an idealized bits/parameter estimate."""
        tensors = [self.scale]
        if self.compact_state:
            tensors.extend([self._codes_packed, self._acc_packed])
        else:
            tensors.extend([self._codes_dense, self._acc_dense])
        if self.kappa.numel():
            tensors.append(self.kappa)
        tensors = [t for t in tensors if t.numel() > 0]
        if self.second_moment == "row":
            tensors.append(self.v_rms)
        elif self.second_moment == "block":
            tensors.append(self.v_block)
        elif self.second_moment == "col":
            tensors.extend([self.v_row, self.v_col])
        elif self.second_moment == "layer":
            tensors.append(self.v_layer)
        return sum(t.numel() * t.element_size() for t in tensors)

    def pack_state(self) -> torch.Tensor:
        if self.integrator == 3:
            residual = (self.acc.to(torch.int16) + 1).to(torch.uint8)
            return pack.pack(self.codes, residual, self.scale)
        if self.kappa.numel():
            return pack.pack(self.codes, self.kappa, self.scale)
        if self.integrator <= 1:
            return pack.pack(self.codes, torch.zeros_like(self.codes), self.scale)
        raise NotImplementedError("packed checkpoint format is defined for residual mode and 3-state integrator")

    def load_state(self, blob: torch.Tensor) -> None:
        codes, residual, scale = pack.unpack(blob, self.out_features, self.in_features)
        self._store_codes(codes)
        if self.integrator == 3:
            self._store_acc(residual.to(torch.int16).sub(1).to(torch.int8))
        elif self.kappa.numel():
            self.kappa.copy_(residual if self.feedback else torch.zeros_like(residual))
        self.scale.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {x.shape[-1]}")
        shape = x.shape
        if torch.is_grad_enabled() and self.training:
            if self._pending:
                raise RuntimeError(
                    "MX6Linear was forwarded twice before backward; exactly one forward "
                    "per backward is required (wrap measurement passes in torch.no_grad() "
                    "or call .eval())")
            self._pending = True
            flat = x.reshape(-1, shape[-1])
            out = _MX6Function.apply(flat, self, self._token)
            return out.reshape(*shape[:-1], self.out_features)
        return x @ self.weight().t()

    def apply_update_(self, grad_w: torch.Tensor) -> None:
        """In-place counter update. Called from backward; no optimizer involved."""
        if grad_w.shape != self.codes.shape:
            raise ValueError(f"grad shape {tuple(grad_w.shape)} != state {tuple(self.codes.shape)}")
        if self.second_moment == "layer":
            self.v_layer, grad_hat = update.rms_update_layer(
                grad_w.detach().float(), self.v_layer, self.rms_beta, self.rms_eps)
        elif self.second_moment == "col":
            self.v_row, self.v_col, grad_hat = update.rms_update_col(
                grad_w.detach().float(), self.v_row, self.v_col, self.rms_beta, self.rms_eps)
        elif self.second_moment == "block":
            self.v_block, grad_hat = update.rms_update_block(
                grad_w.detach().float(), self.v_block, self.rms_beta, self.rms_eps,
                codec=update.encode_fp8, decoder=update.decode_fp8)
        else:
            self.v_rms, grad_hat = update.rms_update(
                grad_w.detach().float(), self.v_rms, self.rms_beta, self.rms_eps)
        if self.integrator > 1:
            codes, acc, scale = update.step_with_integrator(
                self.codes, self.acc, self.scale, grad_hat, self.lr, self.integrator,
                cautious=self.cautious, decay=self.integrator_decay,
                headroom=self.scale_headroom, scale_mode=self.scale_mode,
                lr_mode=self.lr_mode)
            self._store_codes(codes)
            self._store_acc(acc)
            self.scale.copy_(scale)
            return
        kappa_in = self.kappa if self.kappa.numel() else torch.zeros_like(self.codes)
        codes, kappa, scale = update.step(
            self.codes, kappa_in, self.scale, grad_hat, self.lr, self.residual_mode,
            rounding=self.rounding, feedback=self.feedback)
        self._store_codes(codes)
        if self.kappa.numel():
            self.kappa.copy_(kappa if self.feedback else torch.zeros_like(kappa))
        self.scale.copy_(scale)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"lr={self.lr}, residual={self.residual_mode}")


# ── block-shared accumulator layer (see docs/BLOCK_ACCUMULATOR.md) ─────────────

class _MX6BlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "MX6BlockLinear", token: torch.Tensor) -> torch.Tensor:
        weight = module.weight()
        ctx.save_for_backward(x, weight, module.codes)
        ctx.module = module
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, codes_before = ctx.saved_tensors
        module = ctx.module
        grad_x = grad_out @ weight
        grad_w = grad_out.t() @ x
        if not torch.equal(codes_before, module.codes):
            raise RuntimeError("weights changed between forward and backward")
        with torch.no_grad():
            module.apply_update_(grad_w)
        module._pending = False
        return grad_x, None, None


class MX6BlockLinear(nn.Module):
    """MXFP4 values + 11-bit block-shared accumulator. 4.50 bits/param total.

    The scale byte is split into 5-bit exponent + 3-bit acc-high.  A separate
    acc-low byte (1 per 32-param block) completes the 11-bit unsigned block
    accumulator.  The forward pass sees only the 5-bit exponent (standard MXFP4);
    the accumulator is invisible to inference.

    See ``docs/BLOCK_ACCUMULATOR.md`` for the full design.
    """

    def __init__(self, in_features: int, out_features: int, lr: float = 0.01,
                 rms_beta: float = 0.9, rms_eps: float = 1e-3,
                 C_block: int = 32, scale_headroom: float = 6.0):
        super().__init__()
        blocks = pack.check_shape(in_features, out_features)
        self.in_features, self.out_features = in_features, out_features
        self.lr, self.rms_beta, self.rms_eps = lr, rms_beta, rms_eps
        self.C_block = C_block
        self.scale_headroom = scale_headroom

        init = torch.empty(out_features, in_features).uniform_(
            -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5)
        codes, scale_byte = pack.quantize_split(init, headroom=scale_headroom)
        self.register_buffer("_codes", codes)
        self.register_buffer("_scale_byte", scale_byte)          # 5-bit exp | 3-bit acc-high
        self.register_buffer("_acc_low", torch.zeros(out_features, blocks, dtype=torch.uint8))
        self.register_buffer("v_rms", torch.full((out_features, 1), rms_eps, dtype=torch.float32))
        self._token = torch.zeros(0, requires_grad=True)
        self._pending = False

    @property
    def codes(self) -> torch.Tensor:
        return self._codes

    @property
    def scale_byte(self) -> torch.Tensor:
        return self._scale_byte

    @property
    def acc_low(self) -> torch.Tensor:
        return self._acc_low

    def weight(self) -> torch.Tensor:
        """Visible weight (float); accumulator is invisible to forward."""
        return pack.dequantize_split(self._codes, self._scale_byte)

    def latent_weight(self) -> torch.Tensor:
        """Visible weight + pending block pressure, for analysis only."""
        w = self.weight()
        s = pack.scale_floats_split(self._scale_byte).repeat_interleave(pack.BLOCK, dim=1)
        d = update.delta(self._codes, "rel") * s
        acc_high = pack.acc_high_get(self._scale_byte)
        acc = pack.block_acc_combine(acc_high, self._acc_low)
        # distribute block pressure equally (diagnostic approximation)
        pressure = acc.to(torch.float32) / self.C_block
        return w + pressure.repeat_interleave(pack.BLOCK, dim=1) * d / pack.BLOCK

    @classmethod
    def from_dense(cls, weight: torch.Tensor, **kwargs) -> "MX6BlockLinear":
        layer = cls(weight.shape[1], weight.shape[0], **kwargs)
        with torch.no_grad():
            codes, scale_byte = pack.quantize_split(
                weight.detach().float(),
                headroom=kwargs.get("scale_headroom", 6.0))
            layer._codes.copy_(codes)
            layer._scale_byte.copy_(scale_byte)
            layer._acc_low.zero_()
        return layer

    def persistent_bytes(self) -> int:
        """Conceptual packed bytes (what a fused kernel would use, not dense tensors)."""
        blocks = self.in_features // pack.BLOCK
        code_bytes = self.out_features * self.in_features // 2   # nibble-packed 4-bit
        block_bytes = self.out_features * blocks * 2              # scale+acc-high + acc-low
        rms_bytes = self.out_features * 4                          # per-row float32
        return code_bytes + block_bytes + rms_bytes

    def bits_per_param(self) -> float:
        return self.persistent_bytes() * 8.0 / (self.out_features * self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {x.shape[-1]}")
        shape = x.shape
        if torch.is_grad_enabled() and self.training:
            if self._pending:
                raise RuntimeError(
                    "MX6BlockLinear was forwarded twice before backward; exactly one "
                    "forward per backward is required")
            self._pending = True
            flat = x.reshape(-1, shape[-1])
            out = _MX6BlockFunction.apply(flat, self, self._token)
            return out.reshape(*shape[:-1], self.out_features)
        return x @ self.weight().t()

    def apply_update_(self, grad_w: torch.Tensor) -> None:
        """Block-accumulator update, called from backward."""
        self.v_rms, grad_hat = update.rms_update(
            grad_w.detach().float(), self.v_rms, self.rms_beta, self.rms_eps)
        new_codes, new_scale, new_acc_low = update.step_with_block_integrator(
            self._codes, self._scale_byte, self._acc_low,
            grad_hat, self.lr, C_block=self.C_block,
            headroom=self.scale_headroom)
        self._codes.copy_(new_codes)
        self._scale_byte.copy_(new_scale)
        self._acc_low.copy_(new_acc_low)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"lr={self.lr}, C_block={self.C_block}")


# ── packed (value, accumulator) layer — 5-bit and 6-bit formats ───────────────
# See mx6/packed_na.py for the format spec.

from .packed_na import PackedNA, PackedNA5, PackedNA6
from .packed_na import BLOCK as _NA_BLOCK
from .triton_kernel import HAS_TRITON, packed_matmul, packed_matmul_dx


class _MXNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, token):
        codes = module.codes
        ctx.save_for_backward(x, codes)
        ctx.module = module
        # Fused path decodes inside the GEMM, so no dense weight is materialised.
        ctx.fused = (x.is_cuda and HAS_TRITON and
                     (module.fused == "on" or
                      (module.fused == "auto" and x.shape[0] <= 8)))
        if ctx.fused:
            return packed_matmul(x, codes, module._scale,
                                 module.fmt.code_to_value.to(x.device))
        out = x @ module.weight().t()
        if module.weight_cache == "none":
            module._weight_cache = None  # re-decoded in backward; frees L x 32 bits/param
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, codes_before = ctx.saved_tensors
        module = ctx.module
        if not torch.equal(codes_before, module.codes):
            raise RuntimeError("weights changed between forward and backward")
        if ctx.fused:
            grad_x = packed_matmul_dx(grad_out, codes_before, module._scale,
                                      module.fmt.code_to_value.to(x.device))
        else:
            grad_x = grad_out @ module.weight()
        grad_w = grad_out.t() @ x
        with torch.no_grad():
            module.apply_update_(grad_w)
        module._pending = False
        return grad_x, None, None


class MXNLinear(nn.Module):
    """Packed (value, accumulator) linear layer — per-param momentum in one word.

    The value and accumulator are packed into a single code word (like the
    ternary counter on ``main``), so there is no separate accumulator buffer.
    The forward pass reads only the value; the accumulator is invisible to
    inference.

    Args:
        fmt: a ``PackedNA`` instance (``PackedNA5`` for 5-bit, ``PackedNA6``
            for 6-bit).
    """

    def __init__(self, in_features: int, out_features: int, fmt: PackedNA,
                 lr: float = 0.01, rms_beta: float = 0.9, rms_eps: float = 1e-3,
                 scale_headroom: float = 1.5, fused: str = "off",
                 weight_cache: str = "step", rescale_every: int = 0,
                 scale_mode: str = "headroom", clip_frac: float = 0.0):
        super().__init__()
        assert in_features % pack.BLOCK == 0, f"in_features must be multiple of {pack.BLOCK}"
        self.in_features, self.out_features = in_features, out_features
        self.lr, self.rms_beta, self.rms_eps = lr, rms_beta, rms_eps
        self.fmt = fmt
        self.scale_headroom = scale_headroom
        # Fused decode-in-GEMM avoids the dense weight but re-decodes per tile.
        # Measured on A100 (1024->4096, batch 1..128): it never wins — 7.8 vs 5.4 ms
        # at M=1 and 24.9 vs 5.8 ms at M=128. Default off; kept for small-VRAM inference.
        self.fused = fused
        # "step": keep the decoded dense weight until this layer's backward frees it.
        #   Fast, but on a deep stack every layer's fp32 weight is live at the end of
        #   forward (measured: 32 bits/param x L, the main reason peak exceeds dense).
        # "none": drop it right after the forward GEMM and re-decode in backward
        #   (costs ~0.36 ms per 1024->4096 layer, saves 32 bits/param per layer).
        assert weight_cache in ("step", "none")
        self.weight_cache = weight_cache
        # 0 = frozen scale (historical behaviour, and the bug this parameter exists to fix):
        # weights grow during training, the E8M0 block scale does not follow, and the mass
        # drifts onto the outermost rung. >0 refreshes the scale every N updates.
        self.rescale_every = int(rescale_every)
        self._steps_since_rescale = 0
        # "headroom" reproduces the historical max-rule; "mse" searches adjacent octaves
        # for the exponent that minimises block reconstruction error (outlier-robust).
        assert scale_mode in ("headroom", "mse")
        self.scale_mode = scale_mode
        self.clip_frac = float(clip_frac)

        init = torch.empty(out_features, in_features).uniform_(
            -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5)
        codes, sb = fmt.quantize(init, headroom=scale_headroom)
        self.register_buffer("_state", fmt.pack_state(codes))
        self.register_buffer("_scale", sb)
        self.register_buffer("v_rms", torch.full((out_features, 1), rms_eps))
        self._codes_cache: torch.Tensor | None = None
        self._token = torch.zeros(0, requires_grad=True)
        self._pending = False
        self._weight_cache: torch.Tensor | None = None

    @property
    def codes(self) -> torch.Tensor:
        """Unpacked code words (value + accumulator in one word)."""
        if self._codes_cache is None:
            self._codes_cache = self.fmt.unpack_state(self._state, self.in_features)
        return self._codes_cache

    def weight(self) -> torch.Tensor:
        """Visible weight (float); accumulator is invisible to forward.
        Cached — re-decodes only after codes change (apply_update_)."""
        if self._weight_cache is None:
            self._weight_cache = self.fmt.dequantize_fast(self.codes, self._scale)
        return self._weight_cache

    def persistent_bytes(self) -> int:
        return self.fmt.persistent_bytes(self.out_features, self.in_features)

    def bits_per_param(self) -> float:
        return self.persistent_bytes() * 8.0 / (self.out_features * self.in_features)

    @classmethod
    def from_dense(cls, weight: torch.Tensor, fmt: PackedNA, **kwargs) -> "MXNLinear":
        layer = cls(weight.shape[1], weight.shape[0], fmt, **kwargs)
        with torch.no_grad():
            codes, sb = fmt.quantize(weight.detach().float(),
                                     headroom=kwargs.get("scale_headroom", 2.0))
            layer._state.copy_(fmt.pack_state(codes))
            layer._scale.copy_(sb)
            layer._codes_cache = None
            layer._weight_cache = None
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features}, got {x.shape[-1]}")
        shape = x.shape
        if torch.is_grad_enabled() and self.training:
            if self._pending:
                raise RuntimeError("MXNLinear forwarded twice before backward")
            self._pending = True
            out = _MXNFunction.apply(x.reshape(-1, shape[-1]), self, self._token)
            return out.reshape(*shape[:-1], self.out_features)
        return x @ self.weight().t()

    def apply_update_(self, grad_w: torch.Tensor) -> None:
        """Per-param integrator update, called from backward."""
        self.v_rms, grad_hat = update.rms_update(
            grad_w.detach().float(), self.v_rms, self.rms_beta, self.rms_eps)
        new_codes = self.fmt.step(self.codes, self._scale, grad_hat, self.lr)
        self._state.copy_(self.fmt.pack_state(new_codes))
        self._codes_cache = None  # invalidate unpacked view
        self._weight_cache = None  # invalidate: codes changed
        if self.rescale_every:
            self._steps_since_rescale += 1
            if self._steps_since_rescale >= self.rescale_every:
                self._steps_since_rescale = 0
                self.refresh_scale_()
        # scale otherwise stays frozen (headroom is an init choice, not an invariant)

    @torch.no_grad()
    def refresh_scale_(self) -> None:
        """Re-fit the E8M0 block scale to the current weights, keeping the accumulator.

        The weight is decoded with the old scale, a new scale is fitted to it with the
        same headroom rule, and only the magnitude index is re-quantised. The
        accumulator residue is carried over untouched, so the optimizer state survives
        the rescale. Without this the scale stays at its init value while the weights
        grow, and the mass drifts onto the outermost rung.
        """
        w = self.fmt.dequantize(self.codes, self._scale)
        if self.scale_mode == "mse":
            new_sb = self.fmt.scale_bytes_mse(w, self.scale_headroom, self.clip_frac)
        else:
            new_sb = self.fmt.scale_bytes(w, self.scale_headroom)
        s_new = self.fmt.scale_floats(new_sb).repeat_interleave(_NA_BLOCK, dim=1)
        mag = torch.bucketize(w / s_new, self.fmt.mids.to(w.device)).clamp(0, self.fmt.n_vals - 1)
        acc = (self.codes.to(torch.long) % self.fmt.acc_levels) - (self.fmt.C - 1)
        self._state.copy_(self.fmt.pack_state(self.fmt.encode(mag, acc)))
        self._scale.copy_(new_sb)
        self._codes_cache = None
        self._weight_cache = None

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"lr={self.lr}, fmt={self.fmt.bits}bit/{self.fmt.n_codes}codes")
