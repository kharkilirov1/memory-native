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
        if self.compact_state and self.integrator != 3:
            raise ValueError("compact_state currently requires integrator=3 (signed 2-bit remainder)")
        if self.compact_state and self.feedback:
            raise ValueError("compact_state does not support the separate 2-bit error-feedback plane")
        init = torch.empty(out_features, in_features).uniform_(
            -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5)
        codes, scale = pack.quantize(init, headroom=scale_headroom, mode=scale_mode)
        if self.compact_state:
            if codes.numel() % 4:
                raise ValueError("compact_state requires parameter count divisible by 4")
            self.register_buffer("_codes_packed", pack.pack_nibbles(codes))
            self.register_buffer("_acc_packed", pack.pack_2bit_signed(
                torch.zeros(out_features, in_features, dtype=torch.int8)))
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
            return pack.unpack_2bit_signed(self._acc_packed, (self.out_features, self.in_features))
        return self._acc_dense

    def _store_codes(self, codes: torch.Tensor) -> None:
        if self.compact_state:
            self._codes_packed.copy_(pack.pack_nibbles(codes))
        else:
            self._codes_dense.copy_(codes)

    def _store_acc(self, acc: torch.Tensor) -> None:
        if self.compact_state:
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
