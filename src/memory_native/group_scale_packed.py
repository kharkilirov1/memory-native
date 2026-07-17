"""Packed trainable group-scale counter linear with strict Triton kernels."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .counter import decode_state, encode_state
from .group_scale_kernels import (
    HAS_TRITON,
    group_counter_update_from_io_hashsr,
    group_update_scratch_bytes,
    triton_group_counter_update_from_io,
    triton_group_decode_matmul,
    triton_group_grad_x,
)
from .packed import pack_codes, unpack_codes

__all__ = ["PackedGroupScaleCounterLinear"]


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


class _PackedGroupScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "PackedGroupScaleCounterLinear", tap: torch.Tensor):
        if module._outstanding_forward:
            raise RuntimeError(
                "PackedGroupScaleCounterLinear was reused before its previous backward; "
                "weight sharing/gradient accumulation need an explicit scheduler"
            )
        module._outstanding_forward = True
        x2 = x.reshape(-1, x.shape[-1])
        y2 = module._forward_2d(x2)
        ctx.module = module
        ctx.x_shape = x.shape
        ctx.save_for_backward(x2)
        return y2.reshape(*x.shape[:-1], module.out_features)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        module: PackedGroupScaleCounterLinear = ctx.module
        (x2,) = ctx.saved_tensors
        go2 = grad_out.reshape(-1, grad_out.shape[-1])
        try:
            # grad_x must use the same pre-update state as the forward.
            grad_x2 = module._grad_x_2d(go2)
            if module.training and module.update_enabled:
                module._update_from_io(x2, go2)
        finally:
            module._outstanding_forward = False
        return grad_x2.reshape(ctx.x_shape), None, None


class PackedGroupScaleCounterLinear(nn.Module):
    """6-bit act-ordered group counter with group-aware forward/grad-x/strict update kernels.

    State is packed in act-order, not original input order. That aligns each scale group with a
    contiguous 6-bit range and makes state updates race-free. ``perm[p]`` maps a packed position to
    its original input column; forward gathers x through perm and grad_x scatters back through it.

    On CUDA+Triton the training path never materializes a dense weight or dense weight-gradient.
    CPU/no-Triton transparently falls back to a dense reference for correctness.

    Salient channel (A4.1): an optional sparse set of EXACT fp16 overrides
    (``salient_idx`` flat original-order, ``salient_val``) for the top-|w|*sqrt(diag H)
    weights the ternary grid cannot represent (BiLLM-style split, made BEFORE the GPTQ
    sweep by the solver). Base (t, c) is zero at salient entries and kept zero by the
    update path, so the override is additive everywhere: Triton kernels add it as a sparse
    correction, the dense reference scatters it into ``visible_weight``. Salient entries
    are frozen (no counter movement); training them is a separate future lever.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: int = 128,
        C: int = 11,
        lr: float = 2e-3,
        lr_scale: float = 2e-4,
        rms_beta: float = 0.9,
        rms_eps: float = 1e-3,
        local_grad_clip: float = 0.0,
        residual_alpha: float = 0.0,
        perm: torch.Tensor | None = None,
        kernel_mode: str = "auto",
        strict_update: bool = True,
        init_gain: float = 1.0,
        flip_sample_size: int = 4096,
    ) -> None:
        super().__init__()
        del init_gain  # PTQ imports the state; kept for factory compatibility.
        if in_features % 4:
            raise ValueError("in_features must be divisible by 4 for 6-bit packing")
        if group <= 0 or group % 4:
            raise ValueError("group must be positive and divisible by 4")
        if 3 * (2 * C - 1) > 256:
            raise ValueError("C is too large for uint8 state encoding")
        if kernel_mode not in {"auto", "triton", "torch"}:
            raise ValueError("kernel_mode must be 'auto', 'triton' or 'torch'")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group = int(group)
        self.n_groups = (self.in_features + self.group - 1) // self.group
        self.C = int(C)
        self.lr = float(lr)
        self.lr_scale = float(lr_scale)
        self.rms_beta = float(rms_beta)
        self.rms_eps = float(rms_eps)
        self.local_grad_clip = float(local_grad_clip)
        self.residual_alpha = float(residual_alpha)
        self.kernel_mode = kernel_mode
        self.strict_update = bool(strict_update)
        self.update_enabled = True
        self._outstanding_forward = False
        self._sr_step = 0

        zeros = torch.zeros((self.out_features, self.in_features), dtype=torch.int16)
        self.register_buffer("state", pack_codes(encode_state(zeros, zeros, self.C)))
        self.register_buffer(
            "scale",
            torch.full((self.out_features, self.n_groups), 1e-2, dtype=torch.float32),
        )
        self.register_buffer("v", torch.zeros((self.out_features, 1), dtype=torch.float32))
        # int32 is sufficient for model dimensions and halves checkpoint/metadata bytes.
        self.register_buffer("perm", torch.empty(self.in_features, dtype=torch.int32))
        # Salient channel: flat ORIGINAL-order indices (o*in_features + j) + fp16 values.
        self.register_buffer("salient_idx", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("salient_val", torch.zeros(0, dtype=torch.float16))
        self.register_buffer(
            "_salient_perm_flat", torch.zeros(0, dtype=torch.int64), persistent=False
        )
        self._salient_sparse_cache = None
        self._salient_zero_code = int(encode_state(
            torch.zeros(1, dtype=torch.int16), torch.zeros(1, dtype=torch.int16), self.C
        )[0])
        self.register_buffer("weight_flips", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("update_events", torch.zeros((), dtype=torch.int64), persistent=False)
        self.set_permutation(torch.arange(self.in_features) if perm is None else perm)
        # Python hot-path SR counter, mirrored into a persistent buffer for exact resume.
        self.register_buffer("sr_step", torch.tensor(self._sr_step, dtype=torch.int64))
        self.flip_sample_size = max(0, int(flip_sample_size))
        total = self.out_features * self.in_features
        sample_n = min(self.flip_sample_size, total)
        if sample_n:
            idx = torch.linspace(0, total - 1, sample_n, dtype=torch.float64).round().long().unique()
        else:
            idx = torch.empty(0, dtype=torch.long)
        self.register_buffer("_flip_sample_indices", idx, persistent=False)
        self.register_buffer("_flip_sample_prev", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("flip_rate_alt", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("counter_edge_sample", torch.zeros((), dtype=torch.float32), persistent=False)

        def _sync_after_load(module, _incompatible):
            module._sr_step = int(module.sr_step.detach().cpu())
            module.observe_flip_sample(reset=True)

        self.register_load_state_dict_post_hook(_sync_after_load)

    def _sample_codes(self) -> torch.Tensor:
        """Ternary codes at the flip-sample positions, read straight out of the 6-bit packing."""
        idx = self._flip_sample_indices
        if idx.numel() == 0:
            return torch.empty(0, device=self.state.device, dtype=torch.int16)
        idx = idx.to(self.state.device)
        row = torch.div(idx, self.in_features, rounding_mode="floor")
        pos = torch.remainder(idx, self.in_features)
        packed_group = torch.div(pos, 4, rounding_mode="floor")
        lane = torch.remainder(pos, 4)
        base = packed_group * 3
        b0 = self.state[row, base].to(torch.int32)
        b1 = self.state[row, base + 1].to(torch.int32)
        b2 = self.state[row, base + 2].to(torch.int32)
        c0 = b0 & 0x3F
        c1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
        c2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
        c3 = (b2 >> 2) & 0x3F
        return torch.where(
            lane == 0, c0, torch.where(lane == 1, c1, torch.where(lane == 2, c2, c3))
        ).to(torch.int16)

    @torch.no_grad()
    def observe_flip_sample(self, *, reset: bool = False) -> dict[str, float]:
        """Flip-rate / counter-edge telemetry on the sample (no full state unpack)."""
        codes = self._sample_codes()
        if codes.numel() == 0:
            self.flip_rate_alt.zero_()
            self.counter_edge_sample.zero_()
            return {"flip_rate_alt": 0.0, "counter_edge_sample": 0.0, "sample_size": 0.0}
        levels = 2 * self.C - 1
        ternary = (torch.div(codes, levels, rounding_mode="floor") - 1).to(torch.int8)
        residual = torch.remainder(codes, levels) - (self.C - 1)
        self.counter_edge_sample.copy_((residual.abs() >= self.C - 1).float().mean())
        if reset or self._flip_sample_prev.numel() != ternary.numel():
            self._flip_sample_prev.resize_as_(ternary).copy_(ternary)
            self.flip_rate_alt.zero_()
        else:
            self.flip_rate_alt.copy_((ternary != self._flip_sample_prev).float().mean())
            self._flip_sample_prev.copy_(ternary)
        return {
            "flip_rate_alt": float(self.flip_rate_alt),
            "counter_edge_sample": float(self.counter_edge_sample),
            "sample_size": float(ternary.numel()),
        }

    @torch.no_grad()
    def set_permutation(self, perm: torch.Tensor) -> None:
        p = perm.detach().to(device=self.state.device, dtype=torch.long).reshape(-1)
        if p.numel() != self.in_features:
            raise ValueError("perm length mismatch")
        expected = torch.arange(self.in_features, device=p.device)
        if not torch.equal(torch.sort(p).values, expected):
            raise ValueError("perm must contain every input column exactly once")
        self.perm.copy_(p.to(torch.int32))

    def _all_codes_perm(self) -> torch.Tensor:
        return unpack_codes(self.state, self.in_features)

    def _decode_perm(self):
        return decode_state(self._all_codes_perm(), self.C)

    def _has_salient(self) -> bool:
        return self.salient_idx.numel() > 0

    def visible_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dense reference weight in original input order (debug/CPU only)."""
        t, c = self._decode_perm()
        positions = torch.arange(self.in_features, device=self.state.device)
        group_idx = torch.div(positions, self.group, rounding_mode="floor")
        w_perm = self.scale[:, group_idx] * (
            t.float() + self.residual_alpha * c.float() / self.C
        )
        w = torch.empty_like(w_perm)
        w[:, self.perm.long()] = w_perm
        if self._has_salient():
            # Exact overrides; base is zero at salient entries, so copy == add.
            w.reshape(-1).index_copy_(0, self.salient_idx.long(),
                                      self.salient_val.float())
        return w if dtype is None else w.to(dtype)

    def _salient_sparse(self) -> torch.Tensor | None:
        """Salient overrides as a coalesced COO [out, in] fp32 (cached per device)."""
        if not self._has_salient():
            return None
        dev = self.salient_val.device
        cache = self._salient_sparse_cache
        if cache is not None and cache.device == dev:
            return cache
        idx = self.salient_idx.long()
        o = torch.div(idx, self.in_features, rounding_mode="floor")
        j = idx - o * self.in_features
        A = torch.sparse_coo_tensor(
            torch.stack([o, j]), self.salient_val.float(),
            (self.out_features, self.in_features), device=dev,
        ).coalesce()
        self._salient_sparse_cache = A
        return A

    def _use_triton(self, tensor: torch.Tensor) -> bool:
        if self.kernel_mode == "torch":
            return False
        available = HAS_TRITON and tensor.is_cuda and tensor.dtype in {
            torch.float32, torch.float16, torch.bfloat16
        }
        if self.kernel_mode == "triton" and not available:
            raise RuntimeError("kernel_mode='triton' requires CUDA + Triton and a floating input")
        return available

    def _forward_2d(self, x2: torch.Tensor) -> torch.Tensor:
        if self._use_triton(x2):
            y2 = triton_group_decode_matmul(
                x2, self.state, self.scale, self.perm,
                C=self.C, group=self.group, residual_alpha=self.residual_alpha,
            )
            A = self._salient_sparse()
            if A is not None:
                # Base is zero at salient entries: the sparse add IS the override.
                y2 = y2 + torch.sparse.mm(A, x2.t()).t().to(y2.dtype)
            return y2
        return x2 @ self.visible_weight(dtype=x2.dtype).t()

    def _grad_x_2d(self, go2: torch.Tensor) -> torch.Tensor:
        if self._use_triton(go2):
            gx2 = triton_group_grad_x(
                go2, self.state, self.scale, self.perm,
                in_features=self.in_features, C=self.C, group=self.group,
                residual_alpha=self.residual_alpha,
            )
            A = self._salient_sparse()
            if A is not None:
                gx2 = gx2 + torch.sparse.mm(A.t(), go2.t()).t().to(gx2.dtype)
            return gx2
        return go2 @ self.visible_weight(dtype=go2.dtype)

    @torch.no_grad()
    def _update_from_io(self, x2: torch.Tensor, go2: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError(
                "PackedGroupScaleCounterLinear strict update is single-rank for now; "
                "a distributed run would need a groupwise correlation all-reduce before state mutation"
            )
        want_triton = self._use_triton(x2) and self.strict_update
        if want_triton and not _is_power_of_two(self.group):
            raise ValueError(
                "strict Triton group update requires a power-of-two group size; "
                "use group=32/64/128/256 or kernel_mode='torch'"
            )
        seed = self._sr_step
        if want_triton and not self._has_salient():
            # No [out,in] grad_w and no dense W are created on this path.
            triton_group_counter_update_from_io(
                self.state, self.scale, self.v, x2, go2, self.perm,
                group=self.group, C=self.C, lr=self.lr, lr_scale=self.lr_scale,
                rms_beta=self.rms_beta, rms_eps=self.rms_eps, seed=seed,
                residual_alpha=self.residual_alpha, clip=self.local_grad_clip,
            )
        else:
            # Reference path. Also the salient path: salient entries are FROZEN — the
            # strict kernel has no freeze mask, so their codes are re-zeroed below.
            old_codes = self._all_codes_perm()
            new_codes = group_counter_update_from_io_hashsr(
                old_codes, self.scale, self.v, x2, go2, self.perm,
                group=self.group, C=self.C, lr=self.lr, lr_scale=self.lr_scale,
                rms_beta=self.rms_beta, rms_eps=self.rms_eps, seed=seed,
                residual_alpha=self.residual_alpha, clip=self.local_grad_clip,
            )
            if self._has_salient():
                new_codes = new_codes.clone()
                new_codes.reshape(-1)[self._salient_perm_flat] = self._salient_zero_code
            old_t, _ = decode_state(old_codes, self.C)
            new_t, _ = decode_state(new_codes, self.C)
            self.state.copy_(pack_codes(new_codes))
            self.weight_flips.add_((new_t != old_t).sum().to(self.weight_flips.dtype))
        self._sr_step += 1
        self.sr_step.fill_(self._sr_step)
        self.update_events.add_(self.out_features * self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and torch.is_grad_enabled()):
            x2 = x.reshape(-1, x.shape[-1])
            y2 = self._forward_2d(x2)
            return y2.reshape(*x.shape[:-1], self.out_features)
        tap = torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
        return _PackedGroupScaleFn.apply(x, self, tap)

    @torch.no_grad()
    def load_group_state(
        self,
        scales: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
        salient_idx: torch.Tensor | None = None,
        salient_val: torch.Tensor | None = None,
    ) -> None:
        if perm is not None:
            self.set_permutation(perm)
        scales = scales.to(device=self.state.device, dtype=torch.float32)
        t = t.to(device=self.state.device, dtype=torch.int16)
        c = torch.zeros_like(t) if c is None else c.to(device=self.state.device, dtype=torch.int16)
        if scales.shape != self.scale.shape:
            raise ValueError(f"scale shape {tuple(scales.shape)} != {tuple(self.scale.shape)}")
        if t.shape != (self.out_features, self.in_features) or c.shape != t.shape:
            raise ValueError("t/c shape mismatch")
        if not set(t.unique().tolist()) <= {-1, 0, 1}:
            raise ValueError("t must be ternary")
        if c.abs().max().item() > self.C - 1:
            raise ValueError("counter residual exceeds representable range")
        salient_idx = torch.zeros(0, dtype=torch.int32) if salient_idx is None else salient_idx
        salient_val = torch.zeros(0, dtype=torch.float16) if salient_val is None else salient_val
        salient_idx = salient_idx.to(device=self.state.device, dtype=torch.int32).reshape(-1)
        salient_val = salient_val.to(device=self.state.device, dtype=torch.float16).reshape(-1)
        if salient_idx.numel() != salient_val.numel():
            raise ValueError("salient_idx/salient_val length mismatch")
        if salient_idx.numel():
            if int(salient_idx.min()) < 0 or int(salient_idx.max()) >= t.numel():
                raise ValueError("salient_idx out of range")
            if salient_idx.unique().numel() != salient_idx.numel():
                raise ValueError("salient_idx must not contain duplicates")
            # The salient component owns these entries: base (t, c) is zero there.
            t = t.clone()
            c = c.clone()
            t.reshape(-1)[salient_idx.long()] = 0
            c.reshape(-1)[salient_idx.long()] = 0
        p = self.perm.long()
        codes_perm = encode_state(t[:, p], c[:, p], self.C)
        self.state.copy_(pack_codes(codes_perm))
        self.scale.copy_(scales.clamp_min(1e-8))
        self.salient_idx = salient_idx
        self.salient_val = salient_val
        if salient_idx.numel():
            inv = torch.argsort(p)
            o = salient_idx.long() // self.in_features
            j = salient_idx.long() % self.in_features
            self._salient_perm_flat = o * self.in_features + inv[j]
        else:
            self._salient_perm_flat = torch.zeros(0, dtype=torch.int64, device=self.state.device)
        self._salient_sparse_cache = None
        self.v.zero_()
        self.weight_flips.zero_()
        self.update_events.zero_()
        self._sr_step = 0
        self.sr_step.zero_()
        self.observe_flip_sample(reset=True)

    @torch.no_grad()
    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    @torch.no_grad()
    def set_residual_alpha(self, alpha: float) -> None:
        self.residual_alpha = float(min(1.0, max(0.0, alpha)))

    @torch.no_grad()
    def state_statistics(self) -> dict[str, float]:
        t, c = self._decode_perm()
        return {
            "minus": float((t == -1).float().mean()),
            "zero": float((t == 0).float().mean()),
            "plus": float((t == 1).float().mean()),
            "counter_abs_mean": float(c.float().abs().mean()),
            "counter_edge": float((c.abs() >= self.C - 1).float().mean()),
            "scale_mean": float(self.scale.mean()),
            "residual_alpha": float(self.residual_alpha),
            "strict_scratch_mib": self.strict_scratch_bytes() / (1024 ** 2),
            "salient_fraction": float(self.salient_idx.numel())
                / (self.out_features * self.in_features),
            "flip_rate_alt": float(self.flip_rate_alt),
            "counter_edge_sample": float(self.counter_edge_sample),
            "sr_step": float(self.sr_step),
        }

    def strict_scratch_bytes(self) -> int:
        return group_update_scratch_bytes(self.out_features, self.in_features, self.group)

    def persistent_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.state, self.scale, self.v, self.perm,
                           self.salient_idx, self.salient_val, self.sr_step)
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, group={self.group}, C={self.C}, "
            f"alpha={self.residual_alpha:.3f}, kernel={self.kernel_mode}, strict={self.strict_update}, "
            f"salient={self.salient_idx.numel()}"
        )
