"""Runtime review fixes for solver-v3 packed group recovery.

The patch is applied from ``memory_native.__init__`` so existing public class/function objects keep
working while the review fixes remain isolated and easy to retire after the next source refactor.
"""
from __future__ import annotations

import functools
import torch

_APPLIED = False


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def apply_review_fixes() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from . import counter as counter_mod
    from . import group_scale_counter as ref_mod
    from . import group_scale_kernels as kernels
    from . import group_scale_packed as packed_mod
    from .donor import ptq as ptq_mod
    from .packed import pack_codes, unpack_codes

    # One canonical carry/saturation implementation. Existing _update resolves this module global.
    ref_mod._carry_resolve = counter_mod._carry_resolve

    old_triton_group_update = kernels.triton_group_counter_update_from_io

    @functools.wraps(old_triton_group_update)
    def safe_triton_group_update(*args, group: int, **kwargs):
        if not _is_power_of_two(group):
            raise ValueError(
                "strict Triton group update requires a power-of-two group size; "
                "use group=32/64/128/256 or the torch reference path"
            )
        return old_triton_group_update(*args, group=group, **kwargs)

    kernels.triton_group_counter_update_from_io = safe_triton_group_update
    packed_mod.triton_group_counter_update_from_io = safe_triton_group_update
    import sys
    root_module = sys.modules.get(__package__)
    if root_module is not None:
        root_module.triton_group_counter_update_from_io = safe_triton_group_update

    cls = packed_mod.PackedGroupScaleCounterLinear
    old_init = cls.__init__
    old_load_group_state = cls.load_group_state
    old_state_statistics = cls.state_statistics
    old_persistent_bytes = cls.persistent_bytes

    @functools.wraps(old_init)
    def reviewed_init(self, *args, flip_sample_size: int = 4096, **kwargs):
        old_init(self, *args, **kwargs)
        # Keep a Python hot-path counter, mirrored into a persistent buffer for exact resume.
        self._sr_step = int(getattr(self, "_sr_step", 0))
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

    def _sample_codes(self):
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
    def observe_flip_sample(self, *, reset: bool = False):
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
    def reviewed_update_from_io(self, x2, go2):
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError(
                "PackedGroupScaleCounterLinear strict update is single-rank for now; "
                "a distributed run needs a groupwise correlation all-reduce"
            )
        use_triton = self._use_triton(x2) and self.strict_update
        if use_triton and not _is_power_of_two(self.group):
            raise ValueError(
                "strict Triton group update requires a power-of-two group size; "
                "use group=32/64/128/256 or kernel_mode='torch'"
            )
        seed = self._sr_step
        if use_triton:
            kernels.triton_group_counter_update_from_io(
                self.state, self.scale, self.v, x2, go2, self.perm,
                group=self.group, C=self.C, lr=self.lr, lr_scale=self.lr_scale,
                rms_beta=self.rms_beta, rms_eps=self.rms_eps, seed=seed,
                residual_alpha=self.residual_alpha, clip=self.local_grad_clip,
            )
        else:
            old_codes = unpack_codes(self.state, self.in_features)
            new_codes = kernels.group_counter_update_from_io_hashsr(
                old_codes, self.scale, self.v, x2, go2, self.perm,
                group=self.group, C=self.C, lr=self.lr, lr_scale=self.lr_scale,
                rms_beta=self.rms_beta, rms_eps=self.rms_eps, seed=seed,
                residual_alpha=self.residual_alpha, clip=self.local_grad_clip,
            )
            old_t, _ = counter_mod.decode_state(old_codes, self.C)
            new_t, _ = counter_mod.decode_state(new_codes, self.C)
            self.state.copy_(pack_codes(new_codes))
            self.weight_flips.add_((new_t != old_t).sum().to(self.weight_flips.dtype))
        self._sr_step += 1
        self.sr_step.fill_(self._sr_step)
        self.update_events.add_(self.out_features * self.in_features)

    @torch.no_grad()
    def reviewed_load_group_state(self, *args, **kwargs):
        old_load_group_state(self, *args, **kwargs)
        self._sr_step = 0
        self.sr_step.zero_()
        self.observe_flip_sample(reset=True)

    @torch.no_grad()
    def reviewed_state_statistics(self):
        result = old_state_statistics(self)
        result.update(
            flip_rate_alt=float(self.flip_rate_alt),
            counter_edge_sample=float(self.counter_edge_sample),
            sr_step=float(self.sr_step),
        )
        return result

    def reviewed_persistent_bytes(self):
        return old_persistent_bytes(self) + self.sr_step.numel() * self.sr_step.element_size()

    cls.__init__ = reviewed_init
    cls._sample_codes = _sample_codes
    cls.observe_flip_sample = observe_flip_sample
    cls._update_from_io = reviewed_update_from_io
    cls.load_group_state = reviewed_load_group_state
    cls.state_statistics = reviewed_state_statistics
    cls.persistent_bytes = reviewed_persistent_bytes
    cls.__review_fixes_applied__ = True

    old_ptq = ptq_mod.ptq_warm_start

    @functools.wraps(old_ptq)
    def safe_ptq_warm_start(model, calib_batches, *, mode="gptq", **kwargs):
        if mode not in {"gptq_group", "group128v3", "group"}:
            for key in ("residual_alpha", "kernel_mode", "strict_update", "flip_sample_size"):
                kwargs.pop(key, None)
        return old_ptq(model, calib_batches, mode=mode, **kwargs)

    ptq_mod.ptq_warm_start = safe_ptq_warm_start
    _APPLIED = True
