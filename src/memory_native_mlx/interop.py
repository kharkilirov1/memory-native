"""torch <-> MLX counter-state interop: train/PTQ anywhere, fine-tune on a MacBook.

The bridge for the MacBook fine-tuning story: a counter model produced on CUDA — from
scratch, or PTQ-warm-started from a pretrained checkpoint (the gptq_group_ternary +
recovery path on the claude/finetune-pretrained-model / solver-v3 branches) — moves to
MLX losslessly, because both sides use the exact same 6-bit code space and the exact same
4-codes/3-bytes packing (bit-for-bit, both match the engine's cc_pack4). Training then
CONTINUES with the same dynamics: both sides implement the same hash-SR update.

torch is imported lazily; this module is importable on a Mac without torch installed
(the functions just require the torch objects you pass in).
"""
from __future__ import annotations

import numpy as np

import mlx.core as mx

from .counter import RMSCounterLinear, encode_state
from .packed import PackedRMSCounterLinear

__all__ = ["mlx_counter_from_torch", "export_counter_to_torch"]


def _torch_layer_tc(torch_layer):
    """Decode a torch counter layer's full (t, c) via its own storage hooks (handles both
    the unpacked and the packed torch layouts)."""
    t, c = torch_layer._decode_rows(0, torch_layer.out_features)
    return t.cpu().numpy(), c.cpu().numpy()


def mlx_counter_from_torch(torch_layer, *, packed: bool | None = None) -> RMSCounterLinear:
    """Build an MLX counter layer from a torch `memory_native` counter layer, copying the
    full training state (codes, per-row scale, RMS second moment v).

    `packed=None` mirrors the torch layer's storage (PackedRMSCounterLinear -> packed);
    pass True/False to force a layout. RMS hyperparameters are carried over; the SR stream
    starts at the torch layer's `_sr_step` when present so a CUDA->MLX handoff continues
    the same deterministic rounding stream."""
    if packed is None:
        packed = type(torch_layer).__name__.startswith("Packed")
    cls = PackedRMSCounterLinear if packed else RMSCounterLinear
    layer = cls(
        torch_layer.in_features,
        torch_layer.out_features,
        C=torch_layer.C,
        lr=torch_layer.lr,
        lr_scale=torch_layer.lr_scale,
        rms_beta=getattr(torch_layer, "rms_beta", 0.9),
        rms_eps=getattr(torch_layer, "rms_eps", 1e-3),
        use_rms=getattr(torch_layer, "use_rms", True),
        sr_seed=int(getattr(torch_layer, "_sr_step", 0)),
    )
    t_np, c_np = _torch_layer_tc(torch_layer)
    codes = encode_state(mx.array(t_np.astype(np.int32)), mx.array(c_np.astype(np.int32)), layer.C)
    layer._store_codes(codes)
    layer.scale = mx.array(torch_layer.scale.detach().cpu().numpy())
    if hasattr(torch_layer, "v"):
        layer.v = mx.array(torch_layer.v.detach().cpu().numpy())
    mx.eval(layer.parameters())
    return layer


def export_counter_to_torch(mlx_layer: RMSCounterLinear, torch_layer) -> None:
    """Write an MLX counter layer's state back into a torch `memory_native` counter layer
    (in place, through the torch layer's own storage hooks, so either torch layout works)."""
    import torch

    from .counter import decode_state as mx_decode

    t, c = mx_decode(mlx_layer._codes(), mlx_layer.C)
    mx.eval(t, c, mlx_layer.scale, mlx_layer.v)
    with torch.no_grad():
        torch_layer._write_rows(
            0, torch_layer.out_features,
            torch.from_numpy(np.array(t, copy=True)).to(torch.int16),
            torch.from_numpy(np.array(c, copy=True)).to(torch.int16),
        )
        torch_layer.scale.copy_(torch.from_numpy(np.array(mlx_layer.scale, copy=True)))
        if hasattr(torch_layer, "v"):
            torch_layer.v.copy_(torch.from_numpy(np.array(mlx_layer.v, copy=True)))
    if hasattr(torch_layer, "_sr_step"):
        torch_layer._sr_step = int(mlx_layer._sr_step)
