"""Experimental 2-bit visible cache for Memory Native.

This module is a CPU/PyTorch correctness oracle, NOT an accelerated GEMV kernel.
For strict alpha=0, forward sees only ternary t; hidden counter c remains in
canonical 6-bit packed state for training. Four ternary values fit in one byte.
"""
from __future__ import annotations

import math
import torch


def pack_visible2(t: torch.Tensor) -> torch.Tensor:
    """Pack {-1,0,+1} using codes {0,1,2}; four values per uint8."""
    if t.ndim < 1 or t.shape[-1] % 4:
        raise ValueError("last dimension must be a positive multiple of four")
    if t.shape[-1] == 0 or not torch.all((t >= -1) & (t <= 1) & (t == t.round())):
        raise ValueError("ternary input must contain only -1, 0, +1")
    z = (t.to(torch.int16) + 1).to(torch.uint8).reshape(*t.shape[:-1], -1, 4)
    return (z[..., 0] | (z[..., 1] << 2) | (z[..., 2] << 4) | (z[..., 3] << 6)).contiguous()


def unpack_visible2(packed: torch.Tensor, *, reject_invalid: bool = True) -> torch.Tensor:
    """Return int8 ternary tensor from packed bytes; reject reserved code 3."""
    if packed.ndim < 1 or packed.dtype != torch.uint8:
        raise ValueError("packed must be a uint8 tensor with >=1 dimensions")
    v = torch.stack(tuple((packed >> shift) & 0x03 for shift in (0, 2, 4, 6)), dim=-1)
    if reject_invalid and torch.any(v == 3):
        raise ValueError("packed visible2 cache contains reserved code 3")
    return v.reshape(*packed.shape[:-1], packed.shape[-1] * 4).to(torch.int8) - 1


def patch_visible2_cpu(packed: torch.Tensor, flat_positions: torch.Tensor, new_t: torch.Tensor) -> torch.Tensor:
    """Deterministic CPU reference patch; GPU needs a dedicated kernel.

    Multiple updates in the same packed byte are handled in order, without
    unsafe indexed assignments with duplicated byte offsets.
    """
    if packed.device.type != "cpu" or flat_positions.device.type != "cpu" or new_t.device.type != "cpu":
        raise ValueError("reference patch is CPU-only")
    if packed.dtype != torch.uint8 or flat_positions.ndim != 1 or new_t.ndim != 1 or flat_positions.numel() != new_t.numel():
        raise ValueError("invalid patch tensors")
    if flat_positions.numel() and (int(flat_positions.min()) < 0 or int(flat_positions.max()) >= packed.numel()*4):
        raise ValueError("patch index out of range")
    if new_t.numel() and not torch.all((new_t >= -1) & (new_t <= 1) & (new_t == new_t.round())):
        raise ValueError("invalid ternary values")
    patched = packed.clone()
    flat = patched.reshape(-1)
    for index, ternary in zip(flat_positions.tolist(), new_t.tolist()):
        packed_pos, lane = divmod(int(index), 4)
        shift = lane*2
        old = int(flat[packed_pos])
        flat[packed_pos] = (old & ~(3 << shift)) | ((int(ternary)+1) << shift)
    return patched


def visible2_matmul_reference(x: torch.Tensor, packed: torch.Tensor,
                              scale: torch.Tensor, perm: torch.Tensor,
                              group_size: int) -> torch.Tensor:
    """Strict group-scale reference: sum_g s[n,g] * sum_{p in g} t[n,p]*x[perm[p]].

    Materializes unpacked int8 t, but no FP32 dense W. NOT a latency benchmark.
    Salient FP overrides are not included; evaluate them as a separate addend.
    """
    if x.ndim != 2 or packed.ndim != 2 or scale.ndim != 2 or perm.ndim != 1:
        raise ValueError("expected x[M,K], packed[N,K/4], scale[N,G], perm[K]")
    M, K = x.shape
    N, packed_width = packed.shape
    if packed_width * 4 != K or group_size < 4 or group_size % 4:
        raise ValueError("invalid group size or packed dimensions")
    if scale.shape != (N, math.ceil(K / group_size)) or perm.shape[0] != K:
        raise ValueError("scale/perm dimensions mismatch")
    if not torch.equal(torch.sort(perm.long()).values, torch.arange(K, device=perm.device)):
        raise ValueError("perm must contain every feature once")
    t = unpack_visible2(packed)
    xp = x[:, perm.long()].float()
    y = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    for g in range(scale.shape[1]):
        lo = g * group_size
        hi = min(K, lo + group_size)
        y += (xp[:, lo:hi] @ t[:, lo:hi].float().T) * scale[:, g].float().unsqueeze(0)
    return y
