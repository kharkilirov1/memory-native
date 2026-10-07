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



def derive_visible2_from_packed6(packed6: torch.Tensor, C: int,
                                  *, chunk_groups: int = 1 << 16) -> torch.Tensor:
    """Streaming conversion of canonical packed6 -> visible2 without dense W or t.

    Each packed6 triple encodes four counter (t,c) coefficients. Convert one
    triple to a visible2 byte by discarding c, using a bounded scratch chunk.
    The CPU/PyTorch implementation is a correctness/export reference, not a
    high-throughput CUDA kernel. Output is exact only for strict alpha=0.
    """
    if packed6.ndim < 1 or packed6.dtype != torch.uint8 or packed6.shape[-1] % 3:
        raise ValueError("packed6 must be uint8 with three-byte groups")
    if not isinstance(C, int) or not 1 <= C <= 11:
        raise ValueError("C must be between 1 and 11")
    if not isinstance(chunk_groups, int) or chunk_groups < 1:
        raise ValueError("chunk_groups must be a positive integer")
    src = packed6.contiguous().reshape(-1,3)
    result = torch.empty(src.shape[0],dtype=torch.uint8,device=packed6.device)
    lv = 2*C-1
    for start in range(0, src.shape[0], chunk_groups):
        end = min(start + chunk_groups, src.shape[0])
        p = src[start:end].to(torch.int32)
        b0,b1,b2 = p[:,0], p[:,1], p[:,2]
        a0 = (b0 & 63) // lv
        a1 = (((b0 >> 6) | (b1 << 2)) & 63) // lv
        a2 = (((b1 >> 4) | (b2 << 4)) & 63) // lv
        a3 = ((b2 >> 2) & 63) // lv
        if torch.any((a0 > 2) | (a1 > 2) | (a2 > 2) | (a3 > 2)):
            raise ValueError("packed6 contains an invalid code")
        result[start:end] = (a0 | (a1 << 2) | (a2 << 4) | (a3 << 6)).to(torch.uint8)
    return result.reshape(*packed6.shape[:-1],packed6.shape[-1]//3)
