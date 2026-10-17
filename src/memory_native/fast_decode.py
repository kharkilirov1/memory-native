"""Dense-decode + cuBLAS fast path for the plain packed 6-bit counter layer.

Why this exists (measured root cause, 2026-10-18 A100):
    The char-LM gate's default counter kinds spent 20-35 ms/step of a 55-60 ms step inside
    the *torch* decode/update path: ~19k-30k CUDA launches per step vs 2.9k for dense AdamW.
    Two cheaper paths already exist in the repo:
      * ``triton_counter.TritonCounterLinear`` decodes inside the GEMM (2 kernels/layer) and
        inherits the one-launch fused update;
      * ``PackedRMSCounterLinear`` keeps the fused update but decodes with ~13 torch ops.
    What was missing is the middle option that wins on small/narrow layers: **one Triton
    kernel decodes the packed state to a transient dense fp32 tile, then cuBLAS does the
    matmul**. cuBLAS is far better than a hand-written Triton matmul at these shapes, and a
    single decode kernel replaces the torch op storm.

Memory contract: the packed 6-bit state stays the only *persistent* weight storage. The
dense weight is a transient buffer, freed after the matmul (cost counted in peak VRAM by
the benchmark harnesses; no persistent FP32 copy, no AdamW moments).

Correctness: the decoded weight is bit-identical to ``scale * t`` from
``decode_state(unpack_codes(state))`` (integer division on 6-bit codes), so the forward and
grad_x are algebraically identical to the torch path; the update is the inherited fused
Triton kernel (quanta-level SR contract).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .packed import PackedRMSCounterLinear

__all__ = ["HAS_TRITON", "decode_dense_triton", "PackedDecodeCounterLinear"]

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only installs
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _decode_dense_kernel(state_ptr, scale_ptr, out_ptr, out_features, n_pack,
                             BLOCK_O: tl.constexpr, BLOCK_P: tl.constexpr, C: tl.constexpr):
        """packed 6-bit state [out, 3*n_pack] + row scale [out] -> dense fp32 [out, 4*n_pack].

        Codes are 4-per-3-bytes in original column order (packed.pack_codes layout):
        code k of pack p lives at bits [6k, 6k+6) of the 24-bit little-endian word formed by
        bytes (3p, 3p+1, 3p+2). One program decodes a [BLOCK_O, BLOCK_P]-pack tile and writes
        4 floats per pack; there is no atomics and no reuse of the input row stride.
        """
        lv = 2 * C - 1
        pid_o = tl.program_id(0)
        pid_p = tl.program_id(1)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        mo = offs_o < out_features
        mp = offs_p < n_pack
        m = mo[:, None] & mp[None, :]
        row_base = offs_o[:, None] * (3 * n_pack) + offs_p[None, :] * 3
        b0 = tl.load(state_ptr + row_base + 0, mask=m, other=0).to(tl.int32)
        b1 = tl.load(state_ptr + row_base + 1, mask=m, other=0).to(tl.int32)
        b2 = tl.load(state_ptr + row_base + 2, mask=m, other=0).to(tl.int32)
        d0 = b0 & 0x3F
        d1 = ((b0 >> 6) | (b1 << 2)) & 0x3F
        d2 = ((b1 >> 4) | (b2 << 4)) & 0x3F
        d3 = (b2 >> 2) & 0x3F
        s = tl.load(scale_ptr + offs_o, mask=mo, other=0.0).to(tl.float32)
        in_features = 4 * n_pack
        out_base = offs_o[:, None] * in_features + offs_p[None, :] * 4
        s2 = s[:, None]
        tl.store(out_ptr + out_base + 0, s2 * (d0 // lv - 1).to(tl.float32), mask=m)
        tl.store(out_ptr + out_base + 1, s2 * (d1 // lv - 1).to(tl.float32), mask=m)
        tl.store(out_ptr + out_base + 2, s2 * (d2 // lv - 1).to(tl.float32), mask=m)
        tl.store(out_ptr + out_base + 3, s2 * (d3 // lv - 1).to(tl.float32), mask=m)


def decode_dense_triton(state: torch.Tensor, scale: torch.Tensor, *, C: int,
                        out_features: int, in_features: int,
                        block_o: int = 32, block_p: int = 64) -> torch.Tensor:
    """One-launch decode of a packed row-scale counter layer to a dense fp32 weight.

    ``state`` is the packed [out, (in//4)*3] uint8 buffer, ``scale`` [out, 1] or [out].
    Returns a NEW fp32 tensor [out, in] (transient; caller frees it).
    """
    if not HAS_TRITON:
        raise RuntimeError("decode_dense_triton needs triton")
    if not (state.is_cuda and scale.is_cuda):
        raise ValueError("decode_dense_triton requires CUDA tensors")
    if in_features % 4:
        raise ValueError("in_features must be divisible by 4")
    n_pack = in_features // 4
    out = torch.empty((out_features, in_features), device=state.device, dtype=torch.float32)
    grid = (triton.cdiv(out_features, block_o), triton.cdiv(n_pack, block_p))
    with torch.cuda.device(state.device):
        _decode_dense_kernel[grid](
            state.contiguous(), scale.reshape(-1).contiguous(), out,
            out_features, n_pack, BLOCK_O=block_o, BLOCK_P=block_p, C=C, num_warps=4,
        )
    return out


class PackedDecodeCounterLinear(PackedRMSCounterLinear):
    """Packed 6-bit counter layer: one Triton decode per direction + cuBLAS + fused update.

    Forward: decode packed state -> transient dense fp32 tile -> ``F.linear``.
    Backward: decode again -> ``grad_out @ W`` for grad_x (cuBLAS); the packed-state
    counter update runs via the inherited one-launch fused Triton kernel.
    No persistent fp32 weight, no AdamW moments; the transient dense tile is 1 KiB-1 MiB
    per layer depending on width and is released before the next layer runs.
    """

    def _forward_matmul(self, x: torch.Tensor) -> torch.Tensor:
        w = decode_dense_triton(self.state, self.scale, C=self.C,
                                out_features=self.out_features,
                                in_features=self.in_features)
        try:
            return F.linear(x, w)
        finally:
            del w

    def _has_fast_grad_x(self) -> bool:
        return True

    def _backward_grad_x(self, grad_out2d: torch.Tensor) -> torch.Tensor:
        w = decode_dense_triton(self.state, self.scale, C=self.C,
                                out_features=self.out_features,
                                in_features=self.in_features)
        try:
            return grad_out2d @ w
        finally:
            del w
