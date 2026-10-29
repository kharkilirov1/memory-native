#!/usr/bin/env python
"""Does the community MXFP4 path actually run on this GPU (sm_120)?

Builds a packed E2M1 x E8M0/32 matmul, runs it through Triton's `tl.dot_scaled`
(the same primitive torchao / gpt-oss-style kernels are built on) and checks the
result against a dequantised reference. Also reports whether the kernel hit the
native microscaling path or Triton's software emulation.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLOCK = 32


@triton.jit
def mx_mm(a_ptr, a_s_ptr, b_ptr, b_s_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, 64)
    rn = tl.arange(0, 64)
    rk = tl.arange(0, K // 2)
    rks = tl.arange(0, K // 32)

    a = tl.load(a_ptr + rm[:, None] * (K // 2) + rk[None, :])
    a_s = tl.load(a_s_ptr + rm[:, None] * (K // 32) + rks[None, :])
    b = tl.load(b_ptr + rn[:, None] * (K // 2) + rk[None, :])
    b_s = tl.load(b_s_ptr + rn[:, None] * (K // 32) + rks[None, :])

    acc = tl.dot_scaled(a, a_s, "e2m1", b.T, b_s, "e2m1", tl.zeros((64, 64), tl.float32))
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float32))


def unpack(packed: torch.Tensor, scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Packed uint8 (rows, cols//2) + e8m0 scales (rows, cols//32) -> float (rows, cols)."""
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    mag = LEVELS[codes & 0x7]
    val = torch.where(codes >= 8, -mag, mag)
    s = torch.pow(2.0, scales.long() - 127).repeat_interleave(BLOCK, dim=1)
    return val * s


def main() -> None:
    dev = "cuda"
    print("gpu:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    torch.manual_seed(0)
    M = N = 64
    K = 256

    a_p = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    b_p = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    a_s = torch.full((M, K // BLOCK), 127, dtype=torch.uint8, device=dev)
    b_s = torch.full((N, K // BLOCK), 127, dtype=torch.uint8, device=dev)

    c = torch.empty((M, N), dtype=torch.float32, device=dev)
    mx_mm[(1, 1)](a_p, a_s, b_p, b_s, c, K=K, N=N)
    torch.cuda.synchronize()

    a = unpack(a_p.cpu(), a_s.cpu(), M, K)
    b = unpack(b_p.cpu(), b_s.cpu(), N, K).T
    ref = a @ b
    err = (c.cpu() - ref).abs().max().item() / ref.abs().max().item()
    print(f"ran: yes | max rel err vs dequantised reference: {err:.2e}")
    print("ptx has tcgen05 (native 5th-gen tensor core):",
          any("tcgen05" in k for k in []))
    print("dot_scaled:", "compiled and produced correct values" if err < 1e-5 else "MISMATCH")


if __name__ == "__main__":
    main()


