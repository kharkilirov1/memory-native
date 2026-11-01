"""Matmuls with the deployed NVFP4 weights (E2M1 nibbles + E4M3 block scale per 16 along K + fp32 tensor scale).

Layout (the standard NVFP4 layout; also what torch.float4_e2m1fn_x2 / Triton "e2m1" expect):
    nibbles      uint8   [N, K/2]   element 2j in the LOW nibble, 2j+1 in the HIGH nibble of byte j;
                                    nibble = sign<<3 | mag_idx, mag = (0, .5, 1, 1.5, 2, 3, 4, 6)[mag_idx]
    block_scales uint8   [N, K/16]  E4M3 (float8_e4m3fn) bytes
    tensor_scale fp32    scalar     w[n, k] = e2m1(nib) * e4m3(byte[n, k//16]) * tensor_scale
`pack_fp4(...)` returns an `Fp4Weight`, which unpacks as the 3-tuple (nibbles, block_scales, tensor_scale).
A LatentGrid layer deploys RTN_E2M1(G[idx]) * s, i.e. exactly E2M1 * scale, so packing is lossless.

Backends (y = x @ W^T, dx = dy @ W; `fp4_matmul` / `fp4_matmul_dx`):
    reference       decode to bf16 (exact) + fp32 matmul of bf16-rounded x; the exact baseline (any device).
    triton_bf16     nibbles -> bf16 in registers inside a tiled tl.dot (bf16 tensor cores, fp32 accumulate,
                    tensor scale on the accumulator).  EXACT vs reference up to summation order.  fwd and dx.
    dot_scaled      tl.dot_scaled(bf16 x, e2m1 W with e4m3 block-16 scales).  Exact-x mode.  On sm_120 with
                    Triton 3.6 this is expected to be the software-upcast path (no native mixed bf16 x fp4 MMA).
    dot_scaled_fp4  tl.dot_scaled(e2m1 x, e2m1 W): x is quantized to NVFP4 on the fly (APPROXIMATION of x).
    torch_scaled_mm torch._scaled_mm on float4_e2m1fn_x2 with swizzled e4m3 scales (cuBLASLt NVFP4); x is
                    quantized to NVFP4 (APPROXIMATION of x).
dx with the fp4-activation backends additionally needs W^T in NVFP4 with blocks along N (the contraction axis),
which the LatentGrid scales (blocks along K) cannot express: W^T is RE-QUANTIZED (second approximation, cached
in `Fp4Weight.transposed()`).  `triton_bf16` dx uses the exact weights.

Importable without Triton / CUDA (kernels are only defined when Triton imports).
"""
from __future__ import annotations

from typing import Optional

import torch

from .packed_na import E4M3_MAX, e4m3_decode, e4m3_encode

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:                                    # pragma: no cover - CPU boxes
    HAS_TRITON = False
    triton = None
    tl = None

BLOCK = 16
E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
_INT32_MAX = 2 ** 31 - 1
BACKENDS = ("reference", "triton_bf16", "dot_scaled", "dot_scaled_fp4", "torch_scaled_mm")
#: backends whose x enters exactly (bf16-rounded) vs. quantized to NVFP4 (approximation mode)
FP4_ACT_BACKENDS = ("dot_scaled_fp4", "torch_scaled_mm")


class Fp4Unsupported(RuntimeError):
    """A backend (or a shape for a backend) is not available; message says why."""


def _jit(fn):
    return triton.jit(fn) if HAS_TRITON else fn


# ---------------------------------------------------------------------------------------------- packing

_MIDS: dict = {}


def _mids(device):
    key = str(device)
    if key not in _MIDS:
        _MIDS[key] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=device)
    return _MIDS[key]


def _mags(device):
    return torch.tensor(E2M1_MAGS, dtype=torch.float32, device=device)


def _e2m1_code(a: torch.Tensor) -> torch.Tensor:
    """fp32 values in block-scale units -> E2M1 nibble codes (int64 0..15), round to nearest, clamp at 6."""
    mag = a.abs()
    idx = torch.bucketize(mag, _mids(a.device))
    sign = ((a < 0) & (idx > 0)).to(torch.int64) << 3
    return idx | sign


def _nibbles_to_values(codes: torch.Tensor) -> torch.Tensor:
    """int codes 0..15 -> fp32 E2M1 values."""
    v = _mags(codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -v, v)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    lo, hi = codes[..., 0::2], codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _scale_full(scales: torch.Tensor, tensor_scale) -> torch.Tensor:
    """[R, C/16] fp32 actual block scale = e4m3(byte) * tensor_scale (same op order as LatentGridLinear)."""
    ts = torch.as_tensor(tensor_scale, dtype=torch.float32, device=scales.device)
    return e4m3_decode(scales) * ts


def unpack_fp4(nibbles: torch.Tensor, scales: torch.Tensor, tensor_scale, dtype=torch.float32) -> torch.Tensor:
    """-> dequantized weight [N, K] = e2m1 * e4m3 * tensor_scale (fp32 by default; bit-equal to the layer's)."""
    n, kh = nibbles.shape
    b = nibbles.to(torch.int64)
    codes = torch.stack([b & 15, b >> 4], dim=-1).reshape(n, kh * 2)
    val = _nibbles_to_values(codes)
    s = _scale_full(scales, tensor_scale)
    w = (val.reshape(n, -1, BLOCK) * s.unsqueeze(-1)).reshape(n, kh * 2)
    return w.to(dtype)


class Fp4Weight:
    """NVFP4 tensor: (nibbles, block_scales, tensor_scale) plus lazily built derived forms."""

    def __init__(self, nibbles, scales, tensor_scale):
        if nibbles.dtype != torch.uint8 or scales.dtype != torch.uint8:
            raise ValueError("nibbles and scales must be uint8")
        if nibbles.ndim != 2 or scales.shape != (nibbles.shape[0], nibbles.shape[1] * 2 // BLOCK):
            raise ValueError(f"shape mismatch: nibbles {tuple(nibbles.shape)}, scales {tuple(scales.shape)}")
        self.nibbles = nibbles.contiguous()
        self.scales = scales.contiguous()
        self.tensor_scale = torch.as_tensor(tensor_scale, dtype=torch.float32, device=nibbles.device).reshape(())
        self._cache: dict = {}

    def __iter__(self):
        return iter((self.nibbles, self.scales, self.tensor_scale))

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return (self.nibbles, self.scales, self.tensor_scale)[i]

    @property
    def shape(self):
        return (self.nibbles.shape[0], self.nibbles.shape[1] * 2)

    @property
    def device(self):
        return self.nibbles.device

    def to(self, device):
        return Fp4Weight(self.nibbles.to(device), self.scales.to(device), self.tensor_scale.to(device))

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        return unpack_fp4(self.nibbles, self.scales, self.tensor_scale, dtype)

    def dense_bf16(self) -> torch.Tensor:
        """Deployed weight in bf16 (tensor scale folded in: rounded, unlike the kernels) for the cuBLAS baseline."""
        if "bf16" not in self._cache:
            self._cache["bf16"] = self.dequantize(torch.bfloat16)
        return self._cache["bf16"]

    def transposed(self) -> "Fp4Weight":
        """NVFP4 of W^T with blocks along N (the dx contraction axis): a RE-QUANTIZATION (lossy)."""
        if "t" not in self._cache:
            if self.shape[0] % BLOCK:
                raise Fp4Unsupported(f"N={self.shape[0]} must be a multiple of {BLOCK} to quantize W^T along N")
            self._cache["t"] = quantize_act(self.dequantize().t().contiguous())
        return self._cache["t"]

    def swizzled_scales(self) -> torch.Tensor:
        """Scales in cuBLAS' SWIZZLE_32_4_4 layout, as float8_e4m3fn (cached)."""
        if "sw" not in self._cache:
            self._cache["sw"] = to_blocked(self.scales).view(torch.float8_e4m3fn)
        return self._cache["sw"]


def to_blocked(sc: torch.Tensor) -> torch.Tensor:
    """[R, C] scale matrix -> flat cuBLAS 128x4-tiled / 32_4_4-swizzled layout (zero padded to 128 x 4)."""
    rows, cols = sc.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    if (rb * 128, cb * 4) != (rows, cols):
        p = torch.zeros(rb * 128, cb * 4, dtype=sc.dtype, device=sc.device)
        p[:rows, :cols] = sc
        sc = p
    blocks = sc.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _pack_values(w: torch.Tensor, scale_bytes: torch.Tensor, tensor_scale) -> Fp4Weight:
    n, k = w.shape
    if k % BLOCK:
        raise ValueError(f"K={k} must be a multiple of {BLOCK}")
    s = _scale_full(scale_bytes, tensor_scale)
    a = (w.float().reshape(n, k // BLOCK, BLOCK) / s.unsqueeze(-1)).reshape(n, k)
    a = torch.nan_to_num(a, nan=0.0, posinf=E2M1_MAX, neginf=-E2M1_MAX)
    return Fp4Weight(_pack_codes(_e2m1_code(a)), scale_bytes.to(torch.uint8), tensor_scale)


def pack_fp4(source, scale_bytes: Optional[torch.Tensor] = None, tensor_scale=None) -> Fp4Weight:
    """Deployed weights -> NVFP4 layout.  Lossless when the values are E2M1 * e4m3(scale_bytes) * tensor_scale.

    source: a LatentGridLinear (uses its deployed rungs and frozen scales), or a dense [N, K] tensor of deployed
    values together with `scale_bytes` [N, K/16] uint8 and `tensor_scale` (if both are omitted the tensor is
    fitted + RTN-quantized with `quantize_nvfp4`).  Returns Fp4Weight == (nibbles, block_scales, tensor_scale).
    """
    if hasattr(source, "fp4_weight") and hasattr(source, "_scale"):          # LatentGridLinear: nibbles straight from idx
        return source.fp4_weight()
    if hasattr(source, "rung_index") and hasattr(source, "_scale"):
        w = source.weight().detach()
        return _pack_values(w, source._scale.detach(), source._tscale.detach())
    if scale_bytes is None and tensor_scale is None:
        return quantize_nvfp4(source)
    if scale_bytes is None or tensor_scale is None:
        raise ValueError("give both scale_bytes and tensor_scale, or neither")
    return _pack_values(source.detach(), scale_bytes, tensor_scale)


def quantize_nvfp4(x: torch.Tensor) -> Fp4Weight:
    """Dense [R, C] -> NVFP4 (tensor scale = amax / (6 * 448), E4M3 block scale = RTN(block_amax / (6 * ts)),
    values RTN to E2M1).  Used for activations / gradients (the approximation modes) and for W^T."""
    r, c = x.shape
    if c % BLOCK:
        raise ValueError(f"last dim {c} must be a multiple of {BLOCK}")
    xf = x.float()
    amax = xf.abs().amax()
    ts = torch.where(amax > 0, amax / (E2M1_MAX * E4M3_MAX), torch.ones_like(amax)).to(torch.float32)
    bamax = xf.abs().reshape(r, c // BLOCK, BLOCK).amax(-1)
    sb = e4m3_encode(bamax / (E2M1_MAX * ts))
    return _pack_values(xf, sb, ts)


def dequantize_nvfp4(q: Fp4Weight) -> torch.Tensor:
    return q.dequantize()


# ------------------------------------------------------------------------------------- Triton kernels

@_jit
def _dec_nibble(nib):
    """int32 nibble 0..15 -> fp32 E2M1 value, by bit arithmetic (no table, no gather)."""
    s = (nib >> 3) & 1
    e = (nib >> 1) & 3
    m = nib & 1
    bits = ((e + 126) << 23) | (m << 22)                  # (1 + m/2) * 2^(e-1), e >= 1
    bits = tl.where(e == 0, m * 0x3F000000, bits)    # e == 0: 0 or 0.5
    bits = bits | (s << 31)
    return bits.to(tl.float32, bitcast=True)


@_jit
def _dec_scale(sb):
    """int32 E4M3 byte (positive) -> fp32 scale, by bit arithmetic (as mx6.triton_kernel)."""
    e = (sb >> 3) & 15
    m = sb & 7
    sc = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    return tl.where(e == 0, m.to(tl.float32) * 0.001953125, sc)


@_jit
def _w_halves(b, sc, R: tl.constexpr, KP: tl.constexpr):
    """packed bytes [R, KP] (int32) + block scales [R, KP // 8] -> bf16 (lo, hi) tiles [R, KP].
    lo = elements 2j, hi = elements 2j+1 of the original K axis; value * scale is exact in bf16."""
    lo = _dec_nibble(b & 15)
    hi = _dec_nibble((b >> 4) & 15)
    lo = tl.reshape(tl.reshape(lo, [R, KP // 8, 8]) * sc[:, :, None], [R, KP])
    hi = tl.reshape(tl.reshape(hi, [R, KP // 8, 8]) * sc[:, :, None], [R, KP])
    return lo.to(tl.bfloat16), hi.to(tl.bfloat16)


def _cfg(bm, bn, bk, w, s):
    return triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)


def _fwd_configs():
    return [_cfg(*c) for c in [
        (16, 64, 128, 4, 3), (16, 128, 128, 4, 3), (32, 64, 128, 4, 3), (32, 128, 128, 4, 3),
        (64, 64, 128, 4, 3), (64, 128, 128, 8, 3), (64, 128, 256, 8, 3), (128, 64, 128, 4, 3),
        (128, 128, 64, 8, 3), (128, 128, 128, 8, 3), (128, 256, 64, 8, 3), (128, 128, 256, 8, 2),
        (64, 256, 128, 8, 3), (256, 128, 64, 8, 3),
    ]]


def _ds_configs():
    return [_cfg(*c) for c in [
        (16, 64, 128, 4, 3), (32, 64, 128, 4, 3), (32, 128, 128, 4, 3), (64, 64, 128, 4, 3),
        (64, 128, 128, 4, 3), (64, 128, 256, 8, 3), (128, 64, 128, 4, 3), (128, 128, 128, 8, 3),
        (128, 128, 256, 8, 3), (128, 256, 128, 8, 3), (128, 128, 64, 4, 4),
    ]]


def _autotune(configs, key):
    def deco(fn):
        return triton.autotune(configs=configs, key=key)(fn) if HAS_TRITON else fn
    return deco


@_autotune(_fwd_configs() if HAS_TRITON else None, ["M", "N", "K"])
@_jit
def _fwd_bf16_kernel(x_ptr, w_ptr, s_ptr, ts_ptr, out_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[M, N] = (x[M, K] bf16 @ decode(W)[N, K]^T) * tensor_scale.  W nibbles [N, K/2], scales [N, K/16]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    KH = K // 2
    NB = K // 16
    KP: tl.constexpr = BLOCK_K // 2
    KB: tl.constexpr = BLOCK_K // 16
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kp = k0 // 2 + tl.arange(0, KP)
        kb = k0 // 16 + tl.arange(0, KB)
        mk = kp < KH
        xp = x_ptr + offs_m[:, None] * K + 2 * kp[None, :]
        xm = mask_m[:, None] & mk[None, :]
        xe = tl.load(xp, mask=xm, other=0.0)
        xo = tl.load(xp + 1, mask=xm, other=0.0)
        b = tl.load(w_ptr + offs_n[:, None] * KH + kp[None, :], mask=mask_n[:, None] & mk[None, :],
                    other=0).to(tl.int32)
        sb = tl.load(s_ptr + offs_n[:, None] * NB + kb[None, :], mask=mask_n[:, None] & (kb < NB)[None, :],
                     other=0).to(tl.int32)
        wl, wh = _w_halves(b, _dec_scale(sb), BLOCK_N, KP)
        acc += tl.dot(xe.to(tl.bfloat16), tl.trans(wl))
        acc += tl.dot(xo.to(tl.bfloat16), tl.trans(wh))
    acc = acc * tl.load(ts_ptr)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


@_autotune(_fwd_configs() if HAS_TRITON else None, ["M", "N", "K"])
@_jit
def _dx_bf16_kernel(g_ptr, w_ptr, s_ptr, ts_ptr, out_ptr, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[M, K] = (g[M, N] bf16 @ decode(W)[N, K]) * tensor_scale; the K output columns are produced as
    even / odd halves (one accumulator each) and stored interleaved.  BLOCK_K = output columns per program."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    KH = K // 2
    NB = K // 16
    KP: tl.constexpr = BLOCK_K // 2
    KB: tl.constexpr = BLOCK_K // 16
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    kp = pid_k * KP + tl.arange(0, KP)
    kb = pid_k * KB + tl.arange(0, KB)
    mask_m = offs_m < M
    mk = kp < KH
    mkb = kb < NB
    acc_e = tl.zeros((BLOCK_M, KP), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, KP), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        mn = n < N
        g = tl.load(g_ptr + offs_m[:, None] * N + n[None, :], mask=mask_m[:, None] & mn[None, :], other=0.0)
        b = tl.load(w_ptr + n[:, None] * KH + kp[None, :], mask=mn[:, None] & mk[None, :], other=0).to(tl.int32)
        sb = tl.load(s_ptr + n[:, None] * NB + kb[None, :], mask=mn[:, None] & mkb[None, :], other=0).to(tl.int32)
        wl, wh = _w_halves(b, _dec_scale(sb), BLOCK_N, KP)
        g = g.to(tl.bfloat16)
        acc_e += tl.dot(g, wl)
        acc_o += tl.dot(g, wh)
    ts = tl.load(ts_ptr)
    op = out_ptr + offs_m[:, None] * K + 2 * kp[None, :]
    om = mask_m[:, None] & mk[None, :]
    tl.store(op, (acc_e * ts).to(out_ptr.dtype.element_ty), mask=om)
    tl.store(op + 1, (acc_o * ts).to(out_ptr.dtype.element_ty), mask=om)


@_autotune(_ds_configs() if HAS_TRITON else None, ["M", "N", "K", "X_FP4"])
@_jit
def _dot_scaled_kernel(x_ptr, xs_ptr, w_ptr, ws_ptr, ts_ptr, out_ptr, M, N, K,
                       X_FP4: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    """out[M, N] = dot_scaled(x, W^T) * ts.  W: nibbles [N, K/2], scales [N, K/16] (float8_e4m3fn pointer ->
    Triton selects the 16-element NVFP4 block).  X_FP4: x is nibbles [M, K/2] + e4m3 scales [M, K/16]
    ("e2m1" lhs), else x is bf16 [M, K] with no scale."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    KH = K // 2
    NB = K // 16
    KP: tl.constexpr = BLOCK_K // 2
    KB: tl.constexpr = BLOCK_K // 16
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kp = k0 // 2 + tl.arange(0, KP)
        kb = k0 // 16 + tl.arange(0, KB)
        mk = kp < KH
        mkb = kb < NB
        b = tl.load(w_ptr + offs_n[None, :] * KH + kp[:, None], mask=mask_n[None, :] & mk[:, None], other=0)
        b_s = tl.load(ws_ptr + offs_n[:, None] * NB + kb[None, :], mask=mask_n[:, None] & mkb[None, :], other=0)
        if X_FP4:
            a = tl.load(x_ptr + offs_m[:, None] * KH + kp[None, :], mask=mask_m[:, None] & mk[None, :], other=0)
            a_s = tl.load(xs_ptr + offs_m[:, None] * NB + kb[None, :], mask=mask_m[:, None] & mkb[None, :],
                          other=0)
            acc = tl.dot_scaled(a, a_s, "e2m1", b, b_s, "e2m1", acc)
        else:
            k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(x_ptr + offs_m[:, None] * K + k[None, :], mask=mask_m[:, None] & (k < K)[None, :],
                        other=0.0)
            acc = tl.dot_scaled(a, None, "bf16", b, b_s, "e2m1", acc)
    acc = acc * tl.load(ts_ptr)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


# ------------------------------------------------------------------- fused NVFP4 activation quantizer
#
# `quantize_nvfp4_fused(x)`: dense [R, C] (bf16 / fp16 / fp32, any strides) -> the operand `torch._scaled_mm` wants, in TWO
# launches (the eager `quantize_nvfp4` is ~15 elementwise / reduction kernels with fp32 intermediates):
#   1. amax: torch.linalg.vector_norm(x, inf, dtype=fp32) (one read of x, no temporaries)  -> tensor scale
#      ts = amax / (6 * 448)   (the same two torch ops as `quantize_nvfp4`: bit-identical ts)
#   2. `_quant_act_kernel`, ONE pass over x: per 16-element block amax -> E4M3 scale byte (RTNE, clamped to
#      [2^-9, 448]) -> x / (e4m3 * ts) (IEEE round-to-nearest division, tl.math.div_rn) -> E2M1 code by comparing |a| with
#      the 7 midpoints (ties go DOWN, = torch.bucketize(right=False) of the reference) -> two nibbles per byte (element
#      2j in the LOW nibble) -> stores nibbles [R, C/2], the plain scales [R, C/16] and the scales already in cuBLAS'
#      SWIZZLE_32_4_4 blocked layout (zero padded to 128 x 4 tiles) next to each other.
# Result: an `Fp4Weight` whose `.swizzled_scales()` is pre-filled, so `_torch_scaled_mm` runs with NO further torch ops.
# Same numbers as the reference quantizer BIT FOR BIT (tests/test_fp4_quant_fused.py on GPU; the CPU suite checks the
# swizzle index formula and the comparison coder through `emulate_quant_kernel`, a torch transcription). The
# quantization itself stays an APPROXIMATION of the activations (round-to-nearest NVFP4, see `act_quant_error`).

_FUSED = {"on": True, "error": None}


def set_fused_quant(on: bool) -> None:
    """Global switch for `quantize_act` (default on; a failing fused kernel switches itself off, error in _FUSED)."""
    _FUSED["on"] = bool(on)


def swizzle_index(rows: int, ncols: int, device=None) -> torch.Tensor:
    """int64 [rows, ncols]: flat position of scale (r, c) in `to_blocked` layout:
    ((r // 128) * ceil(ncols / 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4."""
    r = torch.arange(rows, device=device).unsqueeze(1)
    c = torch.arange(ncols, device=device).unsqueeze(0)
    cb = -(-ncols // 4)
    return ((r // 128) * cb + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4


def act_tensor_scale(x: torch.Tensor) -> torch.Tensor:
    """fp32 0-dim tensor scale of `quantize_nvfp4` (amax / (6 * 448); 1 for an all-zero tensor), without fp32 copies of x."""
    amax = torch.linalg.vector_norm(x, ord=float("inf"), dtype=torch.float32)
    return torch.where(amax > 0, amax / (E2M1_MAX * E4M3_MAX), torch.ones_like(amax)).to(torch.float32)


def _code_cmp(a: torch.Tensor) -> torch.Tensor:
    """Torch transcription of the kernel's E2M1 coder (comparisons with the midpoints instead of bucketize)."""
    m = a.abs()
    c = sum((m > t).to(torch.int64) for t in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    return c | (((a < 0) & (c > 0)).to(torch.int64) << 3)


def emulate_quant_kernel(x: torch.Tensor):
    """CPU-testable torch transcription of `_quant_act_kernel` (comparison coder, scatter into the swizzled layout).
    -> (nibbles [R, C/2], scales [R, C/16], swizzled flat uint8). Must equal `quantize_nvfp4` + `to_blocked` exactly."""
    r, c = x.shape
    ts = act_tensor_scale(x)
    xf = x.float()
    bam = xf.abs().reshape(r, c // BLOCK, BLOCK).amax(-1)
    sraw = (bam / (E2M1_MAX * ts)).clamp(2.0 ** -9, E4M3_MAX)
    sb = sraw.to(torch.float8_e4m3fn).view(torch.uint8)
    s = e4m3_decode(sb) * ts
    a = (xf.reshape(r, c // BLOCK, BLOCK) / s.unsqueeze(-1)).reshape(r, c)
    nib = _pack_codes(_code_cmp(a))
    rb, cb = -(-r // 128), -(-(c // BLOCK) // 4)
    sw = torch.zeros(rb * 128 * cb * 4, dtype=torch.uint8)
    sw[swizzle_index(r, c // BLOCK).reshape(-1)] = sb.reshape(-1)
    return nib, sb, sw


if HAS_TRITON:
    @triton.jit
    def _code_e2m1(a):
        """fp32 values in block-scale units -> int32 E2M1 codes (RTN, ties toward zero, sign bit only if mag > 0)."""
        m = tl.abs(a)
        c = ((m > 0.25).to(tl.int32) + (m > 0.75).to(tl.int32) + (m > 1.25).to(tl.int32) + (m > 1.75).to(tl.int32)
             + (m > 2.5).to(tl.int32) + (m > 3.5).to(tl.int32) + (m > 5.0).to(tl.int32))
        return c | tl.where((a < 0.0) & (c > 0), 8, 0)

    @triton.jit
    def _quant_act_kernel(x_ptr, ts_ptr, nib_ptr, sc_ptr, sw_ptr, R, C, stride_r, stride_c, n_cb4,
                          BR: tl.constexpr, BC: tl.constexpr):
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        CH = C // 2
        NB = C // 16
        KP: tl.constexpr = BC // 2
        NBT: tl.constexpr = BC // 16
        rows = pid_r * BR + tl.arange(0, BR)
        kp = pid_c * KP + tl.arange(0, KP)
        blk = pid_c * NBT + tl.arange(0, NBT)
        mr = rows < R
        m2 = mr[:, None] & (kp < CH)[None, :]
        base = x_ptr + rows[:, None] * stride_r
        xe = tl.load(base + (2 * kp)[None, :] * stride_c, mask=m2, other=0.0).to(tl.float32)
        xo = tl.load(base + (2 * kp + 1)[None, :] * stride_c, mask=m2, other=0.0).to(tl.float32)
        ts = tl.load(ts_ptr)
        xe3 = tl.reshape(xe, [BR, NBT, 8])
        xo3 = tl.reshape(xo, [BR, NBT, 8])
        bam = tl.max(tl.maximum(tl.abs(xe3), tl.abs(xo3)), axis=2)                 # [BR, NBT] block amax
        sraw = tl.math.div_rn(bam, 6.0 * ts)
        sraw = tl.minimum(tl.maximum(sraw, 0.001953125), 448.0)                    # [2^-9, 448]
        sb = sraw.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.uint8, bitcast=True)
        s = _dec_scale(sb.to(tl.int32)) * ts                                       # fp32 actual block scale
        s3 = tl.broadcast_to(s[:, :, None], [BR, NBT, 8])
        qe = tl.reshape(tl.math.div_rn(xe3, s3), [BR, KP])
        qo = tl.reshape(tl.math.div_rn(xo3, s3), [BR, KP])
        byte = _code_e2m1(qe) | (_code_e2m1(qo) << 4)
        tl.store(nib_ptr + rows[:, None] * CH + kp[None, :], byte.to(tl.uint8), mask=m2)
        ms = mr[:, None] & (blk < NB)[None, :]
        tl.store(sc_ptr + rows[:, None] * NB + blk[None, :], sb, mask=ms)
        r2 = rows[:, None]
        c2 = blk[None, :]
        sw_off = ((r2 // 128) * n_cb4 + c2 // 4) * 512 + (r2 % 32) * 16 + ((r2 % 128) // 32) * 4 + (c2 % 4)
        tl.store(sw_ptr + sw_off, sb, mask=ms)


def quantize_nvfp4_fused(x: torch.Tensor, br: int = 8, bc_max: int = 512) -> Fp4Weight:
    """Fused Triton NVFP4 quantizer (see the section comment). x: 2-D CUDA tensor, last dim a multiple of 16, any dtype
    and strides (a transposed view is read in place). Returns the same `Fp4Weight` as `quantize_nvfp4(x)` (bit-equal)
    with `swizzled_scales()` pre-filled. Raises Fp4Unsupported when Triton / CUDA / float8 are unavailable."""
    if not HAS_TRITON or not x.is_cuda or not hasattr(torch, "float8_e4m3fn"):
        raise Fp4Unsupported("quantize_nvfp4_fused: needs Triton + a CUDA tensor")
    if x.ndim != 2 or x.shape[1] % BLOCK:
        raise ValueError(f"expected a 2-D tensor with last dim a multiple of {BLOCK}, got {tuple(x.shape)}")
    r, c = x.shape
    if x.numel() > _INT32_MAX or max(abs(s) for s in x.stride()) * max(r, c) > _INT32_MAX:
        raise Fp4Unsupported("quantize_nvfp4_fused: tensor too large for int32 index math")
    ts = act_tensor_scale(x)
    nb = c // BLOCK
    rb, cb = -(-r // 128), -(-nb // 4)
    nib = torch.empty(r, c // 2, dtype=torch.uint8, device=x.device)
    sc = torch.empty(r, nb, dtype=torch.uint8, device=x.device)
    pad = (rb * 128 != r) or (cb * 4 != nb)
    sw = (torch.zeros if pad else torch.empty)(rb * 128 * cb * 4, dtype=torch.uint8, device=x.device)
    bc = 16
    while bc < min(bc_max, c):                                     # power of two >= min(bc_max, C), >= 16
        bc *= 2
    grid = (triton.cdiv(r, br), triton.cdiv(c, bc))
    _quant_act_kernel[grid](x, ts.reshape(1), nib, sc, sw, r, c, x.stride(0), x.stride(1), cb,
                            BR=br, BC=bc, num_warps=4)
    q = Fp4Weight(nib, sc, ts)
    q._cache["sw"] = sw.view(torch.float8_e4m3fn)
    return q


def quantize_act(x: torch.Tensor) -> Fp4Weight:
    """The quantizer used by the fp4-activation backends: the fused kernel on CUDA (falls back to the eager reference,
    permanently, if it ever fails to compile / launch; the reason is kept in `_FUSED['error']`), else `quantize_nvfp4`."""
    if _FUSED["on"] and HAS_TRITON and x.is_cuda and x.ndim == 2 and x.shape[1] % BLOCK == 0:
        try:
            return quantize_nvfp4_fused(x)
        except Fp4Unsupported:
            pass
        except Exception as e:                                       # noqa: BLE001 - kernel compile / launch failure
            _FUSED["on"], _FUSED["error"] = False, f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    return quantize_nvfp4(x)


def act_quant_error(x: torch.Tensor) -> dict:
    """Error of the NVFP4 round trip of x (the activation / gradient approximation): relative RMS error
    ||deq - x|| / ||x||, max |err| / max |x|, and the fraction of exactly-zeroed elements (underflow to code 0)."""
    xf = x.float()
    d = quantize_act(x).dequantize() - xf
    n = xf.norm().clamp_min(1e-30)
    return {"rel_rms": float(d.norm() / n), "rel_max": float(d.abs().max() / xf.abs().max().clamp_min(1e-30)),
            "zero_frac": float((quantize_act(x).dequantize() == 0).float().mean())}


# ------------------------------------------------------------------------------------------ references

def reference_matmul(x: torch.Tensor, w: Fp4Weight, out_dtype=torch.float32) -> torch.Tensor:
    """Exact baseline: x rounded to bf16, weight decoded (exact), product accumulated in float64."""
    xb = x.to(torch.bfloat16).double()
    return (xb @ w.dequantize().double().t()).to(out_dtype)


def reference_dx(dy: torch.Tensor, w: Fp4Weight, out_dtype=torch.float32) -> torch.Tensor:
    return (dy.to(torch.bfloat16).double() @ w.dequantize().double()).to(out_dtype)


def expected_matmul(x: torch.Tensor, w: Fp4Weight, backend: str, out_dtype=torch.float32) -> torch.Tensor:
    """What `backend` is supposed to compute, in float64: exact-x backends -> reference_matmul; fp4-activation
    backends -> the product with the NVFP4-quantized x (so the comparison isolates kernel error)."""
    if backend in FP4_ACT_BACKENDS:
        return (quantize_nvfp4(x).dequantize().double() @ w.dequantize().double().t()).to(out_dtype)
    return reference_matmul(x, w, out_dtype)


def expected_dx(dy: torch.Tensor, w: Fp4Weight, backend: str, out_dtype=torch.float32) -> torch.Tensor:
    """Same for dx.  dot_scaled* / torch_scaled_mm use the re-quantized W^T (blocks along N)."""
    if backend in ("reference", "triton_bf16"):
        return reference_dx(dy, w, out_dtype)
    wt = w.transposed().dequantize().double()                       # [K, N]
    if backend in FP4_ACT_BACKENDS:
        return (quantize_nvfp4(dy).dequantize().double() @ wt.t()).to(out_dtype)
    return (dy.to(torch.bfloat16).double() @ wt.t()).to(out_dtype)


# --------------------------------------------------------------------------------------- host launchers

def _need_triton(t: torch.Tensor, what: str) -> None:
    if not HAS_TRITON:
        raise Fp4Unsupported(f"{what}: Triton is not installed")
    if not t.is_cuda:
        raise Fp4Unsupported(f"{what}: needs a CUDA tensor")


def _check(x: torch.Tensor, w: Fp4Weight, kdim: int, what: str) -> None:
    if x.ndim != 2 or x.shape[1] != kdim:
        raise ValueError(f"{what}: expected a 2-D tensor with last dim {kdim}, got {tuple(x.shape)}")
    if max(x.numel(), w.nibbles.numel() * 2) > _INT32_MAX:
        raise ValueError(f"{what}: tensor too large for int32 index math")


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).contiguous()


def _triton_bf16_fwd(x, w, out_dtype):
    _need_triton(x, "triton_bf16")
    n, k = w.shape
    _check(x, w, k, "triton_bf16")
    x = _bf16(x)
    m = x.shape[0]
    out = torch.empty(m, n, device=x.device, dtype=out_dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _fwd_bf16_kernel[grid](x, w.nibbles, w.scales, w.tensor_scale.reshape(1), out, m, n, k)
    return out


def _triton_bf16_dx(dy, w, out_dtype):
    _need_triton(dy, "triton_bf16 dx")
    n, k = w.shape
    _check(dy, w, n, "triton_bf16 dx")
    dy = _bf16(dy)
    m = dy.shape[0]
    out = torch.empty(m, k, device=dy.device, dtype=out_dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(k, meta["BLOCK_K"]))
    _dx_bf16_kernel[grid](dy, w.nibbles, w.scales, w.tensor_scale.reshape(1), out, m, n, k)
    return out


def _dot_scaled(x, w, out_dtype, fp4_x):
    _need_triton(x if not isinstance(x, Fp4Weight) else x.nibbles, "dot_scaled")
    if not hasattr(torch, "float8_e4m3fn"):
        raise Fp4Unsupported("dot_scaled: torch has no float8_e4m3fn")
    n, k = w.shape
    if fp4_x:
        q = x if isinstance(x, Fp4Weight) else quantize_act(x)
        if q.shape[1] != k:
            raise ValueError(f"dot_scaled: x has K={q.shape[1]}, W has K={k}")
        m = q.shape[0]
        xa, xs = q.nibbles, q.scales.view(torch.float8_e4m3fn)
        ts = (q.tensor_scale * w.tensor_scale).reshape(1)
    else:
        _check(x, w, k, "dot_scaled")
        xa, m = _bf16(x), x.shape[0]
        xs = w.scales.view(torch.float8_e4m3fn)                       # unused dummy pointer
        ts = w.tensor_scale.reshape(1)
    out = torch.empty(m, n, device=w.device, dtype=out_dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _dot_scaled_kernel[grid](xa, xs, w.nibbles, w.scales.view(torch.float8_e4m3fn), ts, out, m, n, k,
                             X_FP4=1 if fp4_x else 0)
    return out


def _torch_scaled_mm(x, w, out_dtype):
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
        raise Fp4Unsupported("torch_scaled_mm: torch has no float4_e2m1fn_x2 / _scaled_mm")
    if not w.device.type == "cuda":
        raise Fp4Unsupported("torch_scaled_mm: needs CUDA")
    q = x if isinstance(x, Fp4Weight) else quantize_act(x)
    n, k = w.shape
    if q.shape[1] != k:
        raise ValueError(f"torch_scaled_mm: x has K={q.shape[1]}, W has K={k}")
    try:
        y = torch._scaled_mm(q.nibbles.view(torch.float4_e2m1fn_x2),
                             w.nibbles.view(torch.float4_e2m1fn_x2).t(),
                             q.swizzled_scales(), w.swizzled_scales(), out_dtype=torch.bfloat16)
    except (RuntimeError, TypeError, NotImplementedError) as e:
        raise Fp4Unsupported(f"torch_scaled_mm: {type(e).__name__}: {str(e).splitlines()[0][:300]}") from e
    y = y * (q.tensor_scale * w.tensor_scale)                       # bf16 * fp32 0-dim -> bf16
    return y.to(out_dtype)


def _flat(x):
    return x.reshape(-1, x.shape[-1]) if not isinstance(x, Fp4Weight) else x


def fp4_matmul(x, w: Fp4Weight, backend: str = "triton_bf16", out_dtype=torch.float32) -> torch.Tensor:
    """y = x @ W^T, x [..., K] -> [..., N].  `x` may be an Fp4Weight (pre-quantized) for the fp4-x backends."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    lead = None if isinstance(x, Fp4Weight) else x.shape[:-1]
    x2 = _flat(x)
    if backend == "reference":
        y = reference_matmul(x2, w, out_dtype)
    elif backend == "triton_bf16":
        y = _triton_bf16_fwd(x2, w, out_dtype)
    elif backend == "dot_scaled":
        y = _dot_scaled(x2, w, out_dtype, False)
    elif backend == "dot_scaled_fp4":
        y = _dot_scaled(x2, w, out_dtype, True)
    else:
        y = _torch_scaled_mm(x2, w, out_dtype)
    return y if lead is None else y.reshape(*lead, w.shape[0])


def fp4_matmul_dx(dy, w: Fp4Weight, backend: str = "triton_bf16", out_dtype=torch.float32) -> torch.Tensor:
    """dx = dy @ W, dy [..., N] -> [..., K].  dot_scaled* / torch_scaled_mm use the re-quantized W^T."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    lead = None if isinstance(dy, Fp4Weight) else dy.shape[:-1]
    d2 = _flat(dy)
    if backend == "reference":
        g = reference_dx(d2, w, out_dtype)
    elif backend == "triton_bf16":
        g = _triton_bf16_dx(d2, w, out_dtype)
    elif backend == "dot_scaled":
        g = _dot_scaled(d2, w.transposed(), out_dtype, False)
    elif backend == "dot_scaled_fp4":
        g = _dot_scaled(d2, w.transposed(), out_dtype, True)
    else:
        g = _torch_scaled_mm(d2, w.transposed(), out_dtype)
    return g if lead is None else g.reshape(*lead, w.shape[1])


# ------------------------------------------------------------------------------------------ availability

_STATUS: dict = {}


def backend_status(backend: str, device="cuda") -> tuple[bool, str]:
    """(usable, reason).  Runs a tiny problem once per (backend, device) and compares with the expected result;
    a backend that compiles but returns wrong numbers is reported unusable (fail closed)."""
    if backend == "reference":
        return True, "ok"
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return False, "no CUDA device"
    key = (backend, dev.index if dev.index is not None else torch.cuda.current_device())
    if key in _STATUS:
        return _STATUS[key]
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        m, n, k = 128, 128, 256
        wd = torch.randn(n, k, generator=g).to(dev)
        w = quantize_nvfp4(wd)
        x = torch.randn(m, k, generator=g).to(dev)
        y = fp4_matmul(x, w, backend).double()
        ref = expected_matmul(x, w, backend)
        rel = ((y - ref).abs().max() / ref.abs().max()).item()
        if not rel < 5e-3:
            res = (False, f"wrong numbers on probe (max|err|/max|ref| = {rel:.3g})")
        else:
            res = (True, "ok")
        if res[0] and backend in ("triton_bf16",):
            d = torch.randn(m, n, generator=g).to(dev)
            e = (fp4_matmul_dx(d, w, backend).double() - expected_dx(d, w, backend)).abs().max()
            if not (e / expected_dx(d, w, backend).abs().max()).item() < 5e-3:
                res = (False, "dx wrong numbers on probe")
    except Exception as e:                                           # noqa: BLE001 - report, never raise
        res = (False, f"{type(e).__name__}: {str(e).strip().splitlines()[0][:300] if str(e).strip() else ''}")
    _STATUS[key] = res
    return res
