"""Triton fused decode+matmul kernels for PackedNA formats.

Decodes codes on the fly inside the GEMM (no dense weight in HBM).

    from mx6.triton_kernel import packed_matmul, packed_matmul_dx
    out = packed_matmul(x, codes, scale_bytes, code_to_value)          # bf16 tensor cores
    out = packed_matmul(x, codes, scale_bytes, code_to_value,
                        compute_dtype="fp32-ieee")                     # legacy parity mode

compute_dtype
    "bf16"       (default) x cast to bf16, decoded weight in bf16, fp32 accumulate,
                 tensor cores.  The decoded weight is EXACT in bf16 (see below), so
                 the only error vs. an fp32 matmul is the rounding of x to bf16.
    "tf32"       fp32 tiles, tf32 tensor cores (x rounded to 10 mantissa bits).
    "fp32-ieee"  fp32 tiles, full-precision FFMA dot (no tensor cores): the legacy
                 behaviour, matches the fp32 torch reference to ~1e-4.

bf16 exactness of the decoded weight.  Ladders in use (packed_na.py):
    PackedNA5 signed values {0, +-0.5, +-3}        = {0, +-2^-1, +-3*2^0}
    PackedNA6 signed values {0, +-0.5, +-1, +-6}   = {0, +-2^-1, +-2^0, +-3*2^1}
Every nonzero value is m * 2^e with m in {1, 3} (<= 2 significant bits); bf16
keeps 8.  The block scale is 2^(byte-127), a pure exponent shift, so
value*scale = m * 2^j is exactly representable in bf16 whenever it is finite and
j >= -133 (bf16 subnormal floor).  Smallest case: 0.5 * 2^-127 = 2^-128 (fine).
Overflow (value*scale >= 2^128, e.g. 6 * 2^127 at byte 254) hits fp32 and the
torch reference identically; pack.py clamps scales far below that.
Products x_bf16 * w_bf16 have <= 10 significant bits, exact in fp32, so the only
difference vs. an fp32 reference fed the same bf16-rounded x is summation order.

The ``code_to_value`` table keeps its old contract ([n_codes] float32, constant
runs of ``acc_levels`` codes per signed value).  On the host the (signed ladder,
acc_levels) pair is recovered from it (cached) so the kernel decodes with
``mag = code // ACC`` and a where-chain over <= 16 scalar values: no per-element
global gather.  The E8M0 scale is converted once per 32-element block by building
the fp32 bit pattern, then broadcast over the block.

NVFP4 mode (SCALE_FMT=1, opt-in; E8M0 path above is untouched).  Scale = one FP8 E4M3
byte per 16 elements (decoded in-kernel by bit arithmetic, no table, no fp8 dtype
needed) times ONE fp32 per-tensor scale.  Exactness argument, stated honestly:

  * ladder value x E4M3 block scale is EXACT in bf16 (and tf32/fp32): the ladder
    values are m * 2^e with m in {1, 3} (<= 2 significant bits) and an E4M3 scale is
    (8+m3)/8 * 2^k (<= 4 significant bits), so the product has <= 6 significant bits
    (3 * 15 = 45 < 64), within bf16 range (smallest 0.5*2^-9, largest 6*448).
  * the PER-TENSOR fp32 scale is NOT exact in bf16 (arbitrary 24-bit mantissa).  So it
    is never multiplied into the weight tile: the tile stays ladder*e4m3 (exact), and
    the single per-tensor factor is applied in fp32 to the fp32 accumulator in the
    epilogue.  out = (x @ (ladder*e4m3).T) * tensor_scale.  Because the factor is
    common to every term of a dot product this equals the reference up to one fp32
    rounding of the final product (relative 2^-24), not a bf16 rounding.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
    triton = None
    tl = None

BLOCK = 32
NVFP4_BLOCK = 16
_SCALE_FMTS = {"e8m0": 0, "nvfp4": 1}
_MAX_VALS = 16
_INT32_MAX = 2 ** 31 - 1

# compute modes (constexpr MODE in kernels)
_MODE_FP32_IEEE = 0
_MODE_TF32 = 1
_MODE_BF16 = 2
_MODES = {"fp32-ieee": _MODE_FP32_IEEE, "tf32": _MODE_TF32, "bf16": _MODE_BF16}
_PRECISION_TO_DTYPE = {"ieee": "fp32-ieee", "tf32": "tf32"}


def _jit(fn):
    """``triton.jit`` when available, else a passthrough (CPU import safety)."""
    return triton.jit(fn) if HAS_TRITON else fn


# ── host side: recover (ladder, acc_levels) from the code->value table ───────

_LADDER_CACHE: dict = {}


def derive_ladder(code_to_value: torch.Tensor):
    """Return (signed_values: list[float], acc_levels: int) from a code->value LUT.

    The LUT is ``signed[c // acc_levels]``: constant runs of equal length.
    """
    vals = [float(v) for v in code_to_value.detach().cpu().tolist()]
    n = len(vals)
    acc = 1
    while acc < n and vals[acc] == vals[0]:
        acc += 1
    if n % acc != 0:
        raise ValueError("code_to_value is not a run-length table (n_codes % acc_levels != 0)")
    signed = vals[::acc]
    for c, v in enumerate(vals):
        if v != signed[c // acc]:
            raise ValueError("code_to_value is not constant over acc_levels-sized runs")
    if len(signed) > _MAX_VALS:
        raise ValueError(f"ladder has {len(signed)} values, max {_MAX_VALS}")
    return signed, acc


def _ladder_for(code_to_value: torch.Tensor):
    """Cached (ladder tensor [n_vals] fp32 on device, acc_levels, n_vals).

    Key: (device, data_ptr, n_codes, _version).  A fresh ``.to(device)`` copy
    simply re-derives (one tiny host read).
    """
    key = (str(code_to_value.device), code_to_value.data_ptr(),
           code_to_value.numel(), code_to_value._version)
    hit = _LADDER_CACHE.get(key)
    if hit is not None:
        return hit
    signed, acc = derive_ladder(code_to_value)
    t = torch.tensor(signed, dtype=torch.float32, device=code_to_value.device)
    out = (t, acc, len(signed))
    if len(_LADDER_CACHE) > 64:
        _LADDER_CACHE.clear()
    _LADDER_CACHE[key] = out
    return out


def _resolve_mode(compute_dtype: Optional[str], input_precision: Optional[str]) -> int:
    if compute_dtype is None:
        if input_precision is None:
            compute_dtype = "bf16"
        elif input_precision in _PRECISION_TO_DTYPE:
            compute_dtype = _PRECISION_TO_DTYPE[input_precision]
        else:
            raise ValueError(f"unsupported input_precision {input_precision!r}")
    if compute_dtype not in _MODES:
        raise ValueError(f"compute_dtype must be one of {sorted(_MODES)}, got {compute_dtype!r}")
    return _MODES[compute_dtype]


# ── device side ──────────────────────────────────────────────────────────────

@_jit
def _decode_w(codes_u8, scale_ptrs, scale_mask, ladder_ptr,
              ACC: tl.constexpr, N_VALS: tl.constexpr,
              R: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr,
              SCALE_FMT: tl.constexpr = 0):
    """codes [R, C] uint8 + scale bytes -> fp32 weight tile [R, C].

    Scale pointers/mask have shape [R, C // BLOCK] (one scale byte per block: E8M0 per
    32 elements, or E4M3 per 16 when SCALE_FMT == 1).  For SCALE_FMT == 1 the tile
    holds ladder * e4m3 only; the fp32 per-tensor scale is applied to the accumulator.
    """
    mag = codes_u8.to(tl.int32) // ACC
    val = tl.zeros([R, C], dtype=tl.float32)
    for i in tl.static_range(N_VALS):
        vi = tl.load(ladder_ptr + i)            # scalar, L1/const cached
        val = tl.where(mag == i, vi, val)
    if SCALE_FMT == 1:
        sb = tl.load(scale_ptrs, mask=scale_mask, other=56).to(tl.int32)   # 56 = 0x38 = 1.0
        e = (sb >> 3) & 15
        m = sb & 7
        # normal: (8+m)/8 * 2^(e-7) -> fp32 bits ((e-7+127) << 23) | (m << 20)
        sc = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        # subnormal (e == 0): m * 2^-9
        sc = tl.where(e == 0, m.to(tl.float32) * 0.001953125, sc)
    else:
        sb = tl.load(scale_ptrs, mask=scale_mask, other=127)
        # 2^(byte-127): the fp32 exponent field is the byte; byte 0 is 2^-127 (subnormal)
        sc = (sb.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        sc = tl.where(sb == 0, 5.877471754111438e-39, sc)
    w = tl.reshape(val, [R, C // BLOCK, BLOCK]) * sc[:, :, None]
    return tl.reshape(w, [R, C])


def _configs():
    cfgs = []
    for bm, bn, bk, w, s in [
        (16, 32, 128, 4, 3), (16, 64, 128, 4, 3), (16, 64, 64, 4, 4),
        (16, 128, 64, 4, 3), (32, 64, 128, 4, 3), (32, 128, 64, 4, 3),
        (64, 64, 64, 4, 3), (64, 128, 64, 8, 3), (64, 128, 128, 8, 3),
        (128, 64, 64, 4, 3), (128, 128, 64, 8, 3), (128, 128, 32, 8, 4),
    ]:
        cfgs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                  num_warps=w, num_stages=s))
    return cfgs


def _prune(configs, named_args, **kwargs):
    """fp32 modes: drop big tiles (shared-memory / register blowup)."""
    if named_args.get("MODE", _MODE_BF16) == _MODE_BF16:
        return configs
    keep = [c for c in configs
            if c.kwargs["BLOCK_K"] <= 64 and c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"] <= 64 * 128]
    return keep or configs


def _autotune(fn):
    if not HAS_TRITON:
        return fn
    return triton.autotune(configs=_configs(), key=["M", "N", "K", "MODE", "SCALE_FMT"],
                           prune_configs_by={"early_config_prune": _prune})(fn)


@_autotune
@_jit
def _packed_matmul_kernel(
    x_ptr,          # [M, K] float32 / bf16
    codes_ptr,      # [N, K] uint8
    scale_ptr,      # [N, K // 32] uint8 (E8M0)
    ladder_ptr,     # [N_VALS] float32 signed ladder
    out_ptr,        # [M, N] float32
    tscale_ptr,     # [1] float32 per-tensor scale (read only if SCALE_FMT == 1)
    M, N, K,
    ACC: tl.constexpr, N_VALS: tl.constexpr, MODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr, SCALE_FMT: tl.constexpr,
):
    """out = x @ decode(codes, scale).T  (int32 index math; host checks sizes)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    NB = K // BLOCK
    KB: tl.constexpr = BLOCK_K // BLOCK

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_kb = tl.arange(0, KB)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        mask_k = k < K
        x = tl.load(x_ptr + offs_m[:, None] * K + k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        codes = tl.load(codes_ptr + offs_n[:, None] * K + k[None, :],
                        mask=mask_n[:, None] & mask_k[None, :], other=0)
        kb = k_start // BLOCK + offs_kb
        w = _decode_w(codes,
                      scale_ptr + offs_n[:, None] * NB + kb[None, :],
                      mask_n[:, None] & (kb < NB)[None, :],
                      ladder_ptr, ACC, N_VALS, BLOCK_N, BLOCK_K, BLOCK, SCALE_FMT)
        # OOB k: x is 0 there and w is finite, so the product is 0.
        if MODE == 2:
            acc += tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        elif MODE == 1:
            acc += tl.dot(x.to(tl.float32), tl.trans(w), input_precision="tf32")
        else:
            acc += tl.dot(x.to(tl.float32), tl.trans(w), input_precision="ieee")

    if SCALE_FMT == 1:
        acc = acc * tl.load(tscale_ptr)        # fp32 per-tensor scale, on the accumulator
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=mask_m[:, None] & mask_n[None, :])


@_autotune
@_jit
def _packed_matmul_dx_kernel(
    go_ptr,         # [M, N] float32 / bf16 — upstream gradient
    codes_ptr,      # [N, K] uint8
    scale_ptr,      # [N, K // 32] uint8
    ladder_ptr,     # [N_VALS] float32
    out_ptr,        # [M, K] float32
    tscale_ptr,     # [1] float32 per-tensor scale (read only if SCALE_FMT == 1)
    M, N, K,
    ACC: tl.constexpr, N_VALS: tl.constexpr, MODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr, SCALE_FMT: tl.constexpr,
):
    """out = go @ decode(codes, scale)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    NB = K // BLOCK
    KB: tl.constexpr = BLOCK_K // BLOCK

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    kb = pid_k * KB + tl.arange(0, KB)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask_kb = kb < NB

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n = n_start + tl.arange(0, BLOCK_N)
        mask_n = n < N
        go = tl.load(go_ptr + offs_m[:, None] * N + n[None, :],
                     mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        codes = tl.load(codes_ptr + n[:, None] * K + offs_k[None, :],
                        mask=mask_n[:, None] & mask_k[None, :], other=0)
        w = _decode_w(codes,
                      scale_ptr + n[:, None] * NB + kb[None, :],
                      mask_n[:, None] & mask_kb[None, :],
                      ladder_ptr, ACC, N_VALS, BLOCK_N, BLOCK_K, BLOCK, SCALE_FMT)
        # OOB n: go is 0 there and w is finite -> contributes 0.
        if MODE == 2:
            acc += tl.dot(go.to(tl.bfloat16), w.to(tl.bfloat16))
        elif MODE == 1:
            acc += tl.dot(go.to(tl.float32), w, input_precision="tf32")
        else:
            acc += tl.dot(go.to(tl.float32), w, input_precision="ieee")

    if SCALE_FMT == 1:
        acc = acc * tl.load(tscale_ptr)        # fp32 per-tensor scale, on the accumulator
    tl.store(out_ptr + offs_m[:, None] * K + offs_k[None, :], acc,
             mask=mask_m[:, None] & mask_k[None, :])


def _resolve_scale_fmt(scale_format, block, tensor_scale, ref):
    """-> (SCALE_FMT constexpr, block, tensor-scale pointer tensor [1] fp32)."""
    if scale_format not in _SCALE_FMTS:
        raise ValueError(f"scale_format must be one of {sorted(_SCALE_FMTS)}, got {scale_format!r}")
    fmt = _SCALE_FMTS[scale_format]
    want = NVFP4_BLOCK if fmt else BLOCK
    block = want if block is None else block
    assert block == want, f"{scale_format} kernel is specialised to {want}-element blocks"
    if fmt:
        if tensor_scale is None:
            raise ValueError("scale_format='nvfp4' needs tensor_scale (fp32 per-tensor scale)")
        ts = tensor_scale.detach().to(device=ref.device, dtype=torch.float32).reshape(1).contiguous()
    else:
        ts = torch.zeros(1, device=ref.device, dtype=torch.float32)   # unused dummy
    return fmt, block, ts


def _check_sizes(*tensors):
    for t in tensors:
        if t.numel() > _INT32_MAX:
            raise ValueError("tensor too large for int32 index math in the fused kernel")


def _check_codes_dtype(codes: torch.Tensor) -> None:
    """The kernels are written and tested for uint8 code words (<= 256 codes per format).

    A format with more codes (PackedNA.code_dtype == int16, e.g. the full 15-rung E2M1 ladder at
    C >= 10) is stored in int16 and is NOT run through this kernel: refuse instead of guessing.
    """
    if codes.dtype != torch.uint8:
        raise ValueError(f"the fused kernel reads uint8 codes (formats with <= 256 codes), got "
                         f"{codes.dtype}; use the unfused path (fused='off') for larger formats")


def packed_matmul(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor,
                  code_to_value: torch.Tensor, block: Optional[int] = None,
                  input_precision: Optional[str] = None,
                  compute_dtype: Optional[str] = None,
                  scale_format: str = "e8m0",
                  tensor_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fused decode + matmul: out = x @ decode(codes, scale).T

    Args:
        x: [M, K] float32 (bf16 also accepted)
        codes: [N, K] uint8 — packed (value, accumulator) codes
        scale: [N, K // block] uint8 — E8M0 block scales
        code_to_value: [n_codes] float32 — code -> signed magnitude LUT
        block: block size; default 32 (e8m0) / 16 (nvfp4), anything else is rejected
        compute_dtype: "bf16" (default) | "tf32" | "fp32-ieee"
        input_precision: legacy alias ("ieee" -> fp32-ieee, "tf32" -> tf32);
            used only when compute_dtype is None.
        scale_format: "e8m0" (default; scale = 2^(byte-127)) | "nvfp4" (scale = e4m3(byte),
            then ONE fp32 `tensor_scale` multiplies the fp32 accumulator, see module doc)
        tensor_scale: fp32 scalar tensor, required for nvfp4

    Returns:
        out: [M, N] float32
    """
    _check_codes_dtype(codes)
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    mode = _resolve_mode(compute_dtype, input_precision)
    sfmt, block, ts = _resolve_scale_fmt(scale_format, block, tensor_scale, x)
    M, K = x.shape
    N, K2 = codes.shape
    assert K == K2, f"K mismatch: x has {K}, codes has {K2}"
    assert K % block == 0
    assert scale.shape == (N, K // block)
    _check_sizes(x, codes, scale)

    x = x.contiguous()
    codes = codes.contiguous()
    scale = scale.contiguous()
    ladder, acc_levels, n_vals = _ladder_for(code_to_value)

    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    _check_sizes(out)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _packed_matmul_kernel[grid](
        x, codes, scale, ladder, out, ts, M, N, K,
        ACC=acc_levels, N_VALS=n_vals, MODE=mode, BLOCK=block, SCALE_FMT=sfmt,
    )
    return out


def packed_matmul_dx(grad_out: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor,
                     code_to_value: torch.Tensor, block: Optional[int] = None,
                     input_precision: Optional[str] = None,
                     compute_dtype: Optional[str] = None,
                     scale_format: str = "e8m0",
                     tensor_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fused decode + grad_x: out = grad_out @ decode(codes, scale)."""
    _check_codes_dtype(codes)
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    mode = _resolve_mode(compute_dtype, input_precision)
    sfmt, block, ts = _resolve_scale_fmt(scale_format, block, tensor_scale, grad_out)
    M, N = grad_out.shape
    N2, K = codes.shape
    assert N == N2, f"N mismatch: grad_out has {N}, codes has {N2}"
    assert K % block == 0
    assert scale.shape == (N, K // block)
    _check_sizes(grad_out, codes, scale)

    grad_out = grad_out.contiguous()
    codes = codes.contiguous()
    scale = scale.contiguous()
    ladder, acc_levels, n_vals = _ladder_for(code_to_value)

    out = torch.empty(M, K, device=grad_out.device, dtype=torch.float32)
    _check_sizes(out)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_K"]))
    _packed_matmul_dx_kernel[grid](
        grad_out, codes, scale, ladder, out, ts, M, N, K,
        ACC=acc_levels, N_VALS=n_vals, MODE=mode, BLOCK=block, SCALE_FMT=sfmt,
    )
    return out


def best_config(which: str = "fwd"):
    """Autotune winner of the last launch ('fwd' or 'dx'), or None."""
    k = _packed_matmul_kernel if which == "fwd" else _packed_matmul_dx_kernel
    return getattr(k, "best_config", None)
