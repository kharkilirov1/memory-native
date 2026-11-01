"""Fused Triton update (and bf16/fp32 decode) for `LatentGridLinear`; CPU-importable (Triton optional).

Update kernel `_lg_update_kernel`: ONE launch per layer, ONE program per weight row, two passes over the row:
  pass 1  g_sq = sum(g^2) / cols  ->  v_new = beta*v + (1-beta)*g_sq  (stored in place), denom = max(sqrt(v_new), eps)
  pass 2  (the row of g is re-read from L2: rows are 1-32 KB, so HBM sees grad once)
          i      = idx (uint8/int16)            s = e4m3(byte[col>>4]) * tscale   (bit arithmetic, no table)
          w0     = G(i)                          closed form: |i-7m| = j*m + r, j = gap index 0..6 of the half-grid
          g_hat  = g / denom  [+ wd * w0]        delta = clamp(-lr*g_hat / s, -16, 16), NaN -> 0
          u      = i + (U(w0+delta) - U(w0))     U closed form, odd, 3 linear pieces (|x|<2: slope 2m, <4: m, else m/2)
          u      = clamp(u, 0, n-1);  idx' = floor(u) + [rand < u - floor(u)]   (tl.rand = Philox-10, (seed, row*cols+col))
  Why row-per-program and not a separate reduction kernel: the row reduction needs the whole row before any
  element can move; with >= 512 rows (out_features) there are enough programs to fill the GPU, the second read of g
  hits L2, and it saves a launch + a global v/denominator round trip. (If out_features < ~#SM the GPU is under-filled;
  then a split-row variant would be needed - not implemented, not expected for transformer layers.)
  Bytes/param: grad 4 (fp32) or 2 (bf16) read from HBM (+ L2 re-read), idx 1 read + 1 write, scale byte 1/16,
  so 6 B (fp32 grad) / 4 B (bf16 grad) + ~0.07; the eager torch path is ~465.
  Optional clip telemetry: per-row count of requests beyond the grid (one int32 per row), reduced by the caller.

Decode kernel `_lg_decode_kernel`: deployed weight RTN_E2M1(G[idx]) * s (ties away from zero, closed form
j' = j + (2r >= m)) written as bf16/fp16/fp32; 1 B read + 2 B write (bf16) per param.

Not bit-identical to the torch path (fp32 `/` and sqrt are Triton's, the sum order differs, Philox replaces
torch.rand); `emulate_update` is a torch transcription of the exact arithmetic for CPU tests.
"""
from __future__ import annotations

import torch

from .packed_na import e4m3_decode

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:                                   # CPU-only / no triton: only the torch emulation is available
    HAS_TRITON = False
    triton = None
    tl = None

NVFP4_BLOCK = 16
DELTA_CAP = 16.0


# ── torch transcription of the kernel arithmetic (CPU tests; same closed forms, same order of operations) ─────

def grid_value_cf(idx: torch.Tensor, m: int) -> torch.Tensor:
    """G[idx] in block-scale units by the closed form used in the kernel (float32)."""
    a = (idx.to(torch.int32) - 7 * m)
    neg = a < 0
    a = a.abs()
    j = torch.div(a, m, rounding_mode="floor")
    f = (a - j * m).to(torch.float32) / m
    jf = j.to(torch.float32)
    mag = torch.where(j < 4, (jf + f) * 0.5,
                      torch.where(j < 6, 2.0 + (jf - 4.0) + f, 4.0 + 2.0 * ((jf - 6.0) + f)))
    return torch.where(neg, -mag, mag)


def u_map_cf(w: torch.Tensor, m: int) -> torch.Tensor:
    """U: value -> continuous index, odd about 7m, linear extrapolation beyond +-6 (no clamp)."""
    x = w.abs()
    um = torch.where(x < 2.0, x * (2.0 * m), torch.where(x < 4.0, 4.0 * m + (x - 2.0) * m, 6.0 * m + (x - 4.0) * (m * 0.5)))
    return 7.0 * m + torch.where(w < 0, -um, um)


def deploy_cf(idx: torch.Tensor, m: int) -> torch.Tensor:
    """RTN_E2M1(G[idx]) (ties away from zero) in block-scale units, closed form."""
    a = idx.to(torch.int32) - 7 * m
    neg = a < 0
    a = a.abs()
    j = torch.div(a, m, rounding_mode="floor")
    r = a - j * m
    jp = j + (2 * r >= m).to(j.dtype)
    jf = jp.to(torch.float32)
    mag = torch.where(jp <= 4, 0.5 * jf, torch.where(jp == 5, 3.0, torch.where(jp == 6, 4.0, 6.0)))
    return torch.where(neg, -mag, mag)


def nibble_code_cf(idx: torch.Tensor, m: int) -> torch.Tensor:
    """E2M1 nibble (sign<<3 | magnitude index 0..7, no negative zero) of the DEPLOYED value of every idx, int32.
    Same closed form as deploy_cf: the magnitude index is the rounded half-gap index jp (E2M1_MAGS[jp])."""
    a = idx.to(torch.int32) - 7 * m
    neg = a < 0
    a = a.abs()
    j = torch.div(a, m, rounding_mode="floor")
    r = a - j * m
    jp = j + (2 * r >= m).to(j.dtype)
    return jp | ((neg & (jp > 0)).to(torch.int32) << 3)


def pack_nibbles_cf(idx: torch.Tensor, m: int) -> torch.Tensor:
    """[rows, cols] idx -> [rows, cols/2] uint8 NVFP4 nibbles (even column in the LOW nibble); lossless for the
    deployed weight (it is exactly E2M1 * block scale). Torch transcription of `_lg_pack_kernel`."""
    c = nibble_code_cf(idx, m)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).to(torch.uint8)


def block_scale_cf(sb: torch.Tensor, tscale: torch.Tensor) -> torch.Tensor:
    """[rows, nb] E4M3 bytes -> fp32 block scale (e4m3 * tscale), as in the kernel."""
    return e4m3_decode(sb) * tscale


def emulate_update(grad: torch.Tensor, idx: torch.Tensor, sb: torch.Tensor, tscale: torch.Tensor, v: torch.Tensor,
                   uniforms: torch.Tensor, *, lr: float, beta: float, eps: float, weight_decay: float, m: int):
    """Exact-arithmetic torch emulation of the update kernel with caller-supplied uniforms [rows, cols].
    -> (new idx int64, new v [rows,1], over bool mask)."""
    top = float(14 * m)
    g = grad.float()
    g_sq = g.square().sum(dim=1, keepdim=True) / g.shape[1]
    v_new = v * beta + g_sq * (1.0 - beta)
    denom = v_new.sqrt().clamp_min(eps)
    s = block_scale_cf(sb, tscale).repeat_interleave(NVFP4_BLOCK, dim=1)
    w0 = grid_value_cf(idx, m)
    g_hat = g / denom
    if weight_decay > 0.0:
        g_hat = g_hat + weight_decay * w0
    delta = (-lr * g_hat) / s
    delta = torch.nan_to_num(delta, nan=0.0, posinf=DELTA_CAP, neginf=-DELTA_CAP).clamp(-DELTA_CAP, DELTA_CAP)
    u = idx.to(torch.float32) + (u_map_cf(w0 + delta, m) - u_map_cf(w0, m))
    over = (u < 0.0) | (u > top)
    u = u.clamp(0.0, top)
    fl = torch.floor(u)
    new = (fl + (uniforms < (u - fl)).to(u.dtype)).clamp(0.0, top).to(torch.long)
    return new, v_new, over


# ── Triton kernels ───────────────────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _e4m3_scale(sbyte, ts):
        b = sbyte.to(tl.int32) & 0x7F
        e = b >> 3
        mm = (b & 7).to(tl.float32)
        pow2 = ((e + 120) << 23).to(tl.float32, bitcast=True)               # 2^(e-7), exact for e >= 1
        val = tl.where(e == 0, mm * 0.001953125, (8.0 + mm) * 0.125 * pow2)  # subnormal: m * 2^-9
        return val * ts

    @triton.jit
    def _grid_value(i, M: tl.constexpr):
        a = i - 7 * M
        neg = a < 0
        a = tl.where(neg, -a, a)
        j = a // M
        f = (a - j * M).to(tl.float32) / M
        jf = j.to(tl.float32)
        mag = tl.where(j < 4, (jf + f) * 0.5, tl.where(j < 6, 2.0 + (jf - 4.0) + f, 4.0 + 2.0 * ((jf - 6.0) + f)))
        return tl.where(neg, -mag, mag)

    @triton.jit
    def _u_map(w, M: tl.constexpr):
        x = tl.abs(w)
        um = tl.where(x < 2.0, x * (2.0 * M), tl.where(x < 4.0, 4.0 * M + (x - 2.0) * M, 6.0 * M + (x - 4.0) * (M * 0.5)))
        return 7.0 * M + tl.where(w < 0, -um, um)

    @triton.jit
    def _lg_update_kernel(g_ptr, idx_ptr, sb_ptr, ts_ptr, v_ptr, clip_ptr,
                          cols, nb, lr, beta, eps, wd, seed, lr_ptr, wd_ptr,
                          M: tl.constexpr, TOP: tl.constexpr, BLOCK_C: tl.constexpr,
                          HAS_WD: tl.constexpr, TRACK: tl.constexpr, TENSOR_HP: tl.constexpr):
        row = tl.program_id(0)
        # TENSOR_HP: lr / wd come from 0-dim fp32 device tensors (read at run time => a scheduled lr never changes the
        # launch signature, so torch.compile / CUDA graphs see no new constant); else they are the scalar arguments.
        if TENSOR_HP:
            lr_v = tl.load(lr_ptr)
            wd_v = tl.load(wd_ptr)
        else:
            lr_v = lr
            wd_v = wd
        base = row * cols                                   # int32; host asserts numel < 2^31
        offs0 = tl.arange(0, BLOCK_C)
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for c0 in range(0, cols, BLOCK_C):
            o = c0 + offs0
            g = tl.load(g_ptr + base + o, mask=o < cols, other=0.0).to(tl.float32)
            acc += g * g
        g_sq = tl.sum(acc, axis=0) / cols
        v_old = tl.load(v_ptr + row)
        v_new = v_old * beta + g_sq * (1.0 - beta)
        tl.store(v_ptr + row, v_new)
        denom = tl.maximum(tl.sqrt(v_new), eps)
        ts = tl.load(ts_ptr)
        nclip = tl.zeros([BLOCK_C], dtype=tl.int32)
        for c0 in range(0, cols, BLOCK_C):
            o = c0 + offs0
            mask = o < cols
            g = tl.load(g_ptr + base + o, mask=mask, other=0.0).to(tl.float32)
            i = tl.load(idx_ptr + base + o, mask=mask, other=0).to(tl.int32)
            sbyte = tl.load(sb_ptr + row * nb + (o >> 4), mask=mask, other=1)
            s = _e4m3_scale(sbyte, ts)
            w0 = _grid_value(i, M)
            g_hat = g / denom
            if HAS_WD:
                g_hat = g_hat + wd_v * w0
            delta = (-lr_v * g_hat) / s
            delta = tl.where(delta != delta, 0.0, delta)
            delta = tl.minimum(tl.maximum(delta, -16.0), 16.0)
            u = i.to(tl.float32) + (_u_map(w0 + delta, M) - _u_map(w0, M))
            if TRACK:
                nclip += (((u < 0.0) | (u > TOP)) & mask).to(tl.int32)
            u = tl.minimum(tl.maximum(u, 0.0), TOP)
            fl = tl.floor(u)
            r = tl.rand(seed, base + o)
            new = fl + tl.where(r < (u - fl), 1.0, 0.0)
            new = tl.minimum(tl.maximum(new, 0.0), TOP).to(tl.int32)
            tl.store(idx_ptr + base + o, new.to(idx_ptr.dtype.element_ty), mask=mask)
        if TRACK:
            tl.store(clip_ptr + row, tl.sum(nclip, axis=0))

    @triton.jit
    def _lg_decode_kernel(idx_ptr, sb_ptr, ts_ptr, out_ptr, cols, nb,
                          M: tl.constexpr, BLOCK_C: tl.constexpr):
        row = tl.program_id(0)
        o = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = o < cols
        base = row * cols
        i = tl.load(idx_ptr + base + o, mask=mask, other=0).to(tl.int32)
        sbyte = tl.load(sb_ptr + row * nb + (o >> 4), mask=mask, other=1)
        s = _e4m3_scale(sbyte, tl.load(ts_ptr))
        a = i - 7 * M
        neg = a < 0
        a = tl.where(neg, -a, a)
        j = a // M
        r = a - j * M
        jp = j + tl.where(2 * r >= M, 1, 0)
        jf = jp.to(tl.float32)
        mag = tl.where(jp <= 4, 0.5 * jf, tl.where(jp == 5, 3.0, tl.where(jp == 6, 4.0, 6.0)))
        w = tl.where(neg, -mag, mag) * s
        tl.store(out_ptr + base + o, w.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _nib(i, M: tl.constexpr):
        a = i - 7 * M
        neg = a < 0
        a = tl.where(neg, -a, a)
        j = a // M
        r = a - j * M
        jp = j + tl.where(2 * r >= M, 1, 0)
        return jp | tl.where(neg & (jp > 0), 8, 0)

    @triton.jit
    def _lg_pack_kernel(idx_ptr, out_ptr, half, M: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = o < half
        base = row * half * 2
        lo = tl.load(idx_ptr + base + 2 * o, mask=mask, other=0).to(tl.int32)
        hi = tl.load(idx_ptr + base + 2 * o + 1, mask=mask, other=0).to(tl.int32)
        c = _nib(lo, M) | (_nib(hi, M) << 4)
        tl.store(out_ptr + row * half + o, c.to(tl.uint8), mask=mask)


# ── host wrappers ────────────────────────────────────────────────────────────────────────────────────────────

def _pow2_ge(n: int, cap: int) -> int:
    p = 16
    while p < n and p < cap:
        p *= 2
    return p


def lg_update_(grad: torch.Tensor, idx: torch.Tensor, scale_bytes: torch.Tensor, tscale: torch.Tensor,
               v_rms: torch.Tensor, *, lr, beta: float, eps: float, weight_decay, m: int, seed: int,
               track_clip: bool = True, block_c: int | None = None, num_warps: int | None = None,
               has_wd: bool | None = None):
    """In-place fused update of idx [rows, cols] and v_rms [rows, 1]. grad: bf16/fp16/fp32 contiguous [rows, cols].
    One kernel launch. -> int32 [rows] per-row count of requests beyond the grid (None if track_clip=False).
    lr / weight_decay: python floats (baked into the launch) OR 0-dim fp32 CUDA tensors (read by pointer inside the
    kernel; then pass has_wd = whether the weight-decay term is compiled in)."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    rows, cols = idx.shape
    if grad.shape != idx.shape:
        raise ValueError("grad/idx shape mismatch")
    if cols % NVFP4_BLOCK:
        raise ValueError("cols must be a multiple of 16")
    if rows * cols >= 2 ** 31:
        raise ValueError("numel must be < 2^31 (int32 offsets / Philox counter)")
    if not (grad.is_cuda and idx.is_cuda):
        raise ValueError("lg_update_ needs CUDA tensors")
    if grad.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        grad = grad.float()
    grad = grad.contiguous()
    if not (idx.is_contiguous() and v_rms.is_contiguous() and scale_bytes.is_contiguous()):
        raise ValueError("idx, v_rms and scale_bytes must be contiguous")
    bc = block_c or _pow2_ge(cols, 2048)
    nw = num_warps or (8 if bc >= 2048 else 4 if bc >= 512 else 2)
    clip = torch.empty(rows, dtype=torch.int32, device=idx.device) if track_clip else idx   # dummy ptr when off
    tensor_hp = isinstance(lr, torch.Tensor)
    if tensor_hp:
        if not (isinstance(weight_decay, torch.Tensor) and lr.is_cuda and weight_decay.is_cuda
                and lr.numel() == 1 and weight_decay.numel() == 1 and lr.dtype == weight_decay.dtype == torch.float32):
            raise ValueError("tensor lr / weight_decay must both be 0-dim fp32 CUDA tensors")
        lr_f = wd_f = 0.0
        lr_p, wd_p = lr, weight_decay
        use_wd = bool(has_wd)
    else:
        lr_f, wd_f = float(lr), float(weight_decay)
        lr_p = wd_p = tscale                                  # dummy pointers, never read (TENSOR_HP=False)
        use_wd = wd_f > 0.0 if has_wd is None else bool(has_wd)
    _lg_update_kernel[(rows,)](
        grad, idx, scale_bytes, tscale, v_rms, clip, cols, cols // NVFP4_BLOCK,
        lr_f, float(beta), float(eps), wd_f, int(seed) & 0x7FFFFFFF, lr_p, wd_p,
        M=int(m), TOP=float(14 * m), BLOCK_C=bc, HAS_WD=use_wd, TRACK=bool(track_clip), TENSOR_HP=tensor_hp,
        num_warps=nw)
    return clip if track_clip else None


def lg_decode(idx: torch.Tensor, scale_bytes: torch.Tensor, tscale: torch.Tensor, m: int,
              dtype: torch.dtype = torch.bfloat16, block_c: int = 1024) -> torch.Tensor:
    """Deployed weight RTN_E2M1(G[idx]) * s as `dtype` [rows, cols]; one launch, no fp32 intermediates in HBM."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    rows, cols = idx.shape
    out = torch.empty(rows, cols, dtype=dtype, device=idx.device)
    _lg_decode_kernel[(rows, triton.cdiv(cols, block_c))](
        idx, scale_bytes, tscale, out, cols, cols // NVFP4_BLOCK, M=int(m), BLOCK_C=block_c, num_warps=4)
    return out


def lg_pack_nibbles(idx: torch.Tensor, m: int, block: int = 512) -> torch.Tensor:
    """[rows, cols] idx -> [rows, cols/2] uint8 NVFP4 nibbles of the deployed weight; one launch, 1 B read + 0.5 B write
    per param (feeds torch._scaled_mm / the Triton fp4 GEMMs without any dense intermediate)."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    rows, cols = idx.shape
    if cols % 2:
        raise ValueError("cols must be even")
    if not idx.is_contiguous():
        idx = idx.contiguous()
    out = torch.empty(rows, cols // 2, dtype=torch.uint8, device=idx.device)
    _lg_pack_kernel[(rows, triton.cdiv(cols // 2, block))](idx, out, cols // 2, M=int(m), BLOCK=block, num_warps=4)
    return out


def bench_update(rows: int = 512, cols: int = 2048, iters: int = 100, k: int = 8, grad_dtype=torch.bfloat16,
                 device: str = "cuda", compare_torch: bool = True, sweep: bool = False) -> dict:
    """Micro-benchmark of one layer's update (GPU only). Returns and prints a dict of timings (us) and GB/s.
    Keys: triton_us, torch_us (eager reference, if compare_torch), decode_us, speedup, triton_gbps, bytes_per_param,
    launches (1 + clip reduce), and with sweep=True the best (block_c, num_warps)."""
    from .latent_grid import LatentGridLinear
    import time
    if not (HAS_TRITON and torch.cuda.is_available()):
        raise RuntimeError("bench_update needs CUDA + Triton")

    def timeit(fn, n):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1000.0 / n

    torch.manual_seed(0)
    out = {"rows": rows, "cols": cols, "grad_dtype": str(grad_dtype)}
    g = (torch.randn(rows, cols, device=device) * 1e-2).to(grad_dtype)
    lay = LatentGridLinear(cols, rows, k=k, lr=1e-3, backend="triton").to(device)
    lay.track_clip = False
    out["triton_us"] = timeit(lambda: lay.apply_update_(g), iters)
    lay.track_clip = True
    out["triton_clip_us"] = timeit(lambda: lay.apply_update_(g), iters)
    out["decode_us"] = timeit(lambda: lay.weight(torch.bfloat16), iters)
    bytes_pp = g.element_size() + 2 + 1.0 / 16
    out["bytes_per_param"] = bytes_pp
    out["triton_gbps"] = rows * cols * bytes_pp / (out["triton_us"] * 1e-6) / 1e9
    if sweep:
        best = None
        for bc in (256, 512, 1024, 2048, 4096):
            for nw in (2, 4, 8):
                if bc > cols * 2:
                    continue
                us = timeit(lambda: lg_update_(g, lay.idx, lay._scale, lay._tscale, lay.v_rms, lr=1e-3, beta=0.9,
                                               eps=1e-3, weight_decay=0.0, m=lay.fmt.m, seed=1, track_clip=False,
                                               block_c=bc, num_warps=nw), max(20, iters // 4))
                if best is None or us < best[0]:
                    best = (us, bc, nw)
        out["best_us"], out["best_block_c"], out["best_num_warps"] = best
    if compare_torch:
        ref = LatentGridLinear(cols, rows, k=k, lr=1e-3, backend="torch").to(device)
        out["torch_us"] = timeit(lambda: ref.apply_update_(g), max(10, iters // 5))
        out["speedup"] = out["torch_us"] / out["triton_us"]
    print(out)
    return out
