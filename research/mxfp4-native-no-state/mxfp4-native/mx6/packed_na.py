"""Packed (value, accumulator) formats — per-param momentum in one word.

Two production configurations:

  * **5-bit** (``PackedNA5``): 5 signed magnitudes × 5 accumulator levels = 25 codes
    Magnitudes ``{0, 0.5, 2.0}`` → signed ``{-2, -0.5, 0, 0.5, 2}``, C=3.
    Total with E8M0 scale: 5.25 bits/param.

  * **6-bit** (``PackedNA6``): 7 signed magnitudes × 9 accumulator levels = 63 codes
    Magnitudes ``{0, 0.5, 2.0, 4.0}`` → signed ``{-4, -2, -0.5, 0, 0.5, 2, 4}``, C=5.
    Total with E8M0 scale: 6.25 bits/param.

The accumulator is packed into the same word as the value (like the ternary
counter on ``main``), so there is no separate accumulator buffer.  The forward
pass reads only the value; the accumulator is invisible to inference.

See ``docs/PACKED_NA.md`` for the design and A100 benchmark numbers.
"""
from __future__ import annotations

import math
import torch

BLOCK = 32


def ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """Smallest integer e with 2**e >= x, computed exactly."""
    m, e = torch.frexp(x.clamp_min(1e-30))
    return e - (m == 0.5).to(e.dtype)


def _build_signed(mags_unsigned: list[float]) -> torch.Tensor:
    """Unsigned magnitudes [0, a, b, ...] → symmetric signed tensor."""
    pos = mags_unsigned[1:]  # exclude 0
    return torch.tensor([-v for v in reversed(pos)] + [0.0] + pos)


def _build_midpoints(signed: torch.Tensor) -> torch.Tensor:
    """Midpoints between consecutive signed values (for rounding)."""
    return (signed[1:] + signed[:-1]) / 2


class PackedNA:
    """A packed (value, accumulator) format with configurable magnitudes and C.

    The code word encodes ``mag_index * acc_levels + (acc + C - 1)``, where
    ``mag_index`` selects a signed magnitude and ``acc`` is the per-parameter
    accumulator in ``[-(C-1), C-1]``.  The total code count is
    ``len(signed) * (2*C - 1)``; the bit width is ``ceil(log2(total))``.
    """

    def __init__(self, mags_unsigned: list[float], C: int):
        self.C = C
        self.acc_levels = 2 * C - 1
        self.signed = _build_signed(mags_unsigned)
        self.n_vals = len(self.signed)
        self.mids = _build_midpoints(self.signed)
        self.n_codes = self.n_vals * self.acc_levels
        self.bits = math.ceil(math.log2(self.n_codes)) if self.n_codes > 1 else 1
        # precompute per-magnitude gap (smallest distance to a neighbour)
        gaps = []
        s = self.signed.tolist()
        for i in range(self.n_vals):
            adj = []
            if i > 0:
                adj.append(abs(s[i] - s[i - 1]))
            if i < self.n_vals - 1:
                adj.append(abs(s[i + 1] - s[i]))
            gaps.append(min(adj) if adj else 1.0)
        self._gaps = torch.tensor(gaps)
        self._mid_idx = self.n_vals // 2  # index of 0.0

    # ── encode / decode ──────────────────────────────────────────────────────

    def encode(self, mag_idx: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        return (mag_idx.to(torch.long) * self.acc_levels + (acc.to(torch.long) + self.C - 1))

    def decode(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = code.to(torch.long)
        return code // self.acc_levels, (code % self.acc_levels) - (self.C - 1)

    def value_float(self, code: torch.Tensor) -> torch.Tensor:
        mag_idx = (code.to(torch.long) // self.acc_levels).clamp(0, self.n_vals - 1)
        return self.signed.to(code.device)[mag_idx]

    def signed_index(self, code: torch.Tensor) -> torch.Tensor:
        mag_idx = (code.to(torch.long) // self.acc_levels).clamp(0, self.n_vals - 1)
        return mag_idx - self._mid_idx

    def code_from_index(self, idx: torch.Tensor) -> torch.Tensor:
        mag_idx = (idx + self._mid_idx).clamp(0, self.n_vals - 1)
        return (mag_idx * self.acc_levels + (self.C - 1)).to(torch.uint8)

    def gap_at(self, code: torch.Tensor) -> torch.Tensor:
        mag_idx = (code.to(torch.long) // self.acc_levels).clamp(0, self.n_vals - 1)
        return self._gaps.to(code.device)[mag_idx]

    # ── scale (E8M0 per 32-block) ─────────────────────────────────────────────

    @staticmethod
    def scale_bytes(w: torch.Tensor, headroom: float = 2.0) -> torch.Tensor:
        out_f, in_f = w.shape
        blocks = w.reshape(out_f, in_f // BLOCK, BLOCK)
        amax = blocks.abs().amax(dim=-1).clamp_min(1e-30)
        exp = ceil_log2(amax / headroom)
        return (exp + 127).clamp(0, 254).to(torch.uint8)

    @staticmethod
    def scale_floats(sb: torch.Tensor) -> torch.Tensor:
        return torch.pow(2.0, sb.to(torch.float32) - 127.0)

    # ── quantize / dequantize ─────────────────────────────────────────────────

    def quantize(self, w: torch.Tensor, headroom: float = 1.5
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        sb = self.scale_bytes(w, headroom)
        s = self.scale_floats(sb).repeat_interleave(BLOCK, dim=1)
        a = w / s
        idx = torch.bucketize(a, self.mids.to(a.device)).clamp(0, self.n_vals - 1)
        codes = (idx * self.acc_levels + (self.C - 1)).to(torch.uint8)
        return codes, sb

    def dequantize(self, codes: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
        s = self.scale_floats(sb).repeat_interleave(BLOCK, dim=1)
        mag_idx = (codes.to(torch.long) // self.acc_levels).clamp(0, self.n_vals - 1)
        return self.signed.to(codes.device)[mag_idx] * s

    def dequantize_fast(self, codes: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
        """Fused decode: single index_select + multiply (fewer kernel launches).
        Uses torch.compile when enabled (call enable_compile() first)."""
        if _compile_enabled:
            return _decode_compiled(codes, sb, self.signed.to(codes.device),
                                    self.acc_levels, self.n_vals)
        s = self.scale_floats(sb).repeat_interleave(BLOCK, dim=1)
        mag_idx = (codes.to(torch.long) // self.acc_levels).clamp(0, self.n_vals - 1)
        vals = torch.index_select(self.signed.to(codes.device), 0, mag_idx.reshape(-1))
        return vals.reshape_as(s) * s

    # ── update ────────────────────────────────────────────────────────────────

    @staticmethod
    def _sr_int(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        fl = torch.floor(x)
        return fl + (torch.rand(x.shape, generator=gen, device=x.device) < (x - fl)).to(x.dtype)

    def step(self, codes: torch.Tensor, acc: torch.Tensor, scale: torch.Tensor,
             grad_hat: torch.Tensor, lr: float,
             gen: torch.Generator | None = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """One per-param integrator update. Returns (new_codes, new_acc).
        Uses torch.compile for pre/post-SR when enabled (call enable_compile())."""
        if _compile_enabled:
            # pre-SR (compiled)
            acc_f = _step_pre_sr_compiled(
                codes, acc, scale, grad_hat, lr, self.C,
                self.signed, self.acc_levels, self.n_vals,
                self._gaps, self._mid_idx)
            # SR (eager — preserves RNG stream)
            fl = torch.floor(acc_f)
            cc = fl + (torch.rand(acc_f.shape, generator=gen, device=acc_f.device) < (acc_f - fl)).to(acc_f.dtype)
            # post-SR (compiled)
            return _step_post_sr_compiled(cc, codes, self.C, self.acc_levels, self.n_vals, self._mid_idx)

        # eager path (original)
        s = self.scale_floats(scale).repeat_interleave(BLOCK, dim=1)
        gap = self.gap_at(codes) * s
        ticks = (-lr * grad_hat) / gap * self.C
        acc_f = acc.to(torch.float32) + ticks
        cc = self._sr_int(acc_f, gen)
        carry = torch.trunc(cc / self.C)
        rem = cc - carry * self.C
        idx = self.signed_index(codes)
        new_idx = (idx + carry.to(torch.int16)).clamp(-self._mid_idx, self.n_vals - 1 - self._mid_idx)
        blocked = (idx + carry.to(torch.int16)) != new_idx
        rem = torch.where(blocked, torch.sign(cc) * (self.C - 1), rem).clamp(-(self.C - 1), self.C - 1)
        return self.code_from_index(new_idx), rem.to(torch.int8)

    # ── accounting ───────────────────────────────────────────────────────────

    def bits_per_param(self) -> float:
        """Code bits + E8M0 scale (0.25) + per-row RMS (negligible at scale)."""
        return float(self.bits) + 0.25

    def persistent_bytes(self, out_f: int, in_f: int) -> int:
        blocks = in_f // BLOCK
        code_bytes = out_f * in_f * self.bits // 8
        if out_f * in_f * self.bits % 8:
            code_bytes += 1  # padding for non-byte-aligned widths
        scale_bytes_ = out_f * blocks  # 1 byte/block
        rms_bytes = out_f * 4  # per-row float32
        return code_bytes + scale_bytes_ + rms_bytes


# ── production configurations ─────────────────────────────────────────────────

#: 5-bit format: 5 values, 5 acc levels (C=3), 25 codes -> 5 bits
#: Magnitudes {0, 0.5, 3.0} -> {-3, -0.5, 0, 0.5, 3} ("wide" — best in sweep)
PackedNA5 = PackedNA(mags_unsigned=[0.0, 0.5, 3.0], C=3)

#: 6-bit format: 7 values, 9 acc levels (C=5), 63 codes -> 6 bits
#: Magnitudes {0, 0.5, 1.0, 6.0} -> {-6, -1, -0.5, 0, 0.5, 1, 6} ("e2m1" — best in sweep)
PackedNA6 = PackedNA(mags_unsigned=[0.0, 0.5, 1.0, 6.0], C=5)


# ── torch.compile acceleration (opt-in) ────────────────────────────────────────
# Compiled versions of the decode and step functions.  These fuse the small
# elementwise ops into fewer kernels, giving ~16% speedup on GPU.
# Usage: pass compiled=True to MXNLinear, or call PackedNA with use_compile=True.
# dynamic=False recompiles once per shape (fine for training with fixed shapes).

_compile_enabled = False  # set True to enable torch.compile globally

def enable_compile():
    """Enable torch.compile for all PackedNA decode/step calls."""
    global _compile_enabled
    _compile_enabled = True

def _maybe_compile(fn):
    """Decorate a function with torch.compile if compilation is enabled."""
    if _compile_enabled:
        return torch.compile(fn, dynamic=False)
    return fn


@_maybe_compile
def _decode_compiled(codes, sb, signed_vals, acc_levels, n_vals, block=BLOCK):
    """Fused decode: codes + scale -> float weight (compiled when enabled)."""
    s = torch.pow(2.0, sb.to(torch.float32) - 127.0).repeat_interleave(block, dim=1)
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    vals = torch.index_select(signed_vals, 0, mag_idx.reshape(-1))
    return vals.reshape_as(s) * s


@_maybe_compile
def _step_pre_sr_compiled(codes, acc, scale_bytes, grad_hat, lr, C,
                          signed_vals, acc_levels, n_vals, gaps, mid_idx, block=BLOCK):
    """Pre-SR part of step: compute accumulated ticks (compiled when enabled)."""
    s = torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0).repeat_interleave(block, dim=1)
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    gap = gaps.to(codes.device)[mag_idx]
    ticks = (-lr * grad_hat) / (gap * s) * C
    return acc.to(torch.float32) + ticks


@_maybe_compile
def _step_post_sr_compiled(cc, codes, C, acc_levels, n_vals, mid_idx):
    """Post-SR part of step: carry + encode (compiled when enabled)."""
    carry = torch.trunc(cc / C)
    rem = cc - carry * C
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    idx = mag_idx - mid_idx
    new_idx = (idx + carry.to(torch.int16)).clamp(-(n_vals // 2), n_vals - 1 - n_vals // 2)
    blocked = (idx + carry.to(torch.int16)) != new_idx
    rem = torch.where(blocked, torch.sign(cc) * (C - 1), rem).clamp(-(C - 1), C - 1)
    new_mag_idx = (new_idx + n_vals // 2).clamp(0, n_vals - 1)
    new_codes = (new_mag_idx * acc_levels + (C - 1)).to(torch.uint8)
    return new_codes, rem.to(torch.int8)
