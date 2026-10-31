"""Packed (value, accumulator) formats — per-param momentum in one word.

Two production configurations:

  * **5-bit** (``PackedNA5``): 5 signed magnitudes × 5 accumulator levels = 25 codes
    Magnitudes ``{0, 0.5, 3.0}`` → signed ``{-3, -0.5, 0, 0.5, 3}`` ("wide"), C=3.
    Total with E8M0 scale: 5.25 bits/param.

  * **6-bit** (``PackedNA6``): 7 signed magnitudes × 9 accumulator levels = 63 codes
    Magnitudes ``{0, 0.5, 1.0, 6.0}`` → signed ``{-6, -1, -0.5, 0, 0.5, 1, 6}``
    ("e2m1"), C=5.  Total with E8M0 scale: 6.25 bits/param.

The accumulator occupies the low ``2C-1`` residue of the code, so the state is
one packed word per coefficient (no separate accumulator plane).  ``pack_state``
bit-packs those words (4 codes / 3 bytes at 6 bits, 8 codes / 5 bytes at 5 bits),
which is what ``persistent_bytes`` reports.

The forward pass reads only the value; the accumulator is invisible to inference.

See ``docs/PACKED_NA.md`` for the design and the GPU benchmark numbers
(measured on a G4 / RTX PRO 6000, not an A100).
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
        # Precompute code -> value lookup table (avoids integer division in kernels)
        code_to_val = torch.zeros(self.n_codes)
        for c in range(self.n_codes):
            mi = c // self.acc_levels
            code_to_val[c] = self.signed[mi].item()
        self._code_to_value = code_to_val

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

    def scale_bytes_mse(self, w: torch.Tensor, headroom: float = 1.5,
                        clip_frac: float = 0.0, octaves: int = 1) -> torch.Tensor:
        """E8M0 exponent that minimises block reconstruction MSE.

        The headroom rule only centres the search; the winner is chosen by
        ``sum((w - Q_s(w))**2)`` per block over ``2*octaves+1`` adjacent octaves.
        A single outlier in a 32-wide block otherwise inflates the max-rule scale
        and pushes the whole block's mass to zero, which is exactly the failure we
        measured. ``clip_frac`` clamps each row to its (1 - clip_frac) quantile
        before the search so one weight cannot set the exponent.
        """
        out_f, in_f = w.shape
        nb = in_f // BLOCK
        blocks = w.reshape(out_f, nb, BLOCK)
        if clip_frac > 0:
            qv = torch.quantile(w.abs().float().reshape(out_f, -1), 1.0 - clip_frac,
                                dim=1, keepdim=True)
            blocks = blocks.clamp(-qv[..., None], qv[..., None])

        amax = blocks.abs().amax(dim=-1).clamp_min(1e-30)
        e0 = (ceil_log2(amax / headroom) + 127).clamp(0, 254)   # E8M0 byte, not the raw exponent
        cands = [e0 + d for d in range(-octaves, octaves + 1)]

        mids = self.mids.to(w.device)
        signed = self.signed.to(w.device)
        best_mse = None
        best_e = e0
        for e in cands:
            e = e.clamp(0, 254)
            s = torch.pow(2.0, e.to(torch.float32) - 127.0)[..., None]
            a = blocks / s
            idx = torch.bucketize(a, mids).clamp(0, self.n_vals - 1)
            q = signed[idx] * s
            mse = ((blocks - q) ** 2).sum(dim=-1)
            if best_mse is None:
                best_mse, best_e = mse, e
            else:
                better = mse < best_mse
                best_mse = torch.where(better, mse, best_mse)
                best_e = torch.where(better, e, best_e)
        return best_e.to(torch.uint8)

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
        return _get_compiled(_decode_core)(
            codes, sb, self.signed.to(codes.device), self.acc_levels, self.n_vals)

    # ── update ────────────────────────────────────────────────────────────────

    @staticmethod
    def _sr_int(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        fl = torch.floor(x)
        return fl + (torch.rand(x.shape, generator=gen, device=x.device) < (x - fl)).to(x.dtype)

    def step(self, codes: torch.Tensor, scale: torch.Tensor,
             grad_hat: torch.Tensor, lr: float,
             gen: torch.Generator | None = None) -> torch.Tensor:
        """One per-param integrator update. Returns the new code word.

        The accumulator lives *inside* the code (the low ``2C-1`` residue), so
        there is no separate accumulator tensor: the update decodes it, applies
        the tick, and re-encodes carry + remainder into one word.
        """
        acc_f = _get_compiled(_step_pre_sr)(
            codes, scale, grad_hat, lr, self.C,
            self._gaps, self.acc_levels, self.n_vals, self._mid_idx)
        # SR stays eager: it must consume the global/per-call RNG stream in order
        cc = self._sr_int(acc_f, gen)
        return _get_compiled(_step_post_sr)(
            cc, codes, self.C, self.acc_levels, self.n_vals, self._mid_idx)

    # ── bit packing (the persistent footprint) ────────────────────────────────

    def group(self) -> tuple[int, int]:
        """(codes per group, bytes per group) with no wasted bits."""
        for g in range(1, 33):
            if (g * self.bits) % 8 == 0:
                return g, g * self.bits // 8
        raise ValueError(f"no byte-aligned group for {self.bits} bits")

    def packed_width(self, in_features: int) -> int:
        """Bytes per row of packed state."""
        g, b = self.group()
        if in_features % g:
            raise ValueError(f"in_features {in_features} must be a multiple of {g}")
        return (in_features // g) * b

    def _row_chunks(self, out_f: int, in_f: int) -> list[tuple[int, int]]:
        """Row slices keeping the bit-matrix intermediate under ~64 MiB."""
        per_row = max(1, in_f * self.bits * 4)
        rows = max(1, (64 << 20) // per_row)
        return [(i, min(i + rows, out_f)) for i in range(0, out_f, rows)]

    def pack_state(self, codes: torch.Tensor) -> torch.Tensor:
        """[out, in] codes -> [out, packed_width] uint8 (MSB-first within a group)."""
        out_f, in_f = codes.shape
        g, b = self.group()
        bits = self.bits
        dev = codes.device
        if bits == 6 and in_f % 4 == 0:
            # 4 codes / 3 bytes in three ops (same layout as the generic path)
            c = codes.reshape(-1, 4).to(torch.int32)
            b0 = c[:, 0] | ((c[:, 1] & 0x03) << 6)
            b1 = (c[:, 1] >> 2) | ((c[:, 2] & 0x0F) << 4)
            b2 = (c[:, 2] >> 4) | (c[:, 3] << 2)
            packed = torch.stack((b0, b1, b2), dim=-1).reshape(out_f, -1)
            return (packed & 0xFF).to(torch.uint8)
        shifts = torch.arange(bits - 1, -1, -1, device=dev)
        byte_w = 1 << torch.arange(7, -1, -1, device=dev)
        out = torch.empty(out_f, (in_f // g) * b, dtype=torch.uint8, device=dev)
        for r0, r1 in self._row_chunks(out_f, in_f):
            c = codes[r0:r1].reshape(-1, in_f // g, g).to(torch.int32)
            bitmat = ((c.unsqueeze(-1) >> shifts) & 1).reshape(-1, in_f // g, g * bits)
            packed = (bitmat.reshape(-1, in_f // g, b, 8) * byte_w).sum(-1)
            out[r0:r1] = packed.to(torch.uint8).reshape(r1 - r0, -1)
        return out

    def unpack_state(self, packed: torch.Tensor, in_features: int) -> torch.Tensor:
        """[out, packed_width] uint8 -> [out, in] codes."""
        out_f = packed.shape[0]
        g, b = self.group()
        bits = self.bits
        dev = packed.device
        if bits == 6 and in_features % 4 == 0:
            p = packed.reshape(-1, 3).to(torch.int32)
            c0 = p[:, 0] & 0x3F
            c1 = ((p[:, 0] >> 6) | (p[:, 1] << 2)) & 0x3F
            c2 = ((p[:, 1] >> 4) | (p[:, 2] << 4)) & 0x3F
            c3 = (p[:, 2] >> 2) & 0x3F
            codes = torch.stack((c0, c1, c2, c3), dim=-1).reshape(out_f, in_features)
            return codes.to(torch.uint8)
        bit_pos = torch.arange(7, -1, -1, device=dev)   # shift amounts, not weights
        shifts = torch.arange(bits - 1, -1, -1, device=dev)
        out = torch.empty(out_f, in_features, dtype=torch.uint8, device=dev)
        for r0, r1 in self._row_chunks(out_f, in_features):
            p = packed[r0:r1].reshape(-1, packed.shape[1] // b, b).to(torch.int32)
            bitmat = ((p.unsqueeze(-1) >> bit_pos) & 1).reshape(-1, packed.shape[1] // b, b * 8)
            bitmat = bitmat[..., :g * bits].reshape(-1, in_features // g, g, bits)
            c = (bitmat << shifts).sum(-1)
            out[r0:r1] = c.reshape(r1 - r0, in_features).to(torch.uint8)
        return out


    # ── accounting ───────────────────────────────────────────────────────────

    def bits_per_param(self) -> float:
        """Code bits + E8M0 scale (0.25) + per-row RMS (negligible at scale)."""
        return float(self.bits) + 0.25

    @property
    def code_to_value(self) -> torch.Tensor:
        """Precomputed lookup table: code -> signed magnitude (for Triton kernel)."""
        return self._code_to_value

    def persistent_bytes(self, out_f: int, in_f: int) -> int:
        """Bytes actually stored: bit-packed codes + E8M0 scales + per-row RMS."""
        return out_f * self.packed_width(in_f) + out_f * (in_f // BLOCK) + out_f * 4


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
_compiled_cache: dict[str, object] = {}

def enable_compile():
    """Enable torch.compile for PackedNA decode/step.

    Compilation is lazy (on the next call), so this works whether it is called
    before or after import.  The previous decorator-at-import form silently
    did nothing when called later.
    """
    global _compile_enabled
    _compile_enabled = True
    _compiled_cache.clear()


def disable_compile():
    """Return to the eager path and drop compiled artifacts."""
    global _compile_enabled
    _compile_enabled = False
    _compiled_cache.clear()


def _get_compiled(fn):
    """Compiled variant of ``fn`` when enabled, else ``fn`` itself."""
    if not _compile_enabled:
        return fn
    hit = _compiled_cache.get(fn.__name__)
    if hit is None:
        hit = torch.compile(fn, dynamic=False)
        _compiled_cache[fn.__name__] = hit
    return hit


def _decode_core(codes, sb, signed_vals, acc_levels, n_vals, block=BLOCK):
    """Fused decode: codes + scale -> float weight."""
    s = torch.pow(2.0, sb.to(torch.float32) - 127.0).repeat_interleave(block, dim=1)
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    vals = torch.index_select(signed_vals, 0, mag_idx.reshape(-1))
    return vals.reshape_as(s) * s


def _step_pre_sr(codes, scale_bytes, grad_hat, lr, C, gaps, acc_levels, n_vals,
                 mid_idx, block=BLOCK):
    """Accumulator read from the code + this step's ticks (pre-SR)."""
    s = torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0).repeat_interleave(block, dim=1)
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    gap = gaps.to(codes.device)[mag_idx] * s
    acc = ((codes.to(torch.long) % acc_levels) - (C - 1)).to(torch.float32)
    return acc + (-lr * grad_hat) / gap * C


def _step_post_sr(cc, codes, C, acc_levels, n_vals, mid_idx):
    """Carry + remainder re-encoded into one word (post-SR)."""
    carry = torch.trunc(cc / C)
    rem = cc - carry * C
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    idx = mag_idx - mid_idx
    moved = idx + carry.to(torch.int16)
    new_idx = moved.clamp(-mid_idx, n_vals - 1 - mid_idx)
    rem = torch.where(moved != new_idx, torch.sign(cc) * (C - 1), rem)
    rem = rem.clamp(-(C - 1), C - 1)
    new_mag = new_idx + mid_idx
    return (new_mag * acc_levels + (rem + (C - 1))).to(torch.uint8)
