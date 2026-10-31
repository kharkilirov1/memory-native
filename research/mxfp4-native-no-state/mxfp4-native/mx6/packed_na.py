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

# ── NVFP4 scaling (opt-in, scale_format="nvfp4") ──────────────────────────────
# E2M1 elements, one FP8 E4M3 scale per 16 elements, one FP32 scale per tensor:
#     w ~= ladder[idx] * e4m3(byte) * tensor_scale
#     tensor_scale = amax(w) / (ladder_max * 448)
#     byte         = RTN_e4m3( block_amax / (ladder_max * tensor_scale) )
# ladder_max is the OUTER rung of the chosen ladder (3 for "wide", 6 for "e2m1"),
# so the block maximum lands on the outer rung and the largest block scale is 448.
SCALE_FORMATS = ("e8m0", "nvfp4")
NVFP4_BLOCK = 16
E4M3_MAX = 448.0
E4M3_MIN = 2.0 ** -9          # smallest positive E4M3 (subnormal 0.001 * 2^-6)


def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Positive floats -> E4M3 bytes (uint8), round-to-nearest-even.

    Saturates at 448 (E4M3FN has no inf; torch would give NaN on overflow) and floors at
    the smallest positive value 2^-9, so a scale byte is never 0 (-> w/0) and never NaN.
    """
    return x.to(torch.float32).clamp(E4M3_MIN, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_decode(b: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes -> fp32 by bit arithmetic (exact; sign bit ignored, scales are >0).

    bias 7: normal = (8+m)/8 * 2^(e-7); subnormal (e=0) = m * 2^-9. The same arithmetic
    is used in the Triton kernel, so host and device decode identically.
    """
    b = b.to(torch.int32) & 0x7F
    e = (b >> 3).to(torch.float32)
    m = (b & 7).to(torch.float32)
    normal = (8.0 + m) * 0.125 * torch.pow(2.0, e - 7.0)
    return torch.where(e == 0, m * E4M3_MIN, normal)


_E4M3_LN_GAPS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def e4m3_ln_gaps(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(up, dn) float32 [256]: ln(v[b+1]/v[b]) and ln(v[b]/v[b-1]) for the valid positive E4M3
    bytes b in [1, 126] (v = e4m3_decode). The missing neighbour at either end (b = 1: no code
    below, b = 126 = 448: no code above) reuses the inner gap; the step is clamped there anyway.
    Entries outside [1, 126] are unused."""
    key = str(device)
    if key not in _E4M3_LN_GAPS:
        lv = torch.log(e4m3_decode(torch.arange(128, dtype=torch.uint8)).clamp_min(E4M3_MIN))
        d = lv[1:] - lv[:-1]                      # d[b] = ln(v[b+1] / v[b])
        up = torch.zeros(256)
        dn = torch.zeros(256)
        up[:127] = d
        dn[1:128] = d
        up[126] = dn[126]                         # 126 -> 127 is NaN: reuse the gap below
        dn[1] = up[1]                             # 1 -> 0 is zero: reuse the gap above
        _E4M3_LN_GAPS[key] = (up.to(device), dn.to(device))
    return _E4M3_LN_GAPS[key]


def ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """Smallest integer e with 2**e >= x, computed exactly."""
    m, e = torch.frexp(x.clamp_min(1e-30))
    return e - (m == 0.5).to(e.dtype)


def _scale_f(sb, nvfp4: bool = False, tscale=None):
    """Decoded per-block scale [out, nb] in fp32 (the ACTUAL scale, not assumed 2^k)."""
    if nvfp4:
        return e4m3_decode(sb) * tscale
    return torch.pow(2.0, sb.to(torch.float32) - 127.0)


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

    The unpacked code word lives in ``code_dtype``: uint8 while ``n_codes <= 256``
    (every shipped config), int16 up to 32768 codes (e.g. the full 15-rung E2M1 ladder
    with C >= 10), and the constructor raises ValueError beyond that. A uint8 store of
    a code > 255 would silently wrap mod 256 and corrupt the value rung.
    """

    #: largest code count the unpacked int16 store can hold (codes 0 .. 32767)
    MAX_CODES = 1 << 15

    def __init__(self, mags_unsigned: list[float], C: int, gap_mode: str = "directional",
                 scale_format: str = "e8m0"):
        if gap_mode not in ("directional", "min"):
            raise ValueError(f"gap_mode must be 'directional' or 'min', got {gap_mode!r}")
        if scale_format not in SCALE_FORMATS:
            raise ValueError(f"scale_format must be one of {SCALE_FORMATS}, got {scale_format!r}")
        #: "e8m0" (default): power-of-two scale per 32 elements (MXFP4).
        #: "nvfp4": E4M3 scale per 16 elements + one fp32 per-tensor scale (needs `tscale`
        #: wherever a scale is decoded; the scale is NOT a power of two).
        self.scale_format = scale_format
        self.nvfp4 = scale_format == "nvfp4"
        self.block = NVFP4_BLOCK if self.nvfp4 else BLOCK
        self.scale_bits = 8.0 / self.block           # 0.25 (e8m0/32) or 0.5 (e4m3/16)
        self.ladder_max = float(max(mags_unsigned))  # outer rung: 3 (wide) or 6 (e2m1)
        #: "directional" (default): tick normalised by the gap in the direction of
        #: movement -> unbiased in value space. "min": legacy, tick normalised by the
        #: smallest neighbour gap (biased outward drift on non-uniform ladders).
        self.gap_mode = gap_mode
        self.C = C
        self.acc_levels = 2 * C - 1
        self.signed = _build_signed(mags_unsigned)
        self.n_vals = len(self.signed)
        self.mids = _build_midpoints(self.signed)
        self.n_codes = self.n_vals * self.acc_levels
        if self.n_codes > self.MAX_CODES:
            raise ValueError(
                f"{self.n_vals} signed rungs x {self.acc_levels} accumulator levels (C={C}) = "
                f"{self.n_codes} codes; the unpacked code word is stored in uint8 (<= 256 codes) or "
                f"int16 (<= {self.MAX_CODES} codes), so this format is not representable. "
                f"Lower C or use fewer rungs.")
        #: dtype of the UNPACKED code word (the bit-packed state is always uint8 bytes):
        #: uint8 holds codes 0..255, so n_codes <= 256; int16 beyond (torch's uint16 has no
        #: arithmetic ops). Default configs (<= 256 codes) are unchanged: uint8.
        self.code_dtype = torch.uint8 if self.n_codes <= 256 else torch.int16
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
        # directional gaps: distance to the next rung up / down. At a ladder end the
        # missing neighbour reuses the inner gap (the carry is blocked there anyway).
        gu = [(s[i + 1] - s[i]) if i < self.n_vals - 1 else (s[i] - s[i - 1])
              for i in range(self.n_vals)]
        gd = [(s[i] - s[i - 1]) if i > 0 else (s[1] - s[0]) for i in range(self.n_vals)]
        self._gaps_up = torch.tensor(gu)
        self._gaps_down = torch.tensor(gd)
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
        return (mag_idx * self.acc_levels + (self.C - 1)).to(self.code_dtype)

    def gap_at(self, code: torch.Tensor) -> torch.Tensor:
        mag_idx = (code.to(torch.long) // self.acc_levels).clamp(0, self.n_vals - 1)
        return self._gaps.to(code.device)[mag_idx]

    def round_idx(self, a: torch.Tensor) -> torch.Tensor:
        """Nearest signed-rung index; exact midpoint ties round AWAY from zero.

        Plain ``bucketize(a, mids)`` sends a tie to the lower rung (towards -inf), so
        +0.25 -> 0 but -0.25 -> -0.5. After a rescale one octave up every +-0.5 rung
        sits exactly on the 0/0.5 midpoint, so only the positive side drained.
        Away-from-zero keeps the map odd: f(-a) == -f(a).
        """
        mids = self.mids.to(a.device)
        up = torch.bucketize(a, mids, right=True)      # tie -> upper rung (a > 0)
        dn = torch.bucketize(a, mids, right=False)     # tie -> lower rung (a < 0)
        return torch.where(a >= 0, up, dn).clamp(0, self.n_vals - 1)

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
            idx = self.round_idx(a)
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

    # ── NVFP4 scale (E4M3 per 16 + fp32 per tensor) ───────────────────────────

    def reencode(self, codes: torch.Tensor, s_old: torch.Tensor, s_new: torch.Tensor) -> torch.Tensor:
        """Re-express code words (value + pending accumulator) under a new per-element scale.

        Value: w = ladder[idx]*s_old -> idx' = round_idx(w / s_new) (ties away from zero,
        odd map). Accumulator: carried in VALUE units (acc/C * directional gap_old * s_old),
        re-expressed in the directional tick of the new rung and scale, rounded half away
        from zero and clamped to +-(C-1). Odd: reencode(mirror(c)) == mirror(reencode(c)).
        """
        dev = codes.device
        c = codes.to(torch.long)
        i_old = (c // self.acc_levels).clamp(0, self.n_vals - 1)
        acc = ((c % self.acc_levels) - (self.C - 1)).to(torch.float32)
        gu, gd = self._gaps_up.to(dev), self._gaps_down.to(dev)
        vu = acc / self.C * torch.where(acc >= 0, gu[i_old], gd[i_old]) * s_old
        w = self.signed.to(dev)[i_old] * s_old
        i_new = self.round_idx(w / s_new)
        g_new = torch.where(vu >= 0, gu[i_new], gd[i_new]) * s_new
        a = vu * self.C / g_new
        a = torch.sign(a) * torch.floor(a.abs() + 0.5)
        a = a.clamp(-(self.C - 1), self.C - 1)
        return (i_new * self.acc_levels + (a.to(torch.long) + self.C - 1)).to(self.code_dtype)

    def grow_scale(self, codes: torch.Tensor, sb: torch.Tensor, mask: torch.Tensor,
                   tscale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Bump the scale of the blocks in `mask` [out, nb] one step up and re-encode them.

        One step = next E4M3 code for nvfp4 (~6-12 %, capped at 448), x2 for e8m0 (capped
        at byte 254). Re-encoding: `reencode`. Elements of blocks outside `mask` are
        returned bit-identical. -> (codes, sb).
        """
        top = 126 if self.nvfp4 else 254            # 0x7E = 448; E8M0 255 is NaN
        mask = mask & (sb.to(torch.int32) < top)
        if not bool(mask.any()):
            return codes, sb
        sb_new = torch.where(mask, sb.to(torch.int32) + 1, sb.to(torch.int32)).to(torch.uint8)
        new = self.reencode(codes, self.expand_scale(sb, tscale), self.expand_scale(sb_new, tscale))
        return torch.where(mask.repeat_interleave(self.block, dim=1), new, codes), sb_new

    def rebase_tscale(self, codes: torch.Tensor, sb: torch.Tensor, tscale: torch.Tensor,
                      factor: float = 2.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """NVFP4 only: tensor_scale *= factor and every block byte re-fitted to keep its scale.

        The E4M3 block scale tops out at 448, so once the largest block sits on that code
        it cannot grow, whatever the pressure; the per-tensor scale is the only headroom.
        With factor 2 the byte moves by exactly -8 for normal codes (value preserved
        exactly); blocks that fall into the E4M3 subnormals are RTN-rounded and their
        codes re-encoded. -> (codes, sb, tscale).
        """
        assert self.nvfp4
        ts_new = tscale * factor
        sb_new = e4m3_encode(e4m3_decode(sb) * tscale / ts_new)
        s_old, s_new = self.expand_scale(sb, tscale), self.expand_scale(sb_new, ts_new)
        changed = s_new != s_old
        if bool(changed.any()):
            codes = torch.where(changed, self.reencode(codes, s_old, s_new), codes)
        return codes, sb_new, ts_new

    def scale_code_step(self, codes: torch.Tensor, sb: torch.Tensor, dlog: torch.Tensor,
                        tscale: torch.Tensor, gen: torch.Generator | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Learnable block scale (NVFP4 only): move each block's E4M3 byte by SR(dlog / gap) codes.

        `dlog` [out, nb] is the desired change of ln(block scale). One E4M3 code step is a
        DIRECTIONAL ln-gap (`e4m3_ln_gaps`, 6-12 % for normals, ln 2 in the subnormals); the
        number of steps is stochastically rounded, sign-symmetrically, so that
        E[change of ln s] == dlog (stateless: no extra accumulator; the unbiasedness is
        the same device as the per-weight tick SR). The byte is clamped to the valid positive
        codes [1, 126] (126 = 448: a block already there cannot grow here; MXNLinear uses
        `scale_code_step_rebasing`, which doubles the per-tensor scale first). The blocks
        whose byte moved are re-encoded value-preservingly with `reencode` (rung re-rounded
        by `round_idx`, accumulator carried in value units); other blocks are returned
        bit-identical. -> (codes, sb, changed [out, nb] bool).
        """
        assert self.nvfp4, "learnable block scale needs scale_format='nvfp4'"
        n = self._scale_steps(sb, dlog, gen)
        return self._apply_scale_steps(codes, sb, n, tscale)

    def scale_code_step_rebasing(self, codes: torch.Tensor, sb: torch.Tensor, dlog: torch.Tensor,
                                 tscale: torch.Tensor, gen: torch.Generator | None = None
                                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor, bool]:
        """`scale_code_step` that can grow past E4M3's 448 (NVFP4 only).

        `scale_code_step` clamps the byte to [1, 126], so a block on 448 can never grow, and the
        NVFP4 init puts the largest block exactly there (tensor_scale = amax / (t * 448)). Here
        the SR step counts are drawn first (same draw, same RNG use as `scale_code_step`); if any
        block would end above byte 126, the per-tensor scale is doubled first (`rebase_tscale`:
        every block byte -8, decoded weights preserved) and the steps are applied to the rebased
        bytes. One rebase buys 8 codes (an octave) of headroom for the top block, so it cannot
        thrash. Without any overshoot this is bit-identical to `scale_code_step`.
        -> (codes, sb, tscale, changed [out, nb] bool, rebased).
        """
        assert self.nvfp4, "learnable block scale needs scale_format='nvfp4'"
        n = self._scale_steps(sb, dlog, gen)
        rebased = bool(((sb.to(torch.long) + n) > 126).any())
        if rebased:
            codes, sb, tscale = self.rebase_tscale(codes, sb, tscale)
        codes, sb, changed = self._apply_scale_steps(codes, sb, n, tscale)
        return codes, sb, tscale, changed, rebased

    def _scale_steps(self, sb: torch.Tensor, dlog: torch.Tensor,
                     gen: torch.Generator | None = None) -> torch.Tensor:
        """Signed E4M3 code steps per block [out, nb] (int64): SR(dlog / directional ln-gap)."""
        up, dn = e4m3_ln_gaps(sb.device)
        b = sb.to(torch.long)
        k = dlog / torch.where(dlog >= 0, up[b], dn[b])
        return (torch.sign(k) * self._sr_int(k.abs(), gen)).to(torch.long)

    def _apply_scale_steps(self, codes: torch.Tensor, sb: torch.Tensor, n: torch.Tensor,
                           tscale: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move bytes by `n` steps (clamped to [1, 126]) and re-encode the blocks that moved."""
        sb_new = (sb.to(torch.long) + n).clamp(1, 126).to(torch.uint8)
        changed = sb_new != sb
        if not bool(changed.any()):
            return codes, sb, changed
        new = self.reencode(codes, self.expand_scale(sb, tscale), self.expand_scale(sb_new, tscale))
        return torch.where(changed.repeat_interleave(self.block, dim=1), new, codes), sb_new, changed

    def nvfp4_fit(self, w: torch.Tensor, target: float | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """(E4M3 block-scale bytes [out, in/16], fp32 per-tensor scale ()) for `w`.

        tensor_scale = amax / (target * 448); byte = RTN_e4m3(block_amax / (target * tensor_scale)).
        `target` defaults to the ladder's outer rung (3 or 6), so block maxima sit on it.
        The per-tensor scale is global, so it is the one number that changes on every
        refresh when the weight amax moves; the block bytes are re-fitted with it.
        """
        t = self.ladder_max if target is None else float(target)
        out_f, in_f = w.shape
        w = w.detach().float()
        blocks = w.reshape(out_f, in_f // self.block, self.block)
        bamax = blocks.abs().amax(dim=-1)
        tscale = (w.abs().amax().clamp_min(1e-30) / (t * E4M3_MAX)).to(torch.float32)
        sb = e4m3_encode(bamax / (t * tscale))
        return sb, tscale

    def fit_scale(self, w: torch.Tensor, headroom: float = 1.5, mode: str = "headroom",
                  clip_frac: float = 0.0, nvfp4_target: float | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Format-dispatching scale fit -> (scale bytes, tensor scale or None).

        e8m0: the legacy max / MSE rules with `headroom` (= target amax/s). nvfp4:
        `nvfp4_fit` with `nvfp4_target` (default: ladder max, so block amax -> outer rung;
        `headroom` is ignored); mode "mse" is not implemented.
        """
        if self.nvfp4:
            if mode != "headroom":
                raise NotImplementedError("scale_mode='mse' is E8M0-only")
            return self.nvfp4_fit(w, nvfp4_target)
        if mode == "mse":
            return self.scale_bytes_mse(w, headroom, clip_frac), None
        return self.scale_bytes(w, headroom), None

    def decode_scale(self, sb: torch.Tensor, tscale: torch.Tensor | None = None) -> torch.Tensor:
        """Actual fp32 scale per block [out, nb] (E8M0: 2^(b-127); NVFP4: e4m3(b) * tscale)."""
        if self.nvfp4 and tscale is None:
            raise ValueError("scale_format='nvfp4' needs the per-tensor scale `tscale`")
        return _scale_f(sb, self.nvfp4, tscale)

    def expand_scale(self, sb: torch.Tensor, tscale: torch.Tensor | None = None) -> torch.Tensor:
        """decode_scale broadcast over the elements of each block -> [out, in]."""
        return self.decode_scale(sb, tscale).repeat_interleave(self.block, dim=1)

    # ── quantize / dequantize ─────────────────────────────────────────────────

    def quantize_scaled(self, w: torch.Tensor, headroom: float = 1.5, mode: str = "headroom",
                        clip_frac: float = 0.0, nvfp4_target: float | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(codes, scale bytes, tensor scale or None); works for both scale formats."""
        sb, tscale = self.fit_scale(w, headroom, mode, clip_frac, nvfp4_target)
        a = w / self.expand_scale(sb, tscale)
        codes = (self.round_idx(a) * self.acc_levels + (self.C - 1)).to(self.code_dtype)
        return codes, sb, tscale

    def quantize(self, w: torch.Tensor, headroom: float = 1.5
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.nvfp4:
            raise ValueError("nvfp4 also yields a per-tensor scale: use quantize_scaled()")
        sb = self.scale_bytes(w, headroom)
        s = self.scale_floats(sb).repeat_interleave(BLOCK, dim=1)
        a = w / s
        idx = self.round_idx(a)
        codes = (idx * self.acc_levels + (self.C - 1)).to(self.code_dtype)
        return codes, sb

    def dequantize(self, codes: torch.Tensor, sb: torch.Tensor,
                   tscale: torch.Tensor | None = None) -> torch.Tensor:
        s = self.expand_scale(sb, tscale)
        mag_idx = (codes.to(torch.long) // self.acc_levels).clamp(0, self.n_vals - 1)
        return self.signed.to(codes.device)[mag_idx] * s

    def dequantize_fast(self, codes: torch.Tensor, sb: torch.Tensor,
                        tscale: torch.Tensor | None = None) -> torch.Tensor:
        """Fused decode: single index_select + multiply (fewer kernel launches).
        Uses torch.compile when enabled (call enable_compile() first)."""
        if self.nvfp4 and tscale is None:
            raise ValueError("scale_format='nvfp4' needs the per-tensor scale `tscale`")
        return _get_compiled(_decode_core)(
            codes, sb, self.signed.to(codes.device), self.acc_levels, self.n_vals,
            self.block, self.nvfp4, tscale)

    # ── update ────────────────────────────────────────────────────────────────

    @staticmethod
    def _sr_int(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        fl = torch.floor(x)
        return fl + (torch.rand(x.shape, generator=gen, device=x.device) < (x - fl)).to(x.dtype)

    def step(self, codes: torch.Tensor, scale: torch.Tensor,
             grad_hat: torch.Tensor, lr: float,
             gen: torch.Generator | None = None,
             tscale: torch.Tensor | None = None,
             return_clip: bool = False):
        """One per-param integrator update. Returns the new code word
        (or ``(codes, clip)`` with ``return_clip=True``: ``clip`` is a bool mask of the
        weights whose carry would have moved them past an outer rung and was clipped there.
        It is computed from the same post-SR carry on the side and never feeds back, so
        numerics are bit-identical with and without it).

        The accumulator lives *inside* the code (the low ``2C-1`` residue), so
        there is no separate accumulator tensor: the update decodes it, applies
        the tick, and re-encodes carry + remainder into one word.

        The tick is `gap * s` with the ACTUAL decoded block scale s (E8M0: 2^k;
        NVFP4: e4m3 * `tscale`, not a power of two). Only the pre-SR conversion from
        value units to ticks reads s; carry/remainder are in tick units and need no scale.
        """
        if self.nvfp4 and tscale is None:
            raise ValueError("scale_format='nvfp4' needs the per-tensor scale `tscale`")
        if self.gap_mode == "directional":
            acc_f = _get_compiled(_step_pre_sr_dir)(
                codes, scale, grad_hat, lr, self.C,
                self._gaps_up, self._gaps_down, self.acc_levels, self.n_vals,
                self.block, self.nvfp4, tscale)
            cc = self._sr_int(acc_f, gen)
            new_mag, rem_f = _get_compiled(_step_mid_dir)(
                cc, codes, self.C, self._gaps_up, self._gaps_down,
                self.acc_levels, self.n_vals, self._mid_idx)
            rem = self._sr_int(rem_f, gen)      # re-quantise leftover to the new rung's tick
            new = _get_compiled(_step_enc_dir)(new_mag, rem, self.C, self.acc_levels,
                                               self.code_dtype)
            return (new, self._clip_mask(cc, codes)) if return_clip else new
        acc_f = _get_compiled(_step_pre_sr)(
            codes, scale, grad_hat, lr, self.C,
            self._gaps, self.acc_levels, self.n_vals, self._mid_idx,
            self.block, self.nvfp4, tscale)
        # SR stays eager: it must consume the global/per-call RNG stream in order
        cc = self._sr_int(acc_f, gen)
        new = _get_compiled(_step_post_sr)(
            cc, codes, self.C, self.acc_levels, self.n_vals, self._mid_idx, self.code_dtype)
        return (new, self._clip_mask(cc, codes)) if return_clip else new

    def _clip_mask(self, cc: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """True where this step's carry would leave the ladder (clipped at an outer rung).

        Same carry as the update (`trunc(cc / C)`), tested against the same bounds as
        `_step_mid_dir` / `_step_post_sr`; read-only, so it cannot change numerics.
        """
        carry = torch.trunc(cc / self.C).to(torch.long)
        mag = (codes.to(torch.long) // self.acc_levels).clamp(0, self.n_vals - 1)
        moved = (mag - self._mid_idx) + carry
        return (moved > self.n_vals - 1 - self._mid_idx) | (moved < -self._mid_idx)

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
        """[out, packed_width] uint8 -> [out, in] codes (dtype ``code_dtype``)."""
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
        out = torch.empty(out_f, in_features, dtype=self.code_dtype, device=dev)
        for r0, r1 in self._row_chunks(out_f, in_features):
            p = packed[r0:r1].reshape(-1, packed.shape[1] // b, b).to(torch.int32)
            bitmat = ((p.unsqueeze(-1) >> bit_pos) & 1).reshape(-1, packed.shape[1] // b, b * 8)
            bitmat = bitmat[..., :g * bits].reshape(-1, in_features // g, g, bits)
            c = (bitmat << shifts).sum(-1)
            out[r0:r1] = c.reshape(r1 - r0, in_features).to(self.code_dtype)
        return out


    # ── accounting ───────────────────────────────────────────────────────────

    def bits_per_param(self, numel: int | None = None) -> float:
        """Code bits + scale bits (+ amortised per-tensor fp32); per-row RMS excluded.

        e8m0: bits + 8/32 = bits + 0.25.  nvfp4: bits + 8/16 = bits + 0.5, plus 32/numel
        for the fp32 tensor scale when `numel` is given.
        """
        extra = 32.0 / numel if (self.nvfp4 and numel) else 0.0
        return float(self.bits) + self.scale_bits + extra

    @property
    def code_to_value(self) -> torch.Tensor:
        """Precomputed lookup table: code -> signed magnitude (for Triton kernel)."""
        return self._code_to_value

    def persistent_bytes(self, out_f: int, in_f: int) -> int:
        """Bytes actually stored: bit-packed codes + block scales + per-row RMS
        (+ 4 bytes for the fp32 tensor scale under nvfp4)."""
        return (out_f * self.packed_width(in_f) + out_f * (in_f // self.block)
                + out_f * 4 + (4 if self.nvfp4 else 0))


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


def _decode_core(codes, sb, signed_vals, acc_levels, n_vals, block=BLOCK,
                 nvfp4=False, tscale=None):
    """Fused decode: codes + scale -> float weight."""
    s = _scale_f(sb, nvfp4, tscale).repeat_interleave(block, dim=1)
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    vals = torch.index_select(signed_vals, 0, mag_idx.reshape(-1))
    return vals.reshape_as(s) * s


def _step_pre_sr(codes, scale_bytes, grad_hat, lr, C, gaps, acc_levels, n_vals,
                 mid_idx, block=BLOCK, nvfp4=False, tscale=None):
    """Accumulator read from the code + this step's ticks (pre-SR)."""
    s = _scale_f(scale_bytes, nvfp4, tscale).repeat_interleave(block, dim=1)
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    gap = gaps.to(codes.device)[mag_idx] * s
    acc = ((codes.to(torch.long) % acc_levels) - (C - 1)).to(torch.float32)
    return acc + (-lr * grad_hat) / gap * C


def _step_pre_sr_dir(codes, scale_bytes, grad_hat, lr, C, gaps_up, gaps_down,
                     acc_levels, n_vals, block=BLOCK, nvfp4=False, tscale=None):
    """Directional-gap tick (pre-SR). Unbiased in value space.

    The accumulator is a progress meter measured in the gap of its own sign:
    acc>=0 -> acc/C of the gap to the rung above, acc<0 -> |acc|/C of the gap
    below. Convert to value units, add -lr*grad, convert back using the gap of the
    NEW sign (so a sign flip is exact, not a rescale-free reinterpretation).
    """
    s = _scale_f(scale_bytes, nvfp4, tscale).repeat_interleave(block, dim=1)
    mag_idx = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    gu = gaps_up.to(codes.device)[mag_idx] * s
    gd = gaps_down.to(codes.device)[mag_idx] * s
    acc = ((codes.to(torch.long) % acc_levels) - (C - 1)).to(torch.float32)
    v = acc / C * torch.where(acc >= 0, gu, gd) - lr * grad_hat      # value units
    return v * C / torch.where(v >= 0, gu, gd)


def _step_mid_dir(cc, codes, C, gaps_up, gaps_down, acc_levels, n_vals, mid_idx):
    """Carry, clamp, and re-express the leftover in the NEW rung's directional tick."""
    carry = torch.trunc(cc / C)
    rem = cc - carry * C
    mag = (codes.to(torch.long) // acc_levels).clamp(0, n_vals - 1)
    moved = (mag - mid_idx) + carry.to(torch.long)
    new_idx = moved.clamp(-mid_idx, n_vals - 1 - mid_idx)
    blocked = moved != new_idx
    new_mag = new_idx + mid_idx
    gu = gaps_up.to(codes.device)
    gd = gaps_down.to(codes.device)
    up = rem >= 0
    g_old = torch.where(up, gu[mag], gd[mag])
    g_new = torch.where(up, gu[new_mag], gd[new_mag])
    # blocked at a ladder end: drop the pressure (no wind-up that would make the wall sticky)
    rem_f = torch.where(blocked, torch.zeros_like(rem), rem * g_old / g_new)
    return new_mag, rem_f.clamp(-(C - 1), C - 1)


def _step_enc_dir(new_mag, rem, C, acc_levels, out_dtype=torch.uint8):
    return (new_mag * acc_levels + (rem.to(torch.long) + (C - 1))).to(out_dtype)


def _step_post_sr(cc, codes, C, acc_levels, n_vals, mid_idx, out_dtype=torch.uint8):
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
    return (new_mag * acc_levels + (rem + (C - 1))).to(out_dtype)
