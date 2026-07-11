"""Packed 6-bit storage: the real sub-byte persistent footprint.

Each finite-state code needs only ceil(log2(63)) = 6 bits, so 4 codes pack into 3 bytes.
PackedRMSCounterLinear stores its state packed -- persistent state is 0.75 byte/weight, not
1.0 -- and unpacks at the storage boundary (forward weight, per-tile decode, per-tile write).
The update math is inherited unchanged from RMSCounterLinear via the storage hooks
(_dense_weight / _decode_rows / _write_rows).

This is pure PyTorch and matches the engine's packing bit-for-bit (kernels/compact_counter.cl
cc_pack4/cc_unpack4), so a counter model trained here exports to the same packed format.
"""
from __future__ import annotations

import torch

from .counter import RMSCounterLinear, decode_state, encode_state

__all__ = ["pack_codes", "unpack_codes", "PackedRMSCounterLinear"]


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """codes uint8 [out, in] in [0,63] -> packed uint8 [out, (in//4)*3] (4 codes / 3 bytes)."""
    out, in_ = codes.shape
    assert in_ % 4 == 0, "in_features must be divisible by 4 for 6-bit packing"
    c = codes.to(torch.int32).reshape(out, in_ // 4, 4)
    c0, c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    p0 = (c0 & 0x3F) | ((c1 & 0x03) << 6)
    p1 = ((c1 >> 2) & 0x0F) | ((c2 & 0x0F) << 4)
    p2 = ((c2 >> 4) & 0x03) | ((c3 & 0x3F) << 2)
    packed = torch.stack([p0, p1, p2], dim=-1)  # [out, in//4, 3]
    return (packed & 0xFF).to(torch.uint8).reshape(out, (in_ // 4) * 3)


def unpack_codes(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """packed uint8 [out, (in//4)*3] -> codes uint8 [out, in] in [0,63]."""
    out = packed.shape[0]
    gpr = in_features // 4
    p = packed.to(torch.int32).reshape(out, gpr, 3)
    p0, p1, p2 = p[..., 0], p[..., 1], p[..., 2]
    c0 = p0 & 0x3F
    c1 = ((p0 >> 6) | (p1 << 2)) & 0x3F
    c2 = ((p1 >> 4) | (p2 << 4)) & 0x3F
    c3 = (p2 >> 2) & 0x3F
    codes = torch.stack([c0, c1, c2, c3], dim=-1)  # [out, gpr, 4]
    return codes.to(torch.uint8).reshape(out, in_features)


class PackedRMSCounterLinear(RMSCounterLinear):
    """RMSCounterLinear whose persistent `state` buffer is packed to 6 bits (0.75 B/weight).

    Same learning dynamics as RMSCounterLinear; only the storage layout differs. Codes are
    unpacked transiently for the GEMM and the per-tile update, then repacked, so the resident
    state stays at 0.75 byte/weight.
    """

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)  # builds unpacked state [out, in], scale, v
        codes = self.state  # uint8 [out, in]
        del self._buffers["state"]
        self.register_buffer("state", pack_codes(codes))  # [out, (in//4)*3]
        self._sr_step = 0  # per-call seed for the kernel's deterministic stochastic rounding

    def _fused_update(self, lo: int, hi: int, grad_w: torch.Tensor) -> bool:
        """One-launch Triton RMS+SR update for a packed row range.

        Originally the fused kernel only fired for the full matrix (`lo==0, hi==out`).  That made
        the *fast* path and the *low-peak* path mutually exclusive: `tile_rows>0` used cuBLAS for a
        small grad_w tile but then fell back to the slow torch transition.  The kernel is row-local,
        so a contiguous row slice of the packed state is sufficient.

        This gives a practical middle ground between:
          * full fast path: materialize one [out,in] grad_w and fuse the transition;
          * strict from-IO: no grad_w but a hand-written GEMM (very slow on T4);
          * tiled fast path: materialize only [tile_rows,in] grad_w, then fuse the transition.

        The deterministic hash-SR seed is advanced per tile, so different row tiles do not reuse the
        same rounding stream even though the Triton kernel sees local row indices 0..tile_rows-1.
        """
        from .fused_update import HAS_TRITON, triton_counter_update
        if not (HAS_TRITON and grad_w.is_cuda and self.use_rms
                and self.pulse_mode == "direct"
                and self.rms_mode == "exact" and self.scale_rebase == "eager"
                and not self.decimate_updates):  # kernel doesn't report flip-rate; torch path does
            return False
        if grad_w.shape != (hi - lo, self.in_features):
            return False
        seed = self._sr_step
        self._sr_step += 1
        # local_grad_clip rides into the kernel (folded into the RMS denominator), so the
        # stable recovery recipe (clip=1.0) gets the fused path too -- the clip==0 gate is gone.
        triton_counter_update(self.state[lo:hi], self.scale[lo:hi], self.v[lo:hi], grad_w.contiguous(),
                              C=self.C, lr=self.lr, lr_scale=self.lr_scale,
                              rms_beta=self.rms_beta, rms_eps=self.rms_eps, seed=seed,
                              clip=self.local_grad_clip)
        # The kernel mutates the packed state directly, bypassing _write_rows/_refresh_t_cache.
        # Refresh only the affected rows instead of rebuilding the whole derived T cache.
        if self.cache_mode != "none":
            t_new, _ = self._decode_rows(lo, hi)
            self._refresh_t_cache(lo, hi, t_new)
        return True

    # storage hooks ------------------------------------------------------------
    def _all_codes(self) -> torch.Tensor:
        return unpack_codes(self.state, self.in_features)

    # _dense_weight is inherited from the base: it routes through _visible_t (derived cache when
    # cache_mode != "none", else _decode_rows -> the packed unpack below). No override needed.

    def _decode_rows(self, lo: int, hi: int):
        codes = unpack_codes(self.state[lo:hi], self.in_features)
        return decode_state(codes, self.C)

    def _write_rows(self, lo: int, hi: int, t: torch.Tensor, c: torch.Tensor) -> None:
        self.state[lo:hi].copy_(pack_codes(encode_state(t, c, self.C)))
        self._refresh_t_cache(lo, hi, t)

    @torch.no_grad()
    def state_statistics(self) -> dict[str, float]:
        t, c = decode_state(self._all_codes(), self.C)
        return {
            "minus": float((t == -1).float().mean()),
            "zero": float((t == 0).float().mean()),
            "plus": float((t == 1).float().mean()),
            "counter_abs_mean": float(c.float().abs().mean()),
            "counter_edge": float((c.abs() == self.C - 1).float().mean()),
            "scale_mean": float(self.scale.mean()),
        }
