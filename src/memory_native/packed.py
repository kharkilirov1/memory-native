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

    # storage hooks ------------------------------------------------------------
    def _all_codes(self) -> torch.Tensor:
        return unpack_codes(self.state, self.in_features)

    def _dense_weight(self, dtype: torch.dtype) -> torch.Tensor:
        t, _ = decode_state(self._all_codes(), self.C)
        return self.scale.to(dtype) * t.to(dtype)

    def _decode_rows(self, lo: int, hi: int):
        codes = unpack_codes(self.state[lo:hi], self.in_features)
        return decode_state(codes, self.C)

    def _write_rows(self, lo: int, hi: int, t: torch.Tensor, c: torch.Tensor) -> None:
        self.state[lo:hi].copy_(pack_codes(encode_state(t, c, self.C)))

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
