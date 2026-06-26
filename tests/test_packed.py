import math

import torch

from memory_native import (
    PackedRMSCounterLinear,
    RMSCounterLinear,
    encode_state,
    pack_codes,
    unpack_codes,
)


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    for C in (8, 11):
        out, in_ = 7, 64
        t = torch.randint(-1, 2, (out, in_), dtype=torch.int16)
        c = torch.randint(-(C - 1), C, (out, in_), dtype=torch.int16)
        codes = encode_state(t, c, C)            # uint8 [out,in] in [0,63]
        packed = pack_codes(codes)
        assert packed.shape == (out, (in_ // 4) * 3)
        back = unpack_codes(packed, in_)
        assert torch.equal(back, codes)


def test_packed_state_is_three_quarter_byte():
    layer = PackedRMSCounterLinear(64, 64, C=11)
    logical = layer.in_features * layer.out_features
    assert layer.state.dtype == torch.uint8
    assert layer.state.numel() == logical * 3 // 4   # 0.75 byte/weight, not 1.0


def test_packed_matches_unpacked_dynamics():
    """Same seed -> packed and unpacked layers must train identically (storage-only change)."""
    n, N, C = 16, 256, 11
    teacher_scale = 0.25
    base = math.sqrt(3.0 / (2.0 * n))

    def make(cls):
        torch.manual_seed(0)
        m = cls(n, n, C=C, lr=0.005, lr_scale=0.0, init_gain=teacher_scale / base)
        m.train()
        return m

    torch.manual_seed(123)
    tw = torch.randint(-1, 2, (n, n)).float()
    x = torch.randn(N, n)
    y = x @ (teacher_scale * tw).t()

    plain, packed = make(RMSCounterLinear), make(PackedRMSCounterLinear)
    for _ in range(200):
        torch.manual_seed(1)  # same SR randomness for both
        ((plain(x) - y) ** 2).mean().backward()
        torch.manual_seed(1)
        ((packed(x) - y) ** 2).mean().backward()

    with torch.no_grad():
        wp = plain._dense_weight(torch.float32)
        wq = packed._dense_weight(torch.float32)
    assert torch.allclose(wp, wq), (wp - wq).abs().max()
