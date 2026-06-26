import math

import torch

from memory_native import RMSCounterLinear
from memory_native.actquant import (
    effective_bits,
    pack_int4,
    quantize_codes,
    stochastic_quantize,
    unpack_int4,
)


def test_pack_int4_roundtrip_and_halves_bytes():
    torch.manual_seed(0)
    codes = torch.randint(-7, 8, (133,), dtype=torch.int8)  # odd length on purpose
    packed = pack_int4(codes)
    assert packed.dtype == torch.uint8
    assert packed.numel() == (codes.numel() + 1) // 2          # ~0.5 byte/elem
    back = unpack_int4(packed, codes.numel())
    assert torch.equal(back, codes)


def test_quantizer_is_unbiased():
    torch.manual_seed(0)
    x = torch.randn(500, 32)
    # average many stochastic quantizations -> should converge to x (E[Q(x)|x]=x)
    avg = torch.stack([stochastic_quantize(x, 4) for _ in range(600)]).mean(0)
    assert (avg - x).abs().mean().item() < 0.02


def test_codes_fit_bit_width():
    torch.manual_seed(0)
    x = torch.randn(8, 64)
    for bits in (3, 4, 8):
        codes, scale = quantize_codes(x, bits)
        lim = (1 << (bits - 1)) - 1
        assert int(codes.abs().max()) <= lim
        assert scale.shape == (8, 1)


def test_effective_bits_includes_scale():
    # one fp16 scale amortized over a row of length 32 -> +0.5 bit/elem
    assert abs(effective_bits(4, 32) - 4.5) < 1e-9


def test_low_bit_saved_activation_still_recovers_teacher():
    """The counter update needs only E[Q(x)|x]=x, so int8/4/3 saved activations must train
    as well as fp (the activation-memory lever from the deep-v2 research)."""
    def recover(bits):
        torch.manual_seed(0)
        n, N, C = 16, 256, 11
        ts = 0.25
        base = math.sqrt(3.0 / (2.0 * n))
        tw = torch.randint(-1, 2, (n, n)).float()
        x = torch.randn(N, n)
        y = x @ (ts * tw).t()
        lay = RMSCounterLinear(n, n, C=C, lr=0.005, lr_scale=0.0,
                               init_gain=ts / base, act_save_bits=bits).train()
        for _ in range(400):
            ((lay(x) - y) ** 2).mean().backward()
        with torch.no_grad():
            return ((lay(x) - y) ** 2).mean().item()

    fp = recover(0)
    for bits in (8, 4, 3):
        mse = recover(bits)
        assert mse < 5e-3, (bits, mse)
        assert mse <= fp + 2e-3, (bits, mse, fp)
