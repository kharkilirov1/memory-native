import torch

from memory_native import C_DEFAULT, decode_state, encode_state


def test_roundtrip_all_states():
    for C in (8, 11):
        ts, cs = [], []
        for t in (-1, 0, 1):
            for c in range(-(C - 1), C):
                ts.append(t)
                cs.append(c)
        t = torch.tensor(ts, dtype=torch.int16)
        c = torch.tensor(cs, dtype=torch.int16)
        state = encode_state(t, c, C)
        assert state.dtype == torch.uint8
        assert int(state.max()) < 3 * (2 * C - 1)
        t2, c2 = decode_state(state, C)
        assert torch.equal(t2.to(torch.int16), t)
        assert torch.equal(c2.to(torch.int16), c)


def test_default_C_fits_byte():
    assert 3 * (2 * C_DEFAULT - 1) <= 256
