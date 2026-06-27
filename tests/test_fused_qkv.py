import torch

from memory_native import CounterQKVLinear, GPTConfig, ReversibleGPT
from memory_native.packed import PackedRMSCounterLinear


def test_fused_qkv_equals_three_separate_bitexact():
    """A fused d->3d counter layer must be identical to three separate d->d counter layers fed
    the same input (its rows are independent), so its forward equals their concatenation."""
    torch.manual_seed(0)
    d = 64
    fused = CounterQKVLinear(d, "counter_packed", C=11)
    # three separate layers carrying the fused layer's per-row state/scale/v
    sep = [PackedRMSCounterLinear(d, d, C=11) for _ in range(3)]
    for j, s in enumerate(sep):
        rows = slice(j * d, (j + 1) * d)
        s.state.copy_(fused.proj.state[rows])
        s.scale.copy_(fused.proj.scale[rows])
        s.v.copy_(fused.proj.v[rows])

    x = torch.randn(8, 5, d)
    q, k, v = fused(x)
    assert torch.equal(q, sep[0](x))
    assert torch.equal(k, sep[1](x))
    assert torch.equal(v, sep[2](x))


def test_fused_qkv_model_has_fewer_counter_layers_and_trains():
    """ReversibleGPT(fused_qkv=True) collapses q/k/v into one layer (fewer saved activations)
    and still trains end-to-end."""
    torch.manual_seed(0)
    cfg = GPTConfig(48, 16, 2, 2, 32)  # vocab, block, n_layer, n_head, n_embd
    sep = ReversibleGPT(cfg, "counter_packed", fused_qkv=False, C=11, act_save_bits=4)
    fus = ReversibleGPT(cfg, "counter_packed", fused_qkv=True, C=11, act_save_bits=4)
    # per block: separate has q,k,v,proj,fc,fc2 = 6 counter layers; fused has qkv,proj,fc,fc2 = 4
    assert len(fus.counter_layers()) < len(sep.counter_layers())
    assert len(fus.counter_layers()) == cfg.n_layer * 4

    idx = torch.randint(0, 48, (3, 16))
    tgt = torch.randint(0, 48, (3, 16))
    fus.train()
    _, loss = fus(idx, tgt)
    loss.backward()
    assert torch.isfinite(loss)
