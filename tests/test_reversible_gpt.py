import torch

from memory_native import GPTConfig, ReversibleGPT


def test_full_method_gpt_trains_and_counters_update():
    """The full method end-to-end: counter linears inside an O(1) reversible transformer must
    learn (counters self-update through the reversible recompute) on a learnable task."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=3, n_head=4, n_embd=64)
    m = ReversibleGPT(cfg, kind="counter_packed", lr=3e-3, lr_scale=2e-4, C=11, act_save_bits=4).train()
    assert len(m.counter_layers()) == 3 * 6  # q,k,v,proj,fc,fc2 per block
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=2e-3)

    def batch():
        start = torch.randint(0, cfg.vocab_size, (8, 1))
        off = torch.arange(cfg.block_size + 1)[None]
        seq = (start + off) % cfg.vocab_size
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    first = None
    for _ in range(80):
        x, y = batch()
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    flips = sum(int(c.weight_flips) for c in m.counter_layers())
    assert flips > 0, "counter layers never updated inside the reversible GPT"
    assert loss.item() < first, (first, loss.item())
