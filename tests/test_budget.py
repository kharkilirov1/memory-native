from memory_native import training_budget


def test_reproduces_deep_v2_numbers():
    """The deep-v2 doc's headline budget: L=24, d=2048, V=50257, seq=1024, 6-bit state,
    4-bit reversible activations -> ~21.10 / 2.06 / 1.44 GiB."""
    base, counter, lb = training_budget(
        layers=24, d_model=2048, vocab=50257, seq=1024, batch=1,
        state_bits=6.0, counter_act_bits=4.0, counter_policy="reversible")
    assert abs(base.total_gib - 21.10) < 0.05
    assert abs(counter.total_gib - 2.06) < 0.05
    assert abs(lb.total_gib - 1.44) < 0.05


def test_counter_has_no_grad_or_optimizer_pool():
    _, counter, _ = training_budget()
    assert counter.grad_gib == 0.0 and counter.optim_gib == 0.0


def test_more_state_bits_costs_more():
    b5 = training_budget(state_bits=5)[1].persistent_gib
    b6 = training_budget(state_bits=6)[1].persistent_gib
    assert b6 > b5
