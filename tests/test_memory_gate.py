import torch

from memory_native import CONFIGS, GPT, memory_report


def test_counter_has_no_per_weight_optimizer_state():
    """The counter GPT must carry no FP master weight and no Adam moments for its ternary
    weights: its trainable Parameters are only embeddings/norms/head, and its per-weight
    state is the 1-byte uint8 buffer (no float weight of shape [out,in])."""
    cfg = CONFIGS["micro"]
    counter = GPT("counter_rms", cfg)
    dense = GPT("dense", cfg)

    cr = memory_report(counter)
    dr = memory_report(dense)

    assert cr["counter_weights"] > 0
    # counter persistent state must be below dense (which holds fp32 weights + will get Adam)
    assert cr["persistent_bytes"] < dr["persistent_bytes"]

    # each counter layer must expose NO Parameters (it self-updates; the weight lives in the
    # uint8 state buffer, not as a float Parameter an optimizer would own).
    for m in counter.counter_layers():
        assert list(m.parameters()) == [], "counter layer leaked a trainable Parameter"
        assert m.state.dtype == torch.uint8


def test_counter_state_is_uint8():
    counter = GPT("counter_rms", CONFIGS["micro"])
    for m in counter.counter_layers():
        assert m.state.dtype == torch.uint8
        assert m.state.numel() == m.in_features * m.out_features
