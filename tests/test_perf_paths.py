"""Perf-path safety witnesses: sdpa attention under the counter reuse guard, the top-k
teacher logit cache, on-device diagnostic counters, and the opt-in compiled update chain."""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM

from memory_native.counter import _COMPILE_STATE, RMSCounterLinear
from memory_native.donor.qwen import qwen_to_counter
from memory_native.recovery import TopKLogitCache, kd_divergence


def _tiny(seed=0, attn="sdpa"):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    cfg._attn_implementation = attn
    return Qwen2ForCausalLM(cfg)


def test_sdpa_full_steps_do_not_trip_reuse_guard():
    """SDPA fuses attention math only -- each counter projection still runs exactly once per
    forward, so two full train steps must not raise the 'reused before backward' guard."""
    student = _tiny(0, attn="sdpa")
    qwen_to_counter(student)
    student.train()
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        out = student(ids, labels=ids)
        out.loss.backward()          # guard would raise here on a double forward


def test_sdpa_matches_eager_inference():
    a = _tiny(0, attn="eager")
    b = _tiny(0, attn="sdpa")        # same seed => identical weights, only attention differs
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        la = a(ids).logits
        lb = b(ids).logits
    assert torch.allclose(la, lb, atol=1e-4), (la - lb).abs().max()


class _CountingTeacher:
    """Deterministic toy TeacherSource with a forward-call counter."""

    def __init__(self, vocab=64):
        torch.manual_seed(3)
        self.table = torch.randn(64, vocab)
        self.calls = 0

    def logits(self, input_ids):
        self.calls += 1
        return self.table[input_ids]


def test_topk_cache_replays_without_teacher_forward():
    src = _CountingTeacher()
    cache = TopKLogitCache(src, k=64)          # k == vocab -> lossless up to fp16 storage
    ids = torch.randint(0, 64, (2, 8))

    first = cache.logits(ids)                  # miss: real teacher forward
    second = cache.logits(ids)                 # hit: replayed, no teacher call
    assert src.calls == 1 and cache.misses == 1 and cache.hits == 1

    p_first = torch.softmax(first / 2.0, dim=-1)
    p_second = torch.softmax(second / 2.0, dim=-1)
    assert torch.allclose(p_first, p_second, atol=2e-3)   # fp16 round-trip tolerance

    other = torch.randint(0, 64, (2, 8)) + 0   # different content -> new miss
    other[0, 0] = (ids[0, 0] + 1) % 64
    cache.logits(other)
    assert src.calls == 2 and cache.misses == 2


def test_topk_cache_kd_is_finite_at_small_k():
    src = _CountingTeacher()
    cache = TopKLogitCache(src, k=8)           # heavy truncation: tail mass dropped
    ids = torch.randint(0, 64, (2, 8))
    cache.logits(ids)
    replay = cache.logits(ids)
    student_logits = torch.randn(2, 8, 64, requires_grad=True)
    loss = kd_divergence(student_logits, replay, 2.0)
    assert torch.isfinite(loss)                # large-negative fill must not produce NaN
    loss.backward()                            # and it must be differentiable for the student
    assert torch.isfinite(student_logits.grad).all()


def test_diagnostic_counters_accumulate_on_device():
    """weight_flips / update_events accumulate as tensors (no per-tile host sync) and stay
    readable through int() exactly as before."""
    torch.manual_seed(0)
    layer = RMSCounterLinear(16, 8, lr=0.5)    # big lr => events guaranteed
    layer.train()
    x = torch.randn(4, 16)
    (layer(x) ** 2).sum().backward()
    assert int(layer.update_events) > 0
    assert int(layer.weight_flips) >= 0
    e1 = int(layer.update_events)
    (layer(x) ** 2).sum().backward()
    assert int(layer.update_events) >= e1


def test_decimation_period_still_adapts():
    torch.manual_seed(0)
    layer = RMSCounterLinear(16, 8, lr=1e-6, decimate_updates=True)  # tiny lr -> tiny flip rate
    layer.train()
    x = torch.randn(4, 16)
    for _ in range(6):
        (layer(x) ** 2).sum().backward()
    assert layer._dec_period in (1, 2, 4, 8)


def test_compile_update_falls_back_bit_identical():
    """With compile_update=True on a box without a compiler backend, the permanent eager
    fallback must produce bit-identical states to a plain layer under the same seed."""
    def run(compile_update):
        torch.manual_seed(7)
        layer = RMSCounterLinear(32, 16, lr=0.05, local_grad_clip=1.0,
                                 compile_update=compile_update)
        layer.train()
        x = torch.randn(8, 32)
        for _ in range(3):
            (layer(x) ** 2).sum().backward()
        return layer

    ref, opt = run(False), run(True)
    if _COMPILE_STATE["broken"]:               # this box: no backend -> exact eager fallback
        assert torch.equal(ref.state, opt.state)
        assert torch.equal(ref.scale, opt.scale)
        assert torch.equal(ref.v, opt.v)
    else:                                      # compiled path live: allow last-bit fma drift
        assert torch.isfinite(opt.scale).all() and torch.isfinite(opt.v).all()
        t_ref, _ = ref._decode_rows(0, ref.out_features)
        t_opt, _ = opt._decode_rows(0, opt.out_features)
        assert (t_ref != t_opt).float().mean() < 0.05
