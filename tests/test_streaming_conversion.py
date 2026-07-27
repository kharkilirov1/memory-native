"""Streaming conversion witnesses: agreement with the in-memory path, resume,
sharded checkpoints, and the fail-loudly property carried over from ptq.py."""
import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from memory_native.donor.ptq import ptq_warm_start                    # noqa: E402
from memory_native.donor.streaming import (                           # noqa: E402
    convert_streaming,
    load_streamed_state,
)

SOLVE = dict(group=32, C=11, grid="itf", salient_first=0.02, salient_scope="layer",
             in_sweep_refit=True)


def _tiny_donor(tmp_path, layers=3):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg).eval()
    path = os.path.join(tmp_path, "donor")
    model.save_pretrained(path, safe_serialization=True)
    return path, cfg


def _calib(cfg, n=4, b=2, t=32):
    torch.manual_seed(1)
    return [torch.randint(0, cfg.vocab_size, (b, t)) for _ in range(n)]


def test_streaming_matches_in_memory_classic(tmp_path):
    """With cascade off the driver has the in-memory path's semantics, so every
    counter state must come out bit-identical."""
    path, cfg = _tiny_donor(str(tmp_path))
    calib = _calib(cfg)

    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    ref_report = ptq_warm_start(reference, calib, mode="gptq_group",
                                kind="counter_packed", progress=False, **SOLVE)
    ref_state = {k: v for k, v in reference.state_dict().items() if "counter" in k}

    out = os.path.join(str(tmp_path), "streamed")
    stream_report = convert_streaming(path, calib, out, kind="counter_packed",
                                      cascade=False, micro_batch=2, progress=False,
                                      **SOLVE)
    streamed = load_streamed_state(out)

    assert stream_report.blocks_converted == cfg.num_hidden_layers
    assert stream_report.coeffs == ref_report.coeffs
    shared = [k for k in ref_state if k in streamed]
    assert shared, "no overlapping counter keys between the two paths"
    for key in shared:
        assert torch.equal(ref_state[key].cpu(), streamed[key]), f"mismatch at {key}"


def test_cascade_changes_the_solve(tmp_path):
    """Cascade is not cosmetic: calibrating on converted outputs must move the
    state of later blocks (block 0 sees identical inputs either way)."""
    path, cfg = _tiny_donor(str(tmp_path))
    calib = _calib(cfg)

    classic = os.path.join(str(tmp_path), "classic")
    cascaded = os.path.join(str(tmp_path), "cascade")
    convert_streaming(path, calib, classic, cascade=False, micro_batch=2,
                      progress=False, **SOLVE)
    convert_streaming(path, calib, cascaded, cascade=True, micro_batch=2,
                      progress=False, **SOLVE)
    a, b = load_streamed_state(classic), load_streamed_state(cascaded)

    first = [k for k in a if k.startswith("model.layers.0.") and k.endswith("counter.state")]
    last = [k for k in a
            if k.startswith(f"model.layers.{cfg.num_hidden_layers - 1}.")
            and k.endswith("counter.state")]
    assert all(torch.equal(a[k], b[k]) for k in first), "block 0 must be unaffected"
    assert any(not torch.equal(a[k], b[k]) for k in last), "later blocks must differ"


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    """Dropping finished blocks from the manifest and re-running must land on the
    same state -- the replay of finished blocks keeps the cascade intact."""
    path, cfg = _tiny_donor(str(tmp_path))
    calib = _calib(cfg)
    out = os.path.join(str(tmp_path), "streamed")

    convert_streaming(path, calib, out, micro_batch=2, progress=False, **SOLVE)
    full = load_streamed_state(out)

    manifest = os.path.join(out, "manifest.json")
    with open(manifest, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["blocks_done"] = [0]
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    report = convert_streaming(path, calib, out, micro_batch=2, progress=False, **SOLVE)
    assert report.blocks_resumed == 1
    assert report.blocks_converted == cfg.num_hidden_layers - 1
    resumed = load_streamed_state(out)
    assert set(full) == set(resumed)
    for key in full:
        assert torch.equal(full[key], resumed[key]), f"resume changed {key}"


def test_sharded_checkpoint(tmp_path):
    """A sharded checkpoint must resolve through model.safetensors.index.json."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg).eval()
    path = os.path.join(str(tmp_path), "sharded")
    model.save_pretrained(path, safe_serialization=True, max_shard_size="200KB")
    assert os.path.exists(os.path.join(path, "model.safetensors.index.json")), \
        "fixture did not actually shard"

    out = os.path.join(str(tmp_path), "streamed")
    report = convert_streaming(path, _calib(cfg), out, micro_batch=2, progress=False,
                               **SOLVE)
    assert report.blocks_converted == cfg.num_hidden_layers
    assert load_streamed_state(out)


def test_restore_from_streamed_state(tmp_path):
    """The streamed state must rebuild a working model through the normal
    restore path -- streaming output is not a separate format."""
    from transformers import AutoModelForCausalLM

    from memory_native.recovery.runtime import restore_counter_structure

    path, cfg = _tiny_donor(str(tmp_path), layers=2)
    calib = _calib(cfg)
    out = os.path.join(str(tmp_path), "streamed")
    convert_streaming(path, calib, out, micro_batch=2, progress=False, **SOLVE)
    state = load_streamed_state(out)

    fresh = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    restore_counter_structure(fresh, state, kind="counter_packed", group=32, C=11)
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:4]}"
    with torch.no_grad():
        logits = fresh(calib[0]).logits
    assert torch.isfinite(logits).all()


def _moe_donor(tmp_path, layers=2, experts=4):
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    cfg = MixtralConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=layers, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=128,
                        num_local_experts=experts, num_experts_per_tok=2)
    path = os.path.join(tmp_path, "moe")
    MixtralForCausalLM(cfg).eval().save_pretrained(path, safe_serialization=True)
    return path, cfg


def test_moe_streaming_matches_in_memory(tmp_path):
    """MoE donors stream too. transformers writes experts one module per expert
    and stacks them at load time, so the reader reassembles
    gate_up_proj[e] = cat([w1, w3]) / down_proj[e] = w2; with cascade off the
    result must match the in-memory path exactly."""
    from transformers import AutoModelForCausalLM

    path, cfg = _moe_donor(str(tmp_path))
    calib = _calib(cfg)

    out = os.path.join(str(tmp_path), "streamed")
    report = convert_streaming(path, calib, out, kind="counter_packed", cascade=False,
                               micro_batch=2, progress=False, **SOLVE)
    streamed = load_streamed_state(out)

    experts = [t for t in report.targets if "experts" in t]
    assert len(experts) == cfg.num_hidden_layers * cfg.num_local_experts, \
        f"expected every expert converted, got {len(experts)}"

    reference = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    ref_report = ptq_warm_start(reference, calib, mode="gptq_group",
                                kind="counter_packed", progress=False, **SOLVE)
    assert report.coeffs == ref_report.coeffs
    ref_state = {k: v for k, v in reference.state_dict().items()
                 if "counter" in k or "experts" in k}
    shared = [k for k in ref_state if k in streamed]
    assert len(shared) > cfg.num_hidden_layers * cfg.num_local_experts, \
        "too few overlapping keys to be a meaningful comparison"
    for key in shared:
        assert torch.equal(ref_state[key].cpu(), streamed[key]), f"mismatch at {key}"


def test_unknown_expert_layout_still_refuses(tmp_path):
    """A checkpoint whose expert tensors the reader cannot place must raise, not
    convert attention only."""
    import shutil

    import safetensors.torch as sft

    path, cfg = _moe_donor(str(tmp_path), layers=1)
    broken = os.path.join(str(tmp_path), "broken")
    os.makedirs(broken, exist_ok=True)
    shutil.copy(os.path.join(path, "config.json"), broken)
    tensors = sft.load_file(os.path.join(path, "model.safetensors"))
    renamed = {(k.replace("block_sparse_moe.experts", "block_sparse_moe.weird")
                if "block_sparse_moe.experts" in k else k): v
               for k, v in tensors.items()}
    sft.save_file(renamed, os.path.join(broken, "model.safetensors"),
                  metadata={"format": "pt"})

    with pytest.raises((RuntimeError, NotImplementedError, KeyError)):
        convert_streaming(broken, _calib(cfg), os.path.join(str(tmp_path), "out"),
                          micro_batch=2, progress=False, **SOLVE)


def test_resolve_decoder_finds_a_nested_text_tower():
    """A multimodal donor nests the decoder: gemma-4 keeps it at
    `model.language_model`, not `model`. The driver used to hardcode
    `skeleton.model.layers` and died with AttributeError before reading a single
    block; worse, that same hardcoded string is the CHECKPOINT prefix, so a wrong
    resolution asks the safetensors file for names that do not exist."""
    from torch import nn

    from memory_native.donor.streaming import _resolve_decoder

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    class Wrapper(nn.Module):          # multimodal: model.language_model.layers
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = Decoder()
            self.model.vision_tower = nn.Linear(4, 4)

    inner, stack = _resolve_decoder(Wrapper())
    assert stack == "model.language_model"
    assert len(inner.layers) == 2

    class Plain(nn.Module):            # text-only: model.layers
        def __init__(self):
            super().__init__()
            self.model = Decoder()

    inner, stack = _resolve_decoder(Plain())
    assert stack == "model"


def test_resolve_decoder_refuses_instead_of_guessing():
    """No decoder, or two equally plausible ones, must raise: silently picking one
    would convert the wrong stack and still report success."""
    from torch import nn

    from memory_native.donor.streaming import _resolve_decoder

    class NoDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4)

    with pytest.raises(NotImplementedError, match="no decoder module"):
        _resolve_decoder(NoDecoder())

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

    class TwoTowers(nn.Module):
        def __init__(self):
            super().__init__()
            self.text = Decoder()
            self.audio = Decoder()

    with pytest.raises(NotImplementedError, match="ambiguous decoder"):
        _resolve_decoder(TwoTowers())


def test_real_donor_still_resolves_to_the_model_prefix():
    """Non-regression on the shape every other test uses: a plain CausalLM must
    still resolve to the `model` prefix the committed state keys were written with."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from memory_native.donor.streaming import _resolve_decoder

    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2)
    inner, stack = _resolve_decoder(Qwen2ForCausalLM(cfg))
    assert stack == "model"
    assert len(inner.layers) == 2
