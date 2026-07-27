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
