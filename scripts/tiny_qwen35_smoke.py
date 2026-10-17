"""Tiny synthetic qwen3_5 VLM: end-to-end witness of the streaming conversion path.

Builds a 2-layer (linear_attention + full_attention) random-weight
Qwen3_5ForConditionalGeneration, saves it as a local donor (config.json +
model.safetensors, checkpoint keys == module names), then runs the REAL
convert_streaming on it: skeleton build, probe, materialize, H collection,
solve, swap, cascade re-run, block save, resume, restore, forward.

This is the pre-flight for the 24 h 27B run: every piece of machinery the big
run will use is exercised here first, on a donor that costs nothing.
"""
import importlib.util
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
DONOR = os.path.join(BASE, "tiny_qwen35")
OUT = os.path.join(BASE, "tiny_qwen35_out")


def build_donor():
    from transformers import (
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5TextConfig,
    )

    try:
        from transformers import Qwen3_5VisionConfig

        vis = Qwen3_5VisionConfig(
            depth=2, hidden_size=64, intermediate_size=128, num_heads=2,
            image_size=64, patch_size=16, num_channels=3,
        )
    except Exception as exc:  # noqa: BLE001
        print("vision shrink unavailable, using default:", exc)
        vis = None

    tc = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_conv_kernel_dim=4,
        max_position_embeddings=512, mtp_num_hidden_layers=0,
        attention_bias=False, tie_word_embeddings=False, rms_norm_eps=1e-5,
    )
    cfg = Qwen3_5Config(text_config=tc, **({"vision_config": vis} if vis else {}))
    torch.manual_seed(7)
    model = Qwen3_5ForConditionalGeneration(cfg)

    shutil.rmtree(DONOR, ignore_errors=True)
    os.makedirs(DONOR)
    cfg.save_pretrained(DONOR)
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(DONOR, "model.safetensors"))
    n = sum(v.numel() for v in model.state_dict().values())
    print(f"[donor] built tiny VLM: {n/1e6:.1f}M coeffs, saved to {DONOR}")
    return model


def main():
    model = build_donor()
    from memory_native.donor.streaming import convert_streaming, load_streamed_state

    torch.manual_seed(1234)
    batches = [torch.randint(0, 512, (1, 64)) for _ in range(2)]
    kw = dict(C=11, group=32, dtype=torch.float32, device="cpu", micro_batch=1,
              cascade=True, grid="itf", salient_first=0.02, salient_scope="layer",
              in_sweep_refit=True)

    rep = convert_streaming(DONOR, batches, OUT, resume=True, **kw)
    print(f"[run1] converted={rep.blocks_converted}/{rep.blocks_total} "
          f"targets={len(rep.targets)} coeffs={rep.coeffs:,} "
          f"peak={rep.peak_bytes/2**20:.0f}MiB")

    rep2 = convert_streaming(DONOR, batches, OUT, resume=True, **kw)
    print(f"[run2-resume] resumed={rep2.blocks_resumed} converted={rep2.blocks_converted}")
    assert rep2.blocks_converted == 0 and rep2.blocks_resumed == rep.blocks_total

    state = load_streamed_state(OUT)
    n_state = sum(1 for k in state if k.endswith(".state"))
    print(f"[state] {len(state)} tensors, {n_state} .state entries")

    from memory_native.recovery.runtime import restore_counter_structure

    with torch.no_grad():
        model.eval()
        # Counter state covers the TEXT decoder only (vision/MTP stay fp), so
        # restore into the TextModel subtree -- the module the checkpoint keys
        # as model.language_model.* -- exactly like the real donor's witness.
        prefix = "model.language_model."
        stripped = {k[len(prefix):]: v for k, v in state.items()
                    if k.startswith(prefix)}
        missing = restore_counter_structure(model.model.language_model, stripped,
                                            kind="counter_packed", group=32, C=11)
        print(f"[restore] swapped {len(stripped)} tensors, leftover {len(missing) if hasattr(missing, '__len__') else missing}")
        ids = torch.randint(0, 512, (1, 48))
        out = model(input_ids=ids).logits
        assert torch.isfinite(out).all(), "restored forward produced non-finite logits"
        print(f"[restore] forward finite, logits {tuple(out.shape)}")
    print("[restore] full-VLM phase2 witness (restore_witness.py semantics)")
    os.environ.update({
        "MODEL": DONOR, "STATE_DIR": OUT, "KIND": "counter_packed",
        "GROUP": "32", "C": "11", "DTYPE": "fp32",
    })
    spec = importlib.util.spec_from_file_location(
        "restore_witness", os.path.join(BASE, "..", "scripts", "restore_witness.py"))
    rw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rw)
    model2 = rw.phase2_restore()
    with torch.no_grad():
        out2 = model2(input_ids=torch.randint(0, 512, (1, 32))).logits
        assert torch.isfinite(out2).all(), "phase2 model forward non-finite"
    print(f"[phase2] restored full VLM forward finite, logits {tuple(out2.shape)}")
    print("TINY QWEN3_5 SMOKE: ALL GREEN")


if __name__ == "__main__":
    main()
