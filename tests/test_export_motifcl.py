"""MotifCL bridge exporter gates. Requires the optional `safetensors` dependency
(pip install safetensors); the module skips cleanly when it is absent.

Claims:
  * export -> reference_decode reproduces every counter layer's visible_weight()
    (the reference decoder IS the spec for the MotifCL C++ loader);
  * the fp tail contains the non-counter tensors (biases) and none of the counter
    buffers;
  * the container round-trips paths, shapes and the salient channel.
"""
import copy
import os

import pytest

pytest.importorskip("safetensors", reason="optional dependency of the MotifCL export path")

torch = pytest.importorskip("torch")
nn = torch.nn

from memory_native.donor.ptq import ptq_warm_start

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "export_motifcl",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "export_motifcl.py"),
)
export_motifcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_motifcl)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 24, bias=True)
        self.head = nn.Linear(24, 8, bias=False)

    def forward(self, x):
        return self.head(torch.relu(self.proj(x)))


def test_export_reference_decode_matches_visible_weight(tmp_path):
    torch.manual_seed(3)
    model = _Tiny()
    calib = [torch.randn(6, 5, 32), torch.randn(4, 5, 32)]
    ptq_warm_start(model, calib, mode="gptq_group", kind="counter_packed",
                   group=8, C=11, progress=False, kernel_mode="torch",
                   grid="itf", scale_refit="align", salient_first=0.03,
                   extra_skip=["head"])
    counter = model.proj.counter
    payload = {"model": model.state_dict(), "format": {"C": 11, "model": "tiny"}}
    ckpt = tmp_path / "ckpt.pt"
    torch.save(payload, ckpt)

    manifest = export_motifcl.export(str(ckpt), str(tmp_path / "out"))
    assert len(manifest["layers"]) == 1
    (path, meta), = manifest["layers"].items()
    assert meta["out"] == 24 and meta["in"] == 32 and meta["group"] == 8
    assert meta["salient"] == counter.salient_idx.numel() and meta["salient"] > 0

    decoded = export_motifcl.reference_decode(str(tmp_path / "out" / "counters.mncc"))
    W = decoded[path]
    with torch.no_grad():
        ref = counter.visible_weight()
    assert torch.allclose(W, ref, atol=1e-6), (W - ref).abs().max()

    # fp tail: bias of the wrapped linear + untouched head weights, no counter buffers
    from safetensors.torch import load_file
    tail_path = tmp_path / "out" / "tail.safetensors"
    if tail_path.exists():
        tail = load_file(str(tail_path))
    else:
        tail = torch.load(tmp_path / "out" / "tail.pt")
    assert any(k.endswith("head.weight") for k in tail)
    assert any("bias" in k for k in tail)
    assert not any(".counter." in k for k in tail)
