"""Export a recovery checkpoint into the MotifCL bridge pair:

  1. ``counters.mncc``  — binary container with every counter layer's RAW group state
     (t, c in PERMUTED column order + perm + group scales + salient channel). The
     format is deliberately codec-neutral: no 6-bit packing, no automaton byte
     encoding — the consumer (MotifCL's loader) re-encodes into its own state bytes.
  2. ``tail.safetensors`` — every NON-counter tensor of the student state_dict
     (trained norms/biases/embeddings/lm_head). MotifCL already loads safetensors,
     so the fp tail rides the existing path.

Container layout (little-endian):
  magic  b"MNCC0001"
  u32    n_layers
  per layer:
    u32 path_len, utf8 path       (the ORIGINAL linear path, e.g. model.layers.0.self_attn.q_proj)
    u32 out, in, group, C
    i32 perm[in]                  (permuted position j holds original column perm[j])
    i8  t[out*in]                 (PERMUTED order, row-major)
    i8  c[out*in]                 (PERMUTED order, row-major; alpha=0 inference ignores it)
    f32 scale[out*n_groups]       (n_groups = in/group, groups live on the PERMUTED axis)
    u32 n_salient
    i64 sal_idx[n_salient]        (flat indices into the ORIGINAL-order [out, in] weight)
    f32 sal_val[n_salient]

Dense reconstruction (what MotifCL's decode must produce, matching
PackedGroupScaleCounterLinear.visible_weight() at alpha=0):
    W[o, perm[j]] = scale[o, j // group] * t[o, j]        for j in permuted order
    W.flat[sal_idx[k]] = sal_val[k]                       (overrides, exact)

Usage:
  python scripts/export_motifcl.py CKPT_PT OUT_DIR
"""
from __future__ import annotations

import json
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from memory_native.counter import decode_state
from memory_native.packed import unpack_codes

MAGIC = b"MNCC0001"


def _counter_layers(state_dict: dict) -> dict[str, dict]:
    """Group counter buffers by their owning linear path (wrapped or direct)."""
    layers: dict[str, dict] = {}
    for key, tensor in state_dict.items():
        for buf in ("state", "scale", "perm", "salient_idx", "salient_val"):
            suffix = ".counter." + buf
            if key.endswith(suffix):
                path = key[: -len(suffix)]
                layers.setdefault(path, {})[buf] = tensor
                break
            if key.endswith("." + buf):
                path = key[: -len("." + buf)]
                if path + ".counter.state" not in state_dict and (
                    path + ".state" in state_dict and buf != "state" or buf == "state"
                ):
                    layers.setdefault(path, {})[buf] = tensor
                break
    return {p: b for p, b in layers.items() if "state" in b and "scale" in b}


def export(ckpt_path: str, out_dir: str, C: int = 11) -> dict:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload)
    fmt = payload.get("format", {})
    C = int(fmt.get("C", C))
    os.makedirs(out_dir, exist_ok=True)

    layers = _counter_layers(state_dict)
    if not layers:
        raise SystemExit("no counter layers found in checkpoint")

    manifest = {"model": fmt.get("model", "?"), "C": C, "format": fmt, "layers": {}}
    counter_keys = set()
    with open(os.path.join(out_dir, "counters.mncc"), "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(layers)))
        for path in sorted(layers):
            bufs = layers[path]
            for buf_name in bufs:
                for k in state_dict:
                    if k.endswith(path + ".counter." + buf_name) or k == path + "." + buf_name:
                        counter_keys.add(k)
            state = bufs["state"]
            scale = bufs["scale"].to(torch.float32)
            out_f = scale.shape[0]
            perm = bufs.get("perm")
            if state.dim() == 2 and state.dtype == torch.uint8 and state.shape[0] == out_f:
                in_f = perm.numel() if perm is not None else state.shape[1]
                codes = unpack_codes(state, in_f) if state.shape[1] != in_f else state
            else:
                codes = state
                in_f = codes.shape[1]
            t, c = decode_state(codes.to(torch.int64), C)
            group = in_f // scale.shape[1]
            if perm is None:
                perm = torch.arange(in_f)
            sal_idx = bufs.get("salient_idx", torch.zeros(0, dtype=torch.int32))
            sal_val = bufs.get("salient_val", torch.zeros(0))
            pb = path.encode()
            f.write(struct.pack("<I", len(pb)))
            f.write(pb)
            f.write(struct.pack("<IIII", out_f, in_f, group, C))
            f.write(perm.to(torch.int32).numpy().tobytes())
            f.write(t.to(torch.int8).numpy().tobytes())
            f.write(c.to(torch.int8).numpy().tobytes())
            f.write(scale.numpy().tobytes())
            f.write(struct.pack("<I", sal_idx.numel()))
            f.write(sal_idx.to(torch.int64).numpy().tobytes())
            f.write(sal_val.to(torch.float32).numpy().tobytes())
            manifest["layers"][path] = {
                "out": out_f, "in": in_f, "group": group,
                "salient": int(sal_idx.numel()),
            }

    # fp tail: everything that is not a counter buffer.
    tail = {}
    for k, v in state_dict.items():
        if k in counter_keys or ".counter." in k:
            continue
        if any(k.endswith("." + b) for b in
               ("state", "perm", "salient_idx", "salient_val", "_salient_perm_flat",
                "sr_step", "scale", "v")) and any(k.startswith(p) for p in layers):
            continue
        tail[k] = v.contiguous()
    try:
        from safetensors.torch import save_file
        save_file(tail, os.path.join(out_dir, "tail.safetensors"))
    except ImportError:
        torch.save(tail, os.path.join(out_dir, "tail.pt"))
        manifest["tail_format"] = "torch_pt (safetensors not installed)"

    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return manifest


def reference_decode(container_path: str) -> dict[str, torch.Tensor]:
    """Pure-python reader used by tests and as the SPEC for the MotifCL loader."""
    out: dict[str, torch.Tensor] = {}
    with open(container_path, "rb") as f:
        assert f.read(8) == MAGIC
        (n_layers,) = struct.unpack("<I", f.read(4))
        for _ in range(n_layers):
            (plen,) = struct.unpack("<I", f.read(4))
            path = f.read(plen).decode()
            out_f, in_f, group, C = struct.unpack("<IIII", f.read(16))
            perm = torch.frombuffer(bytearray(f.read(4 * in_f)), dtype=torch.int32).long()
            t = torch.frombuffer(bytearray(f.read(out_f * in_f)), dtype=torch.int8)
            t = t.view(out_f, in_f).to(torch.float32)
            _c = f.read(out_f * in_f)
            n_groups = in_f // group
            scale = torch.frombuffer(bytearray(f.read(4 * out_f * n_groups)),
                                     dtype=torch.float32).view(out_f, n_groups)
            (n_sal,) = struct.unpack("<I", f.read(4))
            sal_idx = torch.frombuffer(bytearray(f.read(8 * n_sal)), dtype=torch.int64)
            sal_val = torch.frombuffer(bytearray(f.read(4 * n_sal)), dtype=torch.float32)
            gidx = torch.arange(in_f) // group
            rec_perm = scale[:, gidx] * t
            W = torch.empty(out_f, in_f)
            W[:, perm] = rec_perm
            if n_sal:
                W.reshape(-1)[sal_idx] = sal_val
            out[path] = W
    return out


if __name__ == "__main__":
    ckpt, out_dir = sys.argv[1], sys.argv[2]
    m = export(ckpt, out_dir)
    print(f"exported {len(m['layers'])} counter layers -> {out_dir}")
