"""Offline counter-learning demo. Synthetic teacher recovery, not an LLM benchmark."""
from __future__ import annotations

import argparse
import json

import torch

from memory_native import PackedRMSCounterLinear, memory_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(0)
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    x = torch.randn(128, 16, device=device)
    teacher = 0.25 * torch.randint(-1, 2, (16, 16), device=device).float()
    target = x @ teacher.T
    layer = PackedRMSCounterLinear(16, 16, C=11, lr=5e-3, lr_scale=0.0).to(device).train()
    with torch.no_grad():
        initial = (layer(x) - target).square().mean().item()
    for _ in range(args.steps):
        (layer(x) - target).square().mean().backward()
    with torch.no_grad():
        final = (layer(x) - target).square().mean().item()
    report = memory_report(layer)
    print(json.dumps({
        "task": "synthetic ternary teacher recovery",
        "device": str(device), "seed": 0, "steps": args.steps,
        "initial_mse": initial, "final_mse": final,
        "packed_code_bytes": layer.state.numel() * layer.state.element_size(),
        "layer_persistent_bytes": int(report["persistent_bytes"]),
        "weight_flips": int(layer.weight_flips),
        "note": "Persistent bytes include layer buffers; activations and scratch are additional.",
    }, indent=2))


if __name__ == "__main__":
    main()
