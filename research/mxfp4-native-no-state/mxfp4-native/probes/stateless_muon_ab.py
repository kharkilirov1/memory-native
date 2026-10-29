"""A/B probe for MXFP4 rounding and a stateless Muon-style direction."""
from __future__ import annotations
import argparse
import math
import torch
from torch import nn
from mx6 import MX6Linear


def zeropower_ns(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    if grad.ndim != 2:
        raise ValueError("gradient must be a matrix")
    x = grad.detach().float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    return x.T if transposed else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    x = torch.randn(128, 32)
    teacher = torch.randn(32, 32) * 0.1
    y = x @ teacher.T
    layer = MX6Linear.from_dense(torch.zeros(32, 32), lr=args.lr, rounding="sr")
    for step in range(args.steps):
        loss = (layer(x) - y).square().mean()
        loss.backward()
    print(f"seed={args.seed} steps={args.steps} lr={args.lr} loss={float(loss):.6f}")


if __name__ == "__main__":
    main()
