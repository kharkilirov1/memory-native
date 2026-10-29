#!/usr/bin/env python
"""Phase 2 harness: tiny char-LM, several weight/update arms, one config object.

Arms:
  dense  fp32 weights + AdamW                       (reference)
  qat    MXFP4 forward, fp32 master, AdamW + STE    (the format control)
  msr    MXFP4 forward, fp32 master, OUR rule       (master control)
  mx6    MXFP4 weights, in-weight integer accumulator, no master

    python probes/charlm_phase2.py --arm mx6 --integrator 9 --steps 800
"""
from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx6 import MX6Linear, pack  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare.txt"


def peak_working_set() -> int:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.K32GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    fn.restype = ctypes.c_int
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    info = PMC()
    info.cb = ctypes.sizeof(PMC)
    fn(k32.GetCurrentProcess(), ctypes.byref(info), info.cb)
    return int(info.PeakWorkingSetSize)


@dataclass
class Cfg:
    arm: str = "mx6"
    steps: int = 800
    lr: float = 0.01
    adam_lr: float = 3e-3
    adam_wd: float = 0.01
    integrator: int = 0
    second_moment: str = "row"
    rms_beta: float = 0.9
    cautious: bool = False
    integrator_decay: float = 0.0
    scale_headroom: float = 6.0
    scale_mode: str = "headroom"
    lr_mode: str = "abs"
    post_quantize: bool = False
    threads: int = 8


class QATLinear(nn.Module):
    """MXFP4 forward, fp32 master, straight-through gradient."""

    def __init__(self, in_features: int, out_features: int, headroom: float = 6.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(
            -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5))
        self.headroom = headroom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        codes, scale = pack.quantize(w.detach(), headroom=self.headroom)
        quantized = pack.dequantize(codes, scale)
        return x @ (w + (quantized - w).detach()).t()


class _MasterRuleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, token):
        ctx.save_for_backward(x, module.weight.data)
        ctx.module = module
        return x @ module.weight.t()

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        m = ctx.module
        grad_w = grad_out.t() @ x
        grad_x = grad_out @ w
        with torch.no_grad():
            g_sq = grad_w.square().mean(dim=1, keepdim=True)
            m.v.mul_(m.beta).add_(g_sq, alpha=1.0 - m.beta)
            g_hat = grad_w / m.v.sqrt().clamp_min(m.eps)
            m.weight.data.add_(g_hat, alpha=-m.lr)
        return grad_x, None, None


class MasterRuleLinear(nn.Module):
    """MXFP4 forward, fp32 master, our row-RMS rule (no Adam). Master control arm."""

    def __init__(self, in_features: int, out_features: int, lr: float = 0.01,
                 headroom: float = 6.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(
            -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5))
        self.lr, self.beta, self.eps, self.headroom = lr, 0.9, 1e-3, headroom
        self.register_buffer("v", torch.full((out_features, 1), 1e-3))
        self._token = torch.zeros(0, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = _MasterRuleFn.apply(flat, self, self._token)
        return out.reshape(*shape[:-1], self.weight.shape[0])


def make_linear(arm: str, in_features: int, out_features: int, cfg: Cfg) -> nn.Module:
    if arm == "dense":
        return nn.Linear(in_features, out_features, bias=False)
    if arm == "qat":
        return QATLinear(in_features, out_features, headroom=cfg.scale_headroom)
    if arm == "msr":
        return MasterRuleLinear(in_features, out_features, lr=cfg.lr,
                                headroom=cfg.scale_headroom)
    if arm == "mx6":
        return MX6Linear(in_features, out_features, lr=cfg.lr,
                         integrator=cfg.integrator,
                         second_moment=cfg.second_moment,
                         rms_beta=cfg.rms_beta,
                         cautious=cfg.cautious,
                         integrator_decay=cfg.integrator_decay,
                         scale_headroom=cfg.scale_headroom,
                         scale_mode=cfg.scale_mode, lr_mode=cfg.lr_mode)
    raise ValueError(arm)


class Block(nn.Module):
    def __init__(self, arm: str, d: int, cfg: Cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = make_linear(arm, d, 3 * d, cfg)
        self.proj = make_linear(arm, d, d, cfg)
        self.ln2 = nn.LayerNorm(d)
        self.fc = make_linear(arm, d, 4 * d, cfg)
        self.out = make_linear(arm, 4 * d, d, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=-1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        x = x + self.proj(att @ v)
        return x + self.out(torch.nn.functional.gelu(self.fc(self.ln2(x))))


class CharLM(nn.Module):
    def __init__(self, vocab: int, cfg: Cfg, d: int = 128, blocks: int = 2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(256, d)
        self.blocks = nn.ModuleList([Block(cfg.arm, d, cfg) for _ in range(blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def batch(data: torch.Tensor, size: int, seq: int, gen: torch.Generator):
    starts = torch.randint(0, data.numel() - seq - 1, (size,), generator=gen)
    x = torch.stack([data[s:s + seq] for s in starts])
    y = torch.stack([data[s + 1:s + seq + 1] for s in starts])
    return x, y


@torch.no_grad()
def evaluate(model: nn.Module, data: torch.Tensor, seq: int = 128, iters: int = 20) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    total = 0.0
    device = next(model.parameters()).device
    for _ in range(iters):
        x, y = batch(data, 32, seq, gen)
        x, y = x.to(device), y.to(device)
        total += float(nn.functional.cross_entropy(
            model(x).reshape(-1, model.head.out_features), y.reshape(-1)))
    model.train()
    return total / iters / math.log(2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mx6", choices=["dense", "qat", "msr", "mx6"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--adam-wd", type=float, default=0.01)
    ap.add_argument("--integrator", type=int, default=0)
    ap.add_argument("--second-moment", default="row", choices=["row", "block", "col", "layer"])
    ap.add_argument("--rms-beta", type=float, default=0.9)
    ap.add_argument("--cautious", action="store_true")
    ap.add_argument("--integrator-decay", type=float, default=0.0)
    ap.add_argument("--scale-headroom", type=float, default=6.0)
    ap.add_argument("--scale-mode", default="headroom", choices=["headroom", "mse", "auto"])
    ap.add_argument("--lr-mode", default="abs", choices=["abs", "scale"])
    ap.add_argument("--post-quantize", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--blocks", type=int, default=2)
    args = ap.parse_args(argv)

    cfg = Cfg(arm=args.arm, steps=args.steps, lr=args.lr, adam_lr=args.adam_lr,
              adam_wd=args.adam_wd, integrator=args.integrator,
              second_moment=args.second_moment, rms_beta=args.rms_beta,
              cautious=args.cautious, integrator_decay=args.integrator_decay,
              scale_headroom=args.scale_headroom, scale_mode=args.scale_mode,
              lr_mode=args.lr_mode,
              post_quantize=args.post_quantize,
              threads=args.threads)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(args.seed)

    raw = torch.tensor(list(DATA.read_text(encoding="utf-8").encode("utf-8")), dtype=torch.long)
    vocab = int(raw.max()) + 1
    split = int(0.9 * raw.numel())
    train_data, val_data = raw[:split], raw[split:]

    device = torch.device(args.device)
    model = CharLM(vocab, cfg, d=args.d, blocks=args.blocks).to(device)
    self_managed = {id(m.weight) for m in model.modules() if isinstance(m, MasterRuleLinear)}
    tail = [p for p in model.parameters() if p.requires_grad and id(p) not in self_managed]
    opt = torch.optim.AdamW(tail, lr=cfg.adam_lr, weight_decay=cfg.adam_wd)
    n_counter = sum(m.codes.numel() for m in model.modules() if isinstance(m, MX6Linear))
    print(f"d={args.d} blocks={args.blocks} arm={cfg.arm} int={cfg.integrator} sm={cfg.second_moment} beta={cfg.rms_beta} "
          f"headroom={cfg.scale_headroom} sm={cfg.scale_mode} counter={n_counter:,} "
          f"tail={sum(p.numel() for p in tail):,}")

    gen = torch.Generator().manual_seed(7 + args.seed)
    start = time.time()
    running = 0.0
    for step in range(1, cfg.steps + 1):
        x, y = batch(train_data, 32, 128, gen)
        x, y = x.to(device), y.to(device)
        loss = nn.functional.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += float(loss.detach())
        if step % 250 == 0:
            print(f"  step {step:5d} train={running / 250:.4f} "
                  f"val={evaluate(model, val_data):.4f} "
                  f"({step / (time.time() - start):.1f} steps/s)")
            running = 0.0

    if cfg.post_quantize:
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Linear) and m.weight.shape[-1] % 32 == 0:
                    codes, scale = pack.quantize(m.weight.data)
                    m.weight.data = pack.dequantize(codes, scale)
        print(f"  post-quantised: val={evaluate(model, val_data):.4f}")

    val = evaluate(model, val_data)
    peak = peak_working_set()
    print(f"RESULT arm={cfg.arm} int={cfg.integrator} sm={cfg.second_moment} "
          f"headroom={cfg.scale_headroom} decay={cfg.integrator_decay} lr={cfg.lr} "
          f"steps={cfg.steps} val_bits_per_char={val:.4f} peak_mib={peak / 2**20:.0f} "
          f"seconds={time.time() - start:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
