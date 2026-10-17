#!/usr/bin/env python
"""State accounting for the deployed packed counter layer: 6 bits is not 0.75 B/param.

The recurring question is "why is the counter state bigger than 6 bits per
weight?". The answer is that 0.750 B/param is only the packed code body. Measured
components, in the order they are added:

  ideal ............ 0.750 B/param   packed ternary+accumulator code (4 codes / 3 bytes)
  persistent ...... +group scales, per-row RMS, act-order perm, salient channel
  resident ........ +the int64 ``_salient_perm_flat`` mirror and flip telemetry,
                     which live in RAM but are not written to the checkpoint
  step peak ....... decode-in-GEMM fp32 weight + fp32 grad_w + the elementwise
                     update chain (the pure-PyTorch path; the Triton kernels are
                     what actually avoid the dense grad)

Run:
    python scripts/state_accounting.py                        # 1536x1536, group 128
    python scripts/state_accounting.py --in 512 --out 512 --salient 0
    python scripts/state_accounting.py --step                 # + one fwd/bwd peak

The step-peak section needs a process working-set probe (Windows / Linux / macOS);
it prints "unavailable" rather than guessing when the platform has none.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_native import (  # noqa: E402
    PackedGroupScaleCounterLinear,
    PackedRMSCounterLinear,
    RMSCounterLinear,
)


def peak_working_set() -> int | None:
    """Peak resident bytes of this process, or None when the platform has no probe."""
    if sys.platform.startswith("win"):
        import ctypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        info = _PMC()
        info.cb = ctypes.sizeof(_PMC)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(info), info.cb):
            return None
        return int(info.PeakWorkingSetSize)
    try:
        import resource  # Unix only
    except ImportError:  # pragma: no cover - non-Unix, non-Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _synthetic_solved_state(layer, in_features: int, out_features: int,
                            salient_fraction: float, seed: int = 0) -> None:
    """Fill the layer as the PTQ solver would: ternary body, group scales, salient."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(-1, 2, (out_features, in_features), generator=g, dtype=torch.int16)
    perm = torch.randperm(in_features, generator=g).to(torch.int32)
    scales = torch.rand(*layer.scale.shape, generator=g) * 0.1 + 0.01
    n_sal = int(salient_fraction * out_features * in_features)
    salient_idx = torch.randperm(out_features * in_features, generator=g)[:n_sal].to(torch.int32)
    salient_val = (torch.rand(n_sal, generator=g) * 0.1).to(torch.float16)
    layer.load_group_state(scales, t, None, perm, salient_idx, salient_val)


def accounting(in_features: int = 1536, out_features: int = 1536, group: int = 128,
               C: int = 11, salient_fraction: float = 0.02) -> dict:
    """Per-component byte table for one deployed packed group-scale counter layer."""
    layer = PackedGroupScaleCounterLinear(
        in_features, out_features, group=group, C=C, kernel_mode="gemm")
    _synthetic_solved_state(layer, in_features, out_features, salient_fraction)

    non_persistent = layer._non_persistent_buffers_set
    rows: list[tuple[str, str, str, int, bool | None]] = []
    for name, buf in layer.named_buffers():
        rows.append((name, str(tuple(buf.shape)), str(buf.dtype).replace("torch.", ""),
                     buf.numel() * buf.element_size(), name not in non_persistent))
    for name, value in sorted(vars(layer).items()):
        if isinstance(value, torch.Tensor):
            rows.append((f"attr:{name}", str(tuple(value.shape)),
                         str(value.dtype).replace("torch.", ""),
                         value.numel() * value.element_size(), None))

    n_params = in_features * out_features
    return {
        "in_features": in_features,
        "out_features": out_features,
        "params": n_params,
        "salient": layer.salient_idx.numel(),
        "rows": rows,
        "ideal_bytes": int(n_params * 0.75),
        "state_bytes": int(layer.state.numel() * layer.state.element_size()),
        "persistent_bytes": int(layer.persistent_bytes()),
        "resident_bytes": int(sum(row[3] for row in rows)),
        "strict_scratch_bytes": int(layer.strict_scratch_bytes()),
    }


def reference_vs_packed(size: int = 1024) -> dict[str, int]:
    """Unpacked reference layer stores one code per byte; packed stores 4 per 3 bytes."""
    out = {}
    for cls in (RMSCounterLinear, PackedRMSCounterLinear):
        layer = cls(size, size, C=11)
        out[cls.__name__] = sum(
            b.numel() * b.element_size() for b in layer.buffers())
    return out


def step_peak(in_features: int = 1536, out_features: int = 1536, group: int = 128,
              layers: int = 4, batch: int = 64) -> dict:
    """Peak working-set growth of one forward+backward through `layers` packed layers."""
    torch.manual_seed(0)
    mods = [PackedGroupScaleCounterLinear(
        in_features, out_features, group=group, C=11, kernel_mode="gemm", lr=1e-3)
        for _ in range(layers)]
    params = in_features * out_features * layers
    x = torch.randn(batch, in_features)
    before = peak_working_set()
    for mod in mods:
        x = mod(x)
    x.pow(2).mean().backward()
    after = peak_working_set()
    return {"params": params,
            "persistent": int(sum(m.persistent_bytes() for m in mods)),
            "peak_before": before, "peak_after": after,
            "delta": None if before is None or after is None else max(0, after - before)}


def _fmt(n: int) -> str:
    return f"{n / 1024:.1f} KiB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.2f} MiB"


def _print_step_peak(args) -> int:
    peak = step_peak(args.in_features, args.out_features, args.group, args.layers, args.batch)
    if peak["delta"] is None:
        print("step peak: unavailable on this platform")
        return 0
    per = peak["delta"] / peak["params"]
    print(f"one fwd+bwd through {args.layers} layers ({peak['params']:,} params): "
          f"persistent {_fmt(peak['persistent'])}, peak working-set growth "
          f"{_fmt(peak['delta'])} = {per:.2f} B/param "
          f"(x{peak['delta'] / max(peak['persistent'], 1):.1f} the state)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_features", type=int, default=1536)
    ap.add_argument("--out", dest="out_features", type=int, default=1536)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--C", type=int, default=11)
    ap.add_argument("--salient", type=float, default=0.02)
    ap.add_argument("--step", action="store_true", help="also measure one fwd+bwd peak")
    ap.add_argument("--step-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--layers", type=int, default=4, help="layers for the --step measurement")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    if args.step_child:
        # Measured in a FRESH interpreter: the working-set peak is a process
        # high-water mark, so prior allocations would hide part of the delta.
        return _print_step_peak(args)

    acc = accounting(args.in_features, args.out_features, args.group, args.C, args.salient)
    n = acc["params"]
    print(f"packed group-scale counter layer {acc['out_features']}x{acc['in_features']} "
          f"group={args.group} C={args.C} params={n:,} salient={acc['salient']:,} "
          f"({acc['salient'] / n:.1%})")
    print(f"{'component':28s} {'shape':18s} {'dtype':8s} {'bytes':>12s}  note")
    for name, shape, dtype, size, persistent in acc["rows"]:
        note = "checkpoint" if persistent else ("runtime only" if persistent is False else "plain attr")
        print(f"  {name:26s} {shape:18s} {dtype:8s} {_fmt(size):>12s}  {note}")

    print()
    print(f"  6-bit ideal ................ {_fmt(acc['ideal_bytes']):>12s}  "
          f"{acc['ideal_bytes'] / n:.3f} B/param")
    print(f"  packed state body .......... {_fmt(acc['state_bytes']):>12s}  "
          f"{acc['state_bytes'] / n:.3f} B/param")
    print(f"  persistent_bytes() ......... {_fmt(acc['persistent_bytes']):>12s}  "
          f"{acc['persistent_bytes'] / n:.3f} B/param  "
          f"x{acc['persistent_bytes'] / max(acc['ideal_bytes'], 1):.2f} vs ideal")
    print(f"  resident (checkpoint+runtime) {_fmt(acc['resident_bytes']):>11s}  "
          f"{acc['resident_bytes'] / n:.3f} B/param  "
          f"x{acc['resident_bytes'] / max(acc['ideal_bytes'], 1):.2f} vs ideal")
    print(f"  strict-update scratch ...... {_fmt(acc['strict_scratch_bytes']):>12s}  "
          f"{acc['strict_scratch_bytes'] / n:.3f} B/param (transient, 3-launch kernel)")

    print()
    totals = reference_vs_packed(1024)
    ref, packed = totals["RMSCounterLinear"], totals["PackedRMSCounterLinear"]
    print(f"layer kind comparison (1024x1024, no salient): reference {ref / 1024:.1f} KiB "
          f"({ref / (1024 * 1024):.3f} B/param), packed {packed / 1024:.1f} KiB "
          f"({packed / (1024 * 1024):.3f} B/param)")

    if args.step:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--step-child",
               "--in", str(args.in_features), "--out", str(args.out_features),
               "--group", str(args.group), "--layers", str(args.layers),
               "--batch", str(args.batch)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print()
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout.strip().splitlines()[-1])
        else:
            tail = result.stderr.strip().splitlines()
            print(f"step peak: measurement failed ({tail[-1] if tail else 'no output'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
