#!/usr/bin/env python
"""A/B wrapper: probes/e1_quality.py + the latent-grid arms (mx6/latent_grid.py). e1_quality.py is NOT edited.

New arms (everything else is delegated to e1_quality unchanged):
    lg<k>_fp4[_h<h>][_rung]     LatentGridLinear: ONE k-bit index into the fine FP4-aligned grid (14*m+1 points,
                                m = (2^k-1)//14: lg8 -> m=18, 253 pts; lg7 -> 9, 127; lg6 -> 4, 57), deployed weight =
                                RTN to the E2M1 values (ties away from zero) x NVFP4 block scale, no hysteresis counter.
                                _h<h>: headroom (NVFP4 target amax/s = 6/h, default 1; same meaning as
                                --headroom-mode ladder --headroom h). _rung: init the latent ON the E2M1 rung
                                (the same deployed init as na_* for a seed) instead of the nearest fine point.
Bits/param = k + 0.5 (E4M3/16) + 32/in (RMS row) + 32/numel (tensor scale): lg8 8.69 at d=128, like na_fp4_c8.

Same CLI as e1_quality (all args are passed through, incl. the scaling-study flags --dataset enwik8, --data-encoding,
--layers, --heads and the parameter-count / memory report). lg arms need --scale-format nvfp4 (they are NVFP4-only; it keeps
the A/B against na_fp4_* like-for-like) and do not support --scale-lr / --scale-growth / --rescale / --scale-mode
(the block scale is frozen); --counter-wd is the latent's restoring force, --lr follows --sched as for na_*.
Telemetry: flip% = deployed rung changes since the previous eval, edge = weights on +-6, clip = requested moves
beyond the grid ends; --hist prints the deployed E2M1 histogram. Also adds LADDERS["fp4"] if e1_quality lacks it.

    python probes/lg_ab.py --d 128 --steps 6000 --sched cosine --lr 0.01 --scale-format nvfp4 --eval-every 500 \\
        --hist --seeds 0,1 --arms lg8_fp4,lg7_fp4,lg6_fp4,na_fp4_c8,qat_fp4
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E1_PATH = ROOT / "probes" / "e1_quality.py"
sys.path.insert(0, str(ROOT))

import torch                                                       # noqa: E402

from mx6.latent_grid import E2M1_SIGNED, N_RUNGS, LatentGridLinear, get_grid   # noqa: E402

FP4_LADDER = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ARM_RE = re.compile(r"^lg(\d+)_fp4(?:_h(\d+(?:\.\d+)?))?(_rung)?$")


def parse_lg(arm: str):
    """'lg8_fp4_h1.25_rung' -> (k, headroom or None, rung_init); None if `arm` is not a latent-grid arm."""
    if not arm.startswith("lg"):
        return None
    mt = ARM_RE.match(arm)
    if mt is None:
        raise ValueError(f"bad latent-grid arm {arm!r}: expected lg<k>_fp4[_h<headroom>][_rung], e.g. lg8_fp4_h1.25")
    return int(mt.group(1)), (float(mt.group(2)) if mt.group(2) else None), bool(mt.group(3))


def load_e1(register: bool = True):
    """Import probes/e1_quality.py by path as a fresh module (register=False: leave sys.modules alone)."""
    spec = importlib.util.spec_from_file_location("e1_quality", E1_PATH)
    mod = importlib.util.module_from_spec(spec)
    prev = sys.modules.get("e1_quality")
    sys.modules["e1_quality"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if not register:
            if prev is None:
                sys.modules.pop("e1_quality", None)
            else:
                sys.modules["e1_quality"] = prev
    return mod


def _check_flags(e1, arm: str) -> None:
    if e1.SCALE_FORMAT != "nvfp4":
        raise ValueError(f"{arm}: latent-grid arms use NVFP4 scaling; pass --scale-format nvfp4 (otherwise the na_* "
                         f"arms of the same run use E8M0 and the A/B is not like-for-like)")
    bad = [name for name, off in (("--scale-lr", e1.SCALE_LR == 0.0), ("--scale-growth", e1.SCALE_GROWTH == "off"),
                                  ("--rescale", e1.RESCALE_EVERY == 0), ("--scale-mode", e1.SCALE_MODE == "headroom"),
                                  ("--clip", e1.CLIP_FRAC == 0.0)) if not off]
    if bad:
        raise ValueError(f"{arm}: {', '.join(bad)} not supported by latent-grid arms (the block scale is frozen)")


def install(e1):
    """Register the lg arms in the e1_quality module `e1` (idempotent). Returns `e1`."""
    if getattr(e1, "_lg_installed", False):
        return e1
    e1.LADDERS.setdefault("fp4", list(FP4_LADDER))
    orig_make, orig_snap, orig_hist = e1.make_linear, e1.snapshot_values, e1.magnitude_histogram

    def make_linear(arm, i, o, lr):
        spec = parse_lg(arm)
        if spec is None:
            return orig_make(arm, i, o, lr)
        k, h, rung = spec
        _check_flags(e1, arm)
        if h is None:        # same default as na_*: --headroom-mode ladder --headroom h, else h = 1 (amax -> outer rung)
            h = e1.HEADROOM if (e1.HEADROOM_MODE == "ladder" and e1.HEADROOM is not None) else 1.0
        return LatentGridLinear(i, o, k=k, lr=lr, headroom=h, weight_decay=e1.COUNTER_WD,
                                init="rung" if rung else "fine")

    def snapshot_values(model):
        snap = [(m.rung_index().to(torch.int8), N_RUNGS) for m in model.modules() if isinstance(m, LatentGridLinear)]
        return snap if snap else orig_snap(model)      # flip = deployed rung changed; edge = on rung 0 / 14 (+-6)

    def magnitude_histogram(model):
        counts = None
        for m in model.modules():
            if isinstance(m, LatentGridLinear):
                c = torch.bincount(m.rung_index().flatten(), minlength=N_RUNGS).float()
                counts = c if counts is None else counts + c
        if counts is None:
            return orig_hist(model)
        return list(E2M1_SIGNED), (counts / counts.sum()).tolist()

    e1.make_linear, e1.snapshot_values, e1.magnitude_histogram = make_linear, snapshot_values, magnitude_histogram
    e1._lg_installed = True
    return e1


def _announce(argv) -> None:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--arms", default="")
    known, _ = ap.parse_known_args(argv)
    for arm in (a.strip() for a in known.arms.split(",") if a.strip()):
        spec = parse_lg(arm)
        if spec is not None:
            k, h, rung = spec
            g = get_grid(k)
            print(f"lg_ab: {arm} -> LatentGridLinear k={k} m={g.m} grid_points={g.n_pts}/{2 ** k} "
                  f"headroom={'ladder-default' if h is None else h} init={'rung' if rung else 'fine'} "
                  f"bits=k+0.5+32/in+32/numel", flush=True)


def main(argv=None) -> int:
    e1 = install(load_e1())
    _announce(argv)
    return e1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
