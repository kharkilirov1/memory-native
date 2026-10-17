"""Render the LUT-GEMM scaling figures as self-contained SVG (no matplotlib here)."""
from __future__ import annotations

import csv
import math
import pathlib

OUT = pathlib.Path(__file__).resolve().parent
PATHS = [("dense_bf16", "#1f77b4"), ("dense_fp32", "#0b6e4f"),
         ("fused_decode", "#c62828"), ("lut_full", "#e07b00"), ("lut_gather", "#6a3fd3")]
W, H = 1180, 760
PAD_L, PAD_R, PAD_T, PAD_B = 86, 26, 44, 58


def load(name: str) -> list[dict]:
    with (OUT / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(x0: float, y0: float, w: float, h: float, xs: list[float], ys_by_path: dict,
          title: str, xlabel: str, ylabel: str, logy: bool = True,
          ymin: float | None = None, ymax: float | None = None) -> list[str]:
    out = []
    allv = [v for seq in ys_by_path.values() for v in seq if v > 0]
    lx = [math.log10(x) for x in xs]
    if logy:
        ly = [math.log10(v) for v in allv]
    else:
        ly = list(allv)
    x_lo, x_hi = min(lx), max(lx)
    y_lo = math.log10(ymin) if (logy and ymin) else (min(ly) if logy else (ymin if ymin else min(ly)))
    y_hi = math.log10(ymax) if (logy and ymax) else (max(ly) if logy else (ymax if ymax else max(ly)))
    if logy and ymin is None:
        y_lo = math.floor(y_lo * 2) / 2
    if logy and ymax is None:
        y_hi = math.ceil(y_hi * 2) / 2
    sx = lambda x: x0 + (math.log10(x) - x_lo) / (x_hi - x_lo) * w
    sy = lambda v: (y0 + h - (math.log10(v) - y_lo) / (y_hi - y_lo) * h) if logy else (
        y0 + h - (v - y_lo) / (y_hi - y_lo) * h)
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
               f'fill="#ffffff" stroke="#cccccc"/>')
    # grid: x = decade ticks on M
    for dec in range(int(math.ceil(x_lo)), int(math.floor(x_hi)) + 1):
        x = sx(10 ** dec)
        out.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0+h:.1f}" '
                   f'stroke="#eeeeee"/>')
        out.append(f'<text x="{x:.1f}" y="{y0+h+18:.1f}" font-size="12" fill="#444" '
                   f'text-anchor="middle">M={10**dec}</text>')
    # y grid
    if logy:
        for dec in range(int(math.floor(y_lo)), int(math.ceil(y_hi)) + 1):
            val = 10 ** dec
            if not (y_lo <= dec <= y_hi):
                continue
            y = sy(val)
            lab = f"{val:g}"
            out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0+w:.1f}" y2="{y:.1f}" '
                       f'stroke="#eeeeee"/>')
            out.append(f'<text x="{x0-8:.1f}" y="{y+4:.1f}" font-size="12" fill="#444" '
                       f'text-anchor="end">{lab}</text>')
    # series
    for name, color in PATHS:
        seq = ys_by_path.get(name)
        if not seq:
            continue
        pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, seq) if v > 0)
        dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="2.6" fill="{color}"/>'
                       for x, v in zip(xs, seq) if v > 0)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        out.append(dots)
    out.append(f'<text x="{x0+w/2:.1f}" y="{y0-12:.1f}" font-size="14" font-weight="600" '
               f'text-anchor="middle">{esc(title)}</text>')
    out.append(f'<text x="{x0+w/2:.1f}" y="{y0+h+38:.1f}" font-size="12" fill="#333" '
               f'text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text x="{x0-52:.1f}" y="{y0+h/2:.1f}" font-size="12" fill="#333" '
               f'text-anchor="middle" transform="rotate(-90 {x0-52:.1f} {y0+h/2:.1f})">'
               f'{esc(ylabel)}</text>')
    return out


def legend(x: float, y: float) -> list[str]:
    out = [f'<text x="{x:.1f}" y="{y:.1f}" font-size="13" font-weight="600" '
           f'fill="#222">legend</text>']
    for i, (name, color) in enumerate(PATHS):
        yy = y + 20 + 17 * i
        out.append(f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x+26:.1f}" y2="{yy:.1f}" '
                   f'stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{x+34:.1f}" y="{yy+4:.1f}" font-size="12" fill="#333">'
                   f'{esc(name)}</text>')
    out.append(f'<text x="{x:.1f}" y="{y+20+17*len(PATHS)+12:.1f}" font-size="11" '
               f'fill="#555">CUDA graph replay, 4096^2 / 1024^2,</text>')
    out.append(f'<text x="{x:.1f}" y="{y+20+17*len(PATHS)+27:.1f}" font-size="11" '
               f'fill="#555">strict ternary, log-log axes</text>')
    return out


def main() -> None:
    rows = load("graph_results.csv")
    by = {}
    for r in rows:
        by.setdefault((int(r["shape"]), r["path"]), []).append((int(r["M"]), float(r["graph_ms"])))
    shapes = sorted({s for s, _ in by})
    ms = sorted({m for k in by for m, _ in by[k]})
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Arial, sans-serif">',
             '<rect width="100%" height="100%" fill="#fafafa"/>',
             f'<text x="{W/2}" y="26" font-size="17" font-weight="700" text-anchor="middle">'
             'LUT-GEMM (Method of Four Russians) vs dense GEMM / fused decode — A100 40GB</text>']
    panels = [(PAD_L, PAD_T, 470, 250, "4096² latency (graph replay)"),
              (PAD_L, PAD_T + 330, 470, 250, "1024² latency (graph replay)")]
    for x0, y0, w, h, title in panels:
        shape = 4096 if "4096" in title else 1024
        series = {p: [dict(by[(shape, p)]).get(m, 0.0) for m in ms] for p, _ in PATHS}
        parts += panel(x0, y0, w, h, ms, series, title, "tokens M", "ms (log)")
    # ratio panel
    x0, y0, w, h = 620, PAD_T, 500, 250
    base = "dense_bf16"
    series = {}
    for p in ("dense_fp32", "fused_decode", "lut_full", "lut_gather"):
        series[p] = [dict(by[(4096, p)])[m] / dict(by[(4096, base)])[m] for m in ms]
    parts += panel(x0, y0, w, h, ms, series, "4096²: time relative to dense bf16 (1.0 = parity)",
                   "tokens M", "ratio (log)")
    parts.append(f'<line x1="{x0}" y1="{y0+h/2}" x2="{x0+w}" y2="{y0+h/2}" '
                 f'stroke="#999" stroke-dasharray="4 3"/>')
    parts += legend(660, PAD_T + 330)
    # second ratio panel: LUT vs the repo's own fused decode
    x0, y0, w, h = 620, PAD_T + 330, 500, 250
    series = {"fused_decode": [dict(by[(4096, "fused_decode")])[m] /
                               dict(by[(4096, "lut_gather")])[m] for m in ms]}
    parts += panel(x0, y0, w, h, ms, series,
                   "4096²: speedup of LUT gather over the repo's fused-decode kernel",
                   "tokens M", "x faster (linear)", logy=False, ymin=0)
    parts.append("</svg>")
    (OUT / "latency_scaling.svg").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT / "latency_scaling.svg")


if __name__ == "__main__":
    main()
