"""Build the 5 article charts from real T4 results data.

Outputs PNGs to docs/images/. All numbers come from results/*.md (real Tesla T4 runs).
Run with the project venv:  .venv/Scripts/python scripts/make_article_charts.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

C_METHOD = "#2563eb"
C_DENSE  = "#94a3b8"
C_8BIT   = "#f59e0b"
C_GALORE = "#a855f7"
C_LOMO   = "#ef4444"
C_GREEN  = "#16a34a"


# 1. Memory shootout
fig, ax = plt.subplots(figsize=(8.5, 5))
optims = ["counter+int4\n(the method)", "dense + AdamW", "dense + 8-bit Adam", "dense + GaLore", "dense + LoMo"]
peaks  = [0.99, 1.19, 1.26, 1.35, 1.35]
colors = [C_METHOD, C_DENSE, C_8BIT, C_GALORE, C_LOMO]
bars = ax.bar(optims, peaks, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
bars[0].set_hatch("///")
ax.set_ylabel("Training peak memory (GiB)", fontsize=12)
ax.set_title("Training memory: counter vs memory-efficient optimizers\n(d=512, real tinyshakespeare, single Tesla T4)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 1.6)
for b, p in zip(bars, peaks):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.03, f"{p:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.annotate("lowest peak\nof every contestant", xy=(0, 0.99), xytext=(1.0, 1.45),
            fontsize=9, ha="center", color=C_METHOD, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_METHOD, lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "1_memory_shootout.png"), dpi=150)
plt.close()
print("OK 1_memory_shootout.png")

# 2. Parity at d=512
fig, ax = plt.subplots(figsize=(8.5, 5))
kinds = ["dense", "ternary-QAT", "counter_rms", "counter_packed"]
vals  = [2.6393, 2.6256, 2.5935, 2.6020]
gaps  = [0.0, -0.5, -1.7, -1.4]
colors2 = [C_DENSE, C_8BIT, C_METHOD, "#60a5fa"]
bars = ax.bar(kinds, vals, color=colors2, edgecolor="black", linewidth=0.6, width=0.6)
bars[2].set_hatch("///")
ax.set_ylabel("Validation loss (lower is better)", fontsize=12)
ax.set_title("Quality parity at d=512: counter slightly edges out AdamW\n(s512 / 8 layers, real tinyshakespeare, 300 steps)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(2.55, 2.66)
for b, v, g in zip(bars, vals, gaps):
    lbl = f"{v:.4f}\n({g:+.1f}%)" if g != 0 else f"{v:.4f}\n(baseline)"
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, lbl, ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "2_parity_d512.png"), dpi=150)
plt.close()
print("OK 2_parity_d512.png")

# 3. Scale wall — 1.21B params
fig, ax = plt.subplots(figsize=(8.5, 5))
cfgs = ["dense + AdamW\n(does not fit)", "full method\n(counter + reversible)"]
peaks = [18.0, 2.25]
colors3 = [C_LOMO, C_METHOD]
bars = ax.bar(cfgs, peaks, color=colors3, edgecolor="black", linewidth=0.6, width=0.5)
bars[1].set_hatch("///")
ax.axhline(14.6, color="#000", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(1.4, 14.8, "T4 = 14.6 GiB limit", fontsize=9, color="black", fontweight="bold")
ax.text(0, 18.2, "OOM\n(~18 GiB\nstate needed)", ha="center", fontsize=9, color=C_LOMO, fontweight="bold")
ax.text(1, 2.5, "2.25 GiB\npeak", ha="center", fontsize=10, color=C_METHOD, fontweight="bold")
ax.set_ylabel("Training peak memory (GiB)", fontsize=12)
ax.set_title("Scale: a 1.21B-parameter model trains on a single T4\n(dense+Adam cannot allocate its 18 GiB of state)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 22)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "3_scale_1B.png"), dpi=150)
plt.close()
print("OK 3_scale_1B.png")

# 4. Kernel speedup
fig, ax = plt.subplots(figsize=(8.5, 5))
paths = ["torch update", "Triton fused\nupdate kernel"]
ms    = [9.452, 0.206]
colors4 = [C_DENSE, C_GREEN]
bars = ax.bar(paths, ms, color=colors4, edgecolor="black", linewidth=0.6, width=0.5)
bars[1].set_hatch("///")
ax.set_ylabel("Time per update (ms, lower is better)", fontsize=12)
ax.set_title("Counter update kernel: Triton fused path is 45.9x faster\n(same math, same device - per-launch Python overhead was the wall)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 11)
for b, m in zip(bars, ms):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.2, f"{m:.3f} ms", ha="center", fontsize=10, fontweight="bold")
ax.annotate("x45.9", xy=(1, 0.206), xytext=(0.5, 6),
            fontsize=22, ha="center", color=C_GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "4_kernel_speedup.png"), dpi=150)
plt.close()
print("OK 4_kernel_speedup.png")

# 5. Memory pools breakdown
fig, ax = plt.subplots(figsize=(9, 5.2))
pools = ["Parameters", "Gradients", "Optimizer\nstate", "Activations"]
dense  = [0.20, 0.20, 0.40, 0.84]
method = [0.012, 0.0, 0.0, 0.26]
x = np.arange(len(pools))
w = 0.36
b1 = ax.bar(x - w / 2, dense, w, label="dense + AdamW", color=C_DENSE, edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + w / 2, method, w, label="counter + reversible (method)", color=C_METHOD, edgecolor="black", linewidth=0.6)
b2[0].set_hatch("///"); b2[3].set_hatch("///")
ax.set_xticks(x); ax.set_xticklabels(pools)
ax.set_ylabel("Memory pool size (GiB)", fontsize=12)
ax.set_title("The method attacks all four memory pools at once\n(dense has 3 huge pools the method eliminates; counter params ~ 0.75 byte/weight)",
             fontsize=11.5, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=10)
ax.set_ylim(0, 1.0)
for b, v in zip(b1, dense):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
for b, v in zip(b2, method):
    lbl = f"{v:.3f}" if v < 0.05 else f"{v:.2f}"
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, lbl, ha="center", fontsize=8.5, color=C_METHOD, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "5_memory_pools.png"), dpi=150)
plt.close()
print("OK 5_memory_pools.png")

print("\n=== ALL 5 CHARTS GENERATED ===")
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f))
    print(f"  {sz:>8}  {f}")
