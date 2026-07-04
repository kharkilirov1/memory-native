"""Build the saturation-vs-scale charts from the Kaggle kernel output.

Charts:
  6_saturation_vs_scale.png — saturation rate per step, 3 curves
  7_saturation_vs_loss.png  — bar chart: saturation plateaus while loss grows

Run with the project venv:
    .venv/Scripts/python scripts/make_saturation_charts.py [PATH_TO_KAGGLE_OUTPUT]
"""
import sys, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Kharki\AppData\Local\Temp\mn_out"
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)

CONFIGS = [
    ("d512",  "saturation_d512_L8.json",  "#2563eb", "d=512 (L=8)"),
    ("d1024", "saturation_d1024_L12.json","#16a34a", "d=1024 (L=12)"),
    ("d2048", "saturation_d2048_L12.json","#ef4444", "d=2048 (L=12)"),
]
configs = {}
for tag, fn, color, label in CONFIGS:
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        print(f"MISSING: {p}")
        sys.exit(1)
    configs[tag] = {"data": json.load(open(p, encoding="utf-8")), "color": color, "label": label}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "axes.axisbelow": True,
})

# 1. Saturation rate per step
fig, ax = plt.subplots(figsize=(9, 5.2))
for tag, c in configs.items():
    layers = c["data"]["layers"]
    series_per_layer = [v["saturation_rate"]["series"] for v in layers.values() if "saturation_rate" in v]
    n_steps = min(len(s) for s in series_per_layer)
    mean_per_step = [sum(s[i] for s in series_per_layer) / len(series_per_layer) for i in range(n_steps)]
    ax.plot(range(n_steps), mean_per_step, label=c["label"], color=c["color"], linewidth=2)
ax.set_xlabel("training step")
ax.set_ylabel("fraction of weights on counter boundary")
ax.set_title("Saturation rate during training: barely grows with model width\n(bounded-counter saturation is ~constant across d=512/1024/2048)",
             fontsize=11.5, fontweight="bold", pad=12)
ax.legend()
ax.set_ylim(0, 0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "6_saturation_vs_scale.png"), dpi=150)
plt.close()
print("OK 6_saturation_vs_scale.png")

# 2. Bar chart: saturation vs loss
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
tags = ["d512", "d1024", "d2048"]
sat_means = []
for t in tags:
    sats = [v["saturation_rate"]["mean"] for v in configs[t]["data"]["layers"].values() if "saturation_rate" in v]
    sat_means.append(sum(sats) / len(sats))
val_losses = [configs[t]["data"]["val_loss"] for t in tags]
colors = [configs[t]["color"] for t in tags]
axes[0].bar([configs[t]["label"] for t in tags], sat_means, color=colors, edgecolor="black", linewidth=0.6)
axes[0].set_ylabel("Mean saturation rate")
axes[0].set_title("Saturation: plateau from d=1024", fontweight="bold")
axes[0].set_ylim(0, 0.15)
for i, v in enumerate(sat_means):
    axes[0].text(i, v + 0.003, f"{v:.3f}", ha="center", fontweight="bold")
axes[1].bar([configs[t]["label"] for t in tags], val_losses, color=colors, edgecolor="black", linewidth=0.6)
axes[1].set_ylabel("Validation loss (200 steps)")
axes[1].set_title("Loss grows with d - saturation does NOT", fontweight="bold")
for i, v in enumerate(val_losses):
    axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "7_saturation_vs_loss.png"), dpi=150)
plt.close()
print("OK 7_saturation_vs_loss.png")

# Print key numbers
print("\n=== NUMBERS ===")
for t in tags:
    d = configs[t]["data"]
    sats = [v["saturation_rate"]["mean"] for v in d["layers"].values() if "saturation_rate" in v]
    flps = [v["flip_rate_alt"]["mean"] for v in d["layers"].values() if "flip_rate_alt" in v]
    print(f"  {configs[t]['label']:18s} sat={sum(sats)/len(sats):.4f}  flip={sum(flps)/len(flps):.4f}  val={d['val_loss']:.3f}  peak={d.get('peak_gib',0):.2f}GiB")
