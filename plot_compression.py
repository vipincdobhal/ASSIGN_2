"""Q3 figures from all compression runs (every results/**/results.csv).

    python plot_compression.py         # writes figures/

figures/compression_tradeoff.png      accuracy vs model compression ratio, per activation width and QAT length
figures/parallel_coordinates.png      parallel-coordinates view of all runs (same axes as the W&B chart + QAT)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from summarize_results import load_compression_runs

FINAL = ("4", "6", "5")       # final model of the report: w4 a6, 5 QAT epochs


def tradeoff(runs, path, floor=70):
    qats = sorted({r["qat_epochs"] for r in runs}, key=int)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    base = float(runs[0]["baseline_acc"])
    for i, a in enumerate(["4", "6", "8"]):
        for q, marker in zip(qats, "os^dv"):
            pts = sorted((float(r["compression_ratio"]), float(r["quantized_acc"])) for r in runs
                         if r["a_bits"] == a and r["qat_epochs"] == q)
            ax[i].plot([p[0] for p in pts], [max(p[1], floor + .3) for p in pts], marker=marker, label=f"QAT {q} epochs")
            for x, y in pts:          # points below the axis floor are drawn on it and labelled
                if y < floor:
                    ax[i].annotate(f"{y:.1f}", (x, floor + .3), textcoords="offset points",
                                   xytext=(4, 4 + 9 * int(q)), fontsize=7)
        ticks = sorted({(float(r["compression_ratio"]), r["w_bits"]) for r in runs if r["a_bits"] == a})
        ax[i].set_xticks([t[0] for t in ticks], [f"w{b}\n{x:.1f}×" for x, b in ticks], fontsize=8)
        ax[i].axhline(base, ls="--", c="gray", lw=1, label=f"FP32 baseline {base:.2f}%")
        ax[i].set(title=f"{a}-bit activations", xlabel="weight bits / model compression ratio", ylim=(floor, 96))
        ax[i].grid(alpha=.3)
    ax[0].set_ylabel("test top-1 (%)"); ax[0].legend(fontsize=8, loc="lower left")
    fig.text(.5, .005, f"Points below {floor}% are drawn on the axis and labelled with their value.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(path, dpi=160)


def parallel_coordinates(runs, path):
    axes = [("a_bits", "activation_quant_bits"), ("w_bits", "weight_quant_bits"), ("qat_epochs", "qat_epochs"),
            ("compression_ratio", "compression_ratio"), ("model_size_mb", "model_size_mb (MiB)"),
            ("quantized_acc", "quantized_acc (%)")]
    data = np.array([[float(r[k]) for k, _ in axes] for r in runs])
    lo, hi = data.min(0), data.max(0)
    norm = (data - lo) / np.where(hi > lo, hi - lo, 1)
    acc = data[:, -1]; cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in np.argsort(acc):
        ax.plot(range(len(axes)), norm[i], color=cmap((acc[i] - acc.min()) / (np.ptp(acc) or 1)), lw=1.3, alpha=.85)
    final = [i for i, r in enumerate(runs) if (r["w_bits"], r["a_bits"], r["qat_epochs"]) == FINAL]
    if final:
        ax.plot(range(len(axes)), norm[final[0]], color="red", lw=3,
                label=f"final model: w{FINAL[0]} a{FINAL[1]}, QAT {FINAL[2]} ({acc[final[0]]:.2f}%)")
        ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, 1.08))
    for j in range(len(axes)):
        ax.axvline(j, color="black", lw=.8)
        for t in np.linspace(0, 1, 5):
            ax.text(j + .03, t, f"{lo[j] + t * (hi[j] - lo[j]):.1f}", fontsize=7, va="center")
    ax.set_xticks(range(len(axes)), [name for _, name in axes], fontsize=9)
    ax.set_yticks([]); ax.set_xlim(-.1, len(axes) - .6)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(acc.min(), acc.max())), ax=ax, pad=.01,
                 label="quantized_acc (%)")
    ax.set_title(f"All {len(runs)} compression runs", pad=22)
    fig.tight_layout(); fig.savefig(path, dpi=160)


def main():
    runs = load_compression_runs()
    if not runs:
        raise SystemExit("No results/**/results.csv found. Run run_compress.py first.")
    os.makedirs(cfg.FIGURES_DIRECTORY, exist_ok=True)
    tradeoff(runs, os.path.join(cfg.FIGURES_DIRECTORY, "compression_tradeoff.png"))
    parallel_coordinates(runs, os.path.join(cfg.FIGURES_DIRECTORY, "parallel_coordinates.png"))
    print(f"{len(runs)} runs plotted. Figures written to {cfg.FIGURES_DIRECTORY}/")


if __name__ == "__main__":
    main()
