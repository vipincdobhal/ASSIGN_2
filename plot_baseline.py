"""Q1c figures from the baseline training histories.

    python plot_baseline.py            # reads baseline_runs/run_NN_history.json, writes figures/

figures/best_run_curves.png   loss and top-1 accuracy of the run with the best validation accuracy
figures/all_runs_val_acc.png  validation accuracy of every baseline run
"""
import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg


def main():
    os.makedirs(cfg.FIGURES_DIRECTORY, exist_ok=True)
    runs = {int(os.path.basename(f)[4:6]): json.load(open(f))
            for f in glob.glob(os.path.join(cfg.BASELINE_RUN_DIR, "run_*_history.json"))}
    if not runs:
        raise SystemExit(f"No histories in {cfg.BASELINE_RUN_DIR}/. Run baseline_sweep.py first.")
    best = max(runs, key=lambda n: (runs[n]["best_val_acc"], -n))   # same rule as select_best_model.py
    h = runs[best]
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, h["train_loss"], label="train")
    ax[0].plot(epochs, h["val_loss"], label="validation")
    ax[0].set(title=f"Run {best:02d}: loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(epochs, h["train_acc"], label="train")
    ax[1].plot(epochs, h["val_acc"], label="validation")
    ax[1].axhline(h["best_checkpoint_test_acc"], ls="--", c="gray",
                  label=f"test {h['best_checkpoint_test_acc']:.2f}%")
    ax[1].set(title=f"Run {best:02d}: top-1 accuracy", xlabel="epoch", ylabel="%", ylim=(40, 100))
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES_DIRECTORY, "best_run_curves.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in sorted(runs):
        c = runs[n]["config"]
        ax.plot(range(1, len(runs[n]["val_acc"]) + 1), runs[n]["val_acc"], lw=2.2 if n == best else .9,
                label=f"{n:02d} {c['optimizer']} {c['schedule']} w{c['width_mult']}")
    ax.set(title=f"Validation accuracy, all {len(runs)} baseline runs", xlabel="epoch", ylabel="%", ylim=(80, 96))
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES_DIRECTORY, "all_runs_val_acc.png"), dpi=160)
    print(f"Best validation run: {best:02d}. Figures written to {cfg.FIGURES_DIRECTORY}/")


if __name__ == "__main__":
    main()
