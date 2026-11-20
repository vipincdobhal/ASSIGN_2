"""Train a single baseline or all eleven baselines.
Examples: python baseline_sweep.py --run 3 --wandb
          python baseline_sweep.py --all --wandb
"""
import argparse
import json
import os
import subprocess
import sys
import config as cfg


def main():
    parser = argparse.ArgumentParser(description="Train selected MobileNet-v2 baselines")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run", type=int, choices=range(1, len(cfg.BASELINE_RUNS) + 1),
                           metavar="N", help="Run only experiment N (1-11)")
    selection.add_argument("--all", action="store_true", help="Run all 11 experiments")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun even if completed")
    args = parser.parse_args()
    os.makedirs(cfg.BASELINE_RUN_DIR, exist_ok=True)
    numbers = range(1, len(cfg.BASELINE_RUNS) + 1) if args.all else [args.run]
    for number in numbers:
        params = cfg.BASELINE_RUNS[number - 1]
        prefix = os.path.join(cfg.BASELINE_RUN_DIR, f"run_{number:02d}")
        history_path = prefix + "_history.json"
        if not args.force and os.path.isfile(history_path) and os.path.isfile(prefix + ".pt"):
            print(f"Run {number} already completed; skipping (use --force to repeat)", flush=True)
            continue
        cmd = [sys.executable, str(os.path.join(os.path.dirname(__file__), "train.py")),
               "--out", prefix + ".pt", "--history_out", history_path,
               "--class_accuracy_out", prefix + "_class_accuracy.json",
               "--seed", str(cfg.SEED), "--val_fraction", str(cfg.VALIDATION_FRACTION)]
        settings = {"width_mult": params["width"], "dropout": params["dropout"],
                    "bn_momentum": params["bn"], "bn_eps": params["bn_eps"],
                    "optimizer": params["optimizer"], "schedule": params["schedule"],
                    "lr": params["lr"], "wd": params["wd"],
                    "epochs": params["epochs"], "batch_size": params["batch"]}
        for key, value in settings.items():
            cmd.extend(["--" + key, str(value)])
        if args.wandb:
            cmd.append("--wandb")
        print(f"Starting baseline run {number}/{len(cfg.BASELINE_RUNS)}", flush=True)
        subprocess.run(cmd, check=True)
    print("Training finished. Run: python select_best_model.py")


if __name__ == "__main__":
    main()
