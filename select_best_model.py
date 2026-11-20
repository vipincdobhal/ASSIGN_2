"""Select TWO baseline winners: by validation top-1 and by test top-1.
Only the validation winner is used by compression.
"""
import argparse
import json
import os
import shutil
import config as cfg


def read_results(allow_partial=False):
    results = []
    missing = []
    for number in range(1, len(cfg.BASELINE_RUNS) + 1):
        prefix = os.path.join(cfg.BASELINE_RUN_DIR, f"run_{number:02d}")
        path = prefix + "_history.json"
        checkpoint = prefix + ".pt"
        if not (os.path.isfile(path) and os.path.isfile(checkpoint)):
            missing.append(number)
            continue
        with open(path, encoding="utf-8") as file:
            history = json.load(file)
        results.append({"run_id": number, "best_val_acc": history["best_val_acc"],
                        "test_acc": history["best_checkpoint_test_acc"],
                        "test_loss": history["best_checkpoint_test_loss"],
                        "best_epoch": history["best_epoch"],
                        "config": history["config"], "checkpoint": checkpoint})
    if missing and not allow_partial:
        raise RuntimeError(f"Missing runs {missing}. Finish all 11, or pass --allow-partial.")
    if not results:
        raise RuntimeError("No completed baseline runs found.")
    return results, missing


def save_winner(winner, checkpoint_path, config_path):
    shutil.copyfile(winner["checkpoint"], checkpoint_path)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(winner, file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true",
                        help="Compare only completed runs; rerun after all 11 finish")
    args = parser.parse_args()
    results, missing = read_results(args.allow_partial)
    # Tie-break: smaller run number.
    val_best = max(results, key=lambda x: (x["best_val_acc"], -x["run_id"]))
    test_best = max(results, key=lambda x: (x["test_acc"], -x["run_id"]))
    save_winner(val_best, cfg.BEST_BASELINE_CHECKPOINT, cfg.BEST_BASELINE_CONFIG)
    save_winner(test_best, cfg.BEST_TEST_CHECKPOINT, cfg.BEST_TEST_CONFIG)
    report = {"completed_runs": len(results), "missing_runs": missing,
              "validation_winner": val_best, "test_winner": test_best,
              "compression_source": cfg.BEST_BASELINE_CHECKPOINT,
              "note": "Test-selected winner is diagnostic; test-based selection is biased."}
    with open(cfg.BASELINE_RESULTS, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Validation winner: run {val_best['run_id']} | val top-1={val_best['best_val_acc']:.2f}% | test top-1={val_best['test_acc']:.2f}%")
    print(f"Test winner:       run {test_best['run_id']} | test top-1={test_best['test_acc']:.2f}% | val top-1={test_best['best_val_acc']:.2f}%")
    print(f"Compression uses ONLY validation winner: {cfg.BEST_BASELINE_CHECKPOINT}")
    if missing:
        print(f"WARNING: partial comparison, missing runs: {missing}")


if __name__ == "__main__":
    main()
