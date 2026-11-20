"""Collect all measured results into one Markdown file (RESULTS.md).

    python summarize_results.py

Reads baseline_runs/run_NN_history.json (baseline accuracies) and every
results/**/results.csv (compression runs: results/qat0, results/qat3, results/qat5
and the final --save_model run in results/). Sizes are MiB (1,048,576 bytes).
"""
import csv
import glob
import json
import os
import config as cfg


HEADER = ["| Run | Width | Optimizer | Schedule | Epochs | Best val top-1 (%) | Best epoch | Test top-1 (%) |",
          "|---|---|---|---|---|---|---|---|"]


def run_row(n, h):
    c = h["config"]
    return (f"| {n} | {c['width_mult']} | {c['optimizer']} | {c['schedule']} | {c['epochs']} | "
            f"{h['best_val_acc']:.2f} | {h['best_epoch']} | {h['best_checkpoint_test_acc']:.2f} |")


def baseline_tables():
    """All baseline runs, plus the run with the best validation accuracy and the run
    with the best test accuracy, each shown separately."""
    rows, done = list(HEADER), {}
    for n in range(1, len(cfg.BASELINE_RUNS) + 1):
        path = os.path.join(cfg.BASELINE_RUN_DIR, f"run_{n:02d}_history.json")
        if not os.path.isfile(path):
            rows.append(f"| {n} | – | – | – | – | not run | – | – |")
            continue
        with open(path) as f:
            done[n] = json.load(f)
        rows.append(run_row(n, done[n]))
    if not done:
        return rows, ["No baseline runs found."], ["No baseline runs found."]
    # Ties go to the smaller run number, as in select_best_model.py.
    val_n = max(done, key=lambda n: (done[n]["best_val_acc"], -n))
    test_n = max(done, key=lambda n: (done[n]["best_checkpoint_test_acc"], -n))
    best_val = [*HEADER, run_row(val_n, done[val_n]), "",
                "Used as the baseline for all compression runs (best_baseline.pt)."]
    best_test = [*HEADER, run_row(test_n, done[test_n]), "",
                 "For comparison only (best_test_baseline.pt); selecting by test accuracy is biased."]
    return rows, best_val, best_test


def load_compression_runs(root=cfg.RESULTS_DIRECTORY):
    """Every results.csv under results/ (e.g. results/qat0, results/qat3, results/qat5, results/).
    A configuration measured more than once (e.g. a sweep run and the later --save_model run)
    is listed once, keeping the row that has a verified packed checkpoint if there is one."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "results.csv"), recursive=True)):
        with open(path) as f:
            for r in csv.DictReader(f):
                r.setdefault("dw_bits", "8")          # files written before dw_bits was logged
                key = (int(r["w_bits"]), int(r["a_bits"]), int(r["qat_epochs"]), int(r["dw_bits"] or 8))
                if key not in runs or r.get("reload_verified") == "True":
                    runs[key] = r
    return [runs[k] for k in sorted(runs, key=lambda k: (k[2], -k[0], -k[1]))]


def compression_table():
    runs = load_compression_runs()
    if not runs:
        return ["No compression results found (results/**/results.csv)."], ""
    rows = ["| QAT ep. | W bits | DW bits | A bits | Model size (MiB) | Model CR | Weights CR (incl. overhead) | "
            "Activation CR | Baseline top-1 (%) | Quantized top-1 (%) | Drop (pp) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    num = lambda r, k: float(r[k])
    for r in runs:
        rows.append(f"| {r['qat_epochs']} | {r['w_bits']} | {r['dw_bits']} | {r['a_bits']} | {num(r, 'model_size_mb'):.3f} | "
                    f"{num(r, 'compression_ratio'):.2f}x | {num(r, 'weights_ratio_with_overhead'):.2f}x | "
                    f"{num(r, 'activation_ratio_total'):.2f}x | {num(r, 'baseline_acc'):.2f} | "
                    f"{num(r, 'quantized_acc'):.2f} | {num(r, 'accuracy_drop_pp'):.2f} |")
    best = max(runs, key=lambda r: float(r["quantized_acc"]))
    above4 = max((r for r in runs if float(r["compression_ratio"]) > 7), key=lambda r: float(r["quantized_acc"]))
    summary = (f"{len(runs)} compression runs. Best compressed test top-1: {float(best['quantized_acc']):.2f} % at "
               f"W{best['w_bits']}/A{best['a_bits']}/QAT{best['qat_epochs']}, {float(best['compression_ratio']):.2f}x. "
               f"Best above 7x compression: {float(above4['quantized_acc']):.2f} % at "
               f"W{above4['w_bits']}/A{above4['a_bits']}/QAT{above4['qat_epochs']}, "
               f"{float(above4['compression_ratio']):.2f}x ({float(above4['model_size_mb']):.3f} MiB).")
    return rows, summary


if __name__ == "__main__":
    all_rows, best_val, best_test = baseline_tables()
    c_rows, c_sum = compression_table()
    text = "\n".join(["## Baseline runs", "", *all_rows, "",
                      "## Baseline with the best validation accuracy", "", *best_val, "",
                      "## Baseline with the best test accuracy", "", *best_test, "",
                      "## Compression runs", "", *c_rows, "", c_sum, ""])
    with open("RESULTS.md", "w") as f:
        f.write(text)
    print(text)
    print("Written to RESULTS.md")
