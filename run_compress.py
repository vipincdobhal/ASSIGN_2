"""Q2-Q4: compression experiments.

Single run:  python run_compress.py --w_bits 4 --a_bits 6 [--qat_epochs 5] [--save_model] [--wandb]
             (--weight_quant_bits / --activation_quant_bits are accepted as aliases, as in the
             template repository; test.py is a thin wrapper with the template's command line)
Q3 sweep:    python run_compress.py --sweep --sweep_qat_epochs 5 --out_dir results/qat5
             W&B grid over cfg.SWEEP_WEIGHT_BITS x cfg.SWEEP_ACTIVATION_BITS (5 x 3 = 15 runs)
             -> Parallel Coordinates chart. Other flags (e.g. --dw_bits 8) apply to every run.
"""
import argparse, copy, json, os, csv
import torch
import config as cfg
from data import get_loaders
from model import build_mobilenetv2
from engine import set_seed, get_device, evaluate
from memory import measure_inference_memory
import compress as C


def run(a):
    dev = get_device()
    # The winning baseline can have non-default width, dropout and BN settings.
    if os.path.exists(cfg.BEST_BASELINE_CONFIG) and a.ckpt == cfg.BEST_BASELINE_CHECKPOINT:
        with open(cfg.BEST_BASELINE_CONFIG) as f:
            selected = json.load(f)["config"]
        a.width_mult = selected["width_mult"]
        a.dropout = selected["dropout"]
        a.bn_eps = selected["bn_eps"]
        a.bn_momentum = selected["bn_momentum"]
        a.seed = selected["seed"]
        a.batch_size = selected["batch_size"]
    if not os.path.isfile(a.ckpt):
        raise FileNotFoundError(f"Missing baseline checkpoint {a.ckpt}. Run baseline_sweep.py and select_best_model.py first.")
    set_seed(a.seed)
    train_loader, val_loader, test_loader, cal_loader = get_loaders(
        batch_size=a.batch_size, calibration=True,
        validation_fraction=cfg.VALIDATION_FRACTION, seed=a.seed)
    base = build_mobilenetv2(width_mult=a.width_mult, dropout=a.dropout,
                              bn_eps=a.bn_eps, bn_momentum=a.bn_momentum).to(dev)
    base.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
    _, base_acc = evaluate(base, test_loader, dev)

    q = copy.deepcopy(base)
    C.convert(q, a.w_bits, a.a_bits, a.first_last_bits, a.dw_bits)
    C.calibrate(q, cal_loader, dev, a.calib_batches)
    if a.qat_epochs > 0:
        C.qat_finetune(q, train_loader, dev, a.qat_epochs)
    C.freeze(q, pack_activations=a.pack_activations)
    _, q_acc = evaluate(q, test_loader, dev)

    rep = C.size_report(base, q, dev, batch_size=a.measure_batch_size)
    # Measured RAM/VRAM of each model on its own: resident tensors + forward peak.
    mem_b = measure_inference_memory(base, dev, a.measure_batch_size, cfg.RUNTIME_REPEATS)
    mem_q = measure_inference_memory(q, dev, a.measure_batch_size, cfg.RUNTIME_REPEATS)
    rep.update({f'baseline_{k}': v for k, v in mem_b.items()})
    rep.update({f'compressed_{k}': v for k, v in mem_q.items()})
    rep['resident_ratio_measured'] = mem_b['resident_mb'] / mem_q['resident_mb']
    rep['forward_peak_ratio_measured'] = mem_b['forward_peak_mb'] / mem_q['forward_peak_mb']
    rep['inference_memory_ratio_measured'] = mem_b['inference_total_mb'] / mem_q['inference_total_mb']
    rep.update(baseline_acc=base_acc, quantized_acc=q_acc, w_bits=a.w_bits, a_bits=a.a_bits,
               qat_epochs=a.qat_epochs, first_last_bits=a.first_last_bits,
               dw_bits=a.dw_bits or a.w_bits, input_bits=cfg.INPUT_BITS)
    rep['accuracy_drop_pp'] = base_acc - q_acc
    rep['layer_precision'] = [
        {'name': name, 'weight_bits': mod.w_bits, 'activation_bits': mod.act.bits,
         'weight_shape': list(mod.wshape), 'packed_weight_bytes': mod.wq.numel()}
        for name, mod in q.named_modules() if isinstance(mod, C.QLayer)
    ]
    rep['pack_activations'] = a.pack_activations
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f'w{a.w_bits}_a{a.a_bits}_dw{a.dw_bits or a.w_bits}_qat{a.qat_epochs}'
    # Same columns for every run (with or without --save_model), so results.csv keeps one header.
    rep['packed_checkpoint'] = ''
    rep['reload_verified'] = False
    if a.save_model:
        config = {k: getattr(a, k) for k in ('width_mult', 'dropout', 'w_bits', 'a_bits',
                                            'first_last_bits', 'dw_bits', 'pack_activations', 'bn_eps', 'bn_momentum')}
        ckpt = os.path.join(a.out_dir, f'{tag}_packed.pt')
        C.save_frozen(q, ckpt, config)
        reloaded = C.load_frozen(ckpt, dev)
        _, reloaded_acc = evaluate(reloaded, test_loader, dev)
        if abs(reloaded_acc - q_acc) > 1e-5:
            raise RuntimeError('Reloaded compressed checkpoint changes accuracy')
        rep['packed_checkpoint'] = ckpt
        rep['reload_verified'] = True
    with open(os.path.join(a.out_dir, f'{tag}.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    # Written last: a run whose reload check failed never reaches results.csv.
    append_csv(os.path.join(a.out_dir, 'results.csv'), rep)
    print(json.dumps({k: v for k, v in rep.items() if k != 'layer_precision'}, indent=2))
    return rep


def append_csv(path, rep):
    """One row per run (scalar fields only). A file written by an older version of
    this script, with different columns, is renamed instead of being mixed in."""
    fields = [k for k, v in rep.items() if not isinstance(v, (list, dict))]
    if os.path.exists(path):
        with open(path, newline='') as f:
            header = next(csv.reader(f), [])
        if header != fields:
            n = 1
            while os.path.exists(path.replace('.csv', f'_old{n}.csv')):
                n += 1
            os.replace(path, path.replace('.csv', f'_old{n}.csv'))
            print(f"results.csv had different columns; moved to {path.replace('.csv', f'_old{n}.csv')}")
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(rep)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=cfg.BEST_BASELINE_CHECKPOINT)
    p.add_argument("--w_bits", "--weight_quant_bits", dest="w_bits", type=int, default=cfg.WEIGHT_BITS,
                   help="weight bits of the pointwise (1x1) convs, 2-8")
    p.add_argument("--a_bits", "--activation_quant_bits", dest="a_bits", type=int, default=cfg.ACTIVATION_BITS,
                   help="activation bits of the inputs of all inner layers, 2-8")
    p.add_argument("--first_last_bits", type=int, default=cfg.FIRST_LAST_BITS)
    p.add_argument("--dw_bits", type=int, default=cfg.DEPTHWISE_BITS)
    p.add_argument("--qat_epochs", type=int, default=cfg.QAT_EPOCHS)
    p.add_argument("--calib_batches", type=int, default=cfg.CALIBRATION_BATCHES)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--wandb", action="store_true", default=cfg.WANDB_ENABLED)
    p.add_argument("--width_mult", type=float, default=cfg.WIDTH_MULT)
    p.add_argument("--dropout", type=float, default=cfg.DROPOUT)
    p.add_argument("--bn_eps", type=float, default=cfg.BN_EPS)
    p.add_argument("--bn_momentum", type=float, default=cfg.BN_MOMENTUM)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--measure_batch_size", type=int, default=cfg.MEASUREMENT_BATCH_SIZE)
    p.add_argument("--out_dir", default=cfg.RESULTS_DIRECTORY)
    p.add_argument("--save_model", action="store_true", default=cfg.SAVE_COMPRESSED_MODEL)
    p.add_argument("--no_pack_activations", dest="pack_activations", action="store_false")
    p.set_defaults(pack_activations=cfg.PACK_ACTIVATIONS)
    p.add_argument("--sweep", action="store_true",
                   help="Run the Q3 W&B grid sweep instead of a single experiment")
    p.add_argument("--sweep_qat_epochs", type=int, default=cfg.SWEEP_QAT_EPOCHS,
                   help="QAT epochs used by every run of --sweep (the report uses 0, 3 and 5)")
    return p.parse_args(argv)


# ----------------------------------------------------------------- W&B / Q3 sweep
def sweep_config(qat_epochs):
    """W&B grid: every weight-bit x activation-bit combination at a fixed QAT length."""
    return {
        "method": "grid",                                        # try every combination
        "metric": {"name": "quantized_acc", "goal": "maximize"},
        "parameters": {
            "w_bits": {"values": cfg.SWEEP_WEIGHT_BITS},
            "a_bits": {"values": cfg.SWEEP_ACTIVATION_BITS},
            "qat_epochs": {"value": qat_epochs},
        },
    }


def log_metrics(wandb, rep, a):
    """Log scalar results only (charts need numbers) plus the sample-plot axis names."""
    metrics = {k: v for k, v in rep.items() if isinstance(v, (int, float))}
    metrics.update(weight_quant_bits=a.w_bits, activation_quant_bits=a.a_bits)  # sample-chart axis names
    wandb.log(metrics)


def run_sweep(base):
    """Each grid trial = run() with the bit-widths W&B picked; all other args from `base`."""
    import wandb

    def trial():
        wandb.init()
        a = copy.deepcopy(base)
        a.w_bits, a.a_bits, a.qat_epochs = (wandb.config[k] for k in ("w_bits", "a_bits", "qat_epochs"))
        log_metrics(wandb, run(a), a)
        wandb.finish()

    sweep_id = wandb.sweep(sweep_config(base.sweep_qat_epochs), project=cfg.WANDB_PROJECT)
    wandb.agent(sweep_id, function=trial)


def main(argv=None):
    a = get_args(argv)
    if a.sweep:
        run_sweep(a)
    elif a.wandb:
        import wandb
        wandb.init(project=cfg.WANDB_PROJECT, config=vars(a))
        log_metrics(wandb, run(a), a)
        wandb.finish()
    else:
        run(a)


if __name__ == "__main__":
    main()
