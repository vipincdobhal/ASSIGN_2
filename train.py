"""Q1: train the uncompressed baseline.   python train.py --epochs 100"""
import argparse, json, os
import torch
import config as cfg
from data import get_loaders
from model import build_mobilenetv2
from engine import set_seed, get_device, evaluate, train_one_epoch


def get_args():
    p = argparse.ArgumentParser(description="Train the MobileNet-v2 CIFAR-10 baseline")
    add = p.add_argument
    # Training hyperparameters
    add("--epochs", type=int, default=cfg.EPOCHS)
    add("--batch_size", type=int, default=cfg.BATCH_SIZE)
    add("--lr", type=float, default=cfg.LEARNING_RATE)
    add("--wd", type=float, default=cfg.WEIGHT_DECAY)
    add("--optimizer", choices=["SGD", "AdamW"], default="SGD")
    add("--schedule", choices=["OneCycle", "Cosine"], default="OneCycle")
    # Model configuration
    add("--width_mult", type=float, default=cfg.WIDTH_MULT)
    add("--dropout", type=float, default=cfg.DROPOUT)
    add("--bn_momentum", type=float, default=cfg.BN_MOMENTUM)
    add("--bn_eps", type=float, default=cfg.BN_EPS)
    # Output files
    add("--history_out", default=cfg.TRAIN_HISTORY)
    add("--class_accuracy_out", default=cfg.CLASS_ACCURACY)
    add("--out", default=cfg.CHECKPOINT)
    # Data, reproducibility and logging
    add("--seed", type=int, default=cfg.SEED)
    add("--workers", type=int, default=cfg.WORKERS)
    add("--val_fraction", type=float, default=cfg.VALIDATION_FRACTION,
        help="Fraction of CIFAR-10 training images held out for validation")
    add("--deterministic", action="store_true", default=cfg.DETERMINISTIC)
    add("--wandb", action="store_true", default=cfg.WANDB_ENABLED, help="Log training metrics to Weights & Biases")
    add("--wandb_project", default=cfg.WANDB_PROJECT)
    return p.parse_args()


def print_device_info():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Count:", torch.cuda.device_count())
    else:
        print("Running on CPU")


def make_deterministic():
    """Bit-exact reruns (slower). Must run before CUDA is initialised."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_optimizer(model, a):
    if a.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=a.lr, momentum=cfg.SGD_MOMENTUM,
                               nesterov=cfg.NESTEROV, weight_decay=a.wd)
    return torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)


def make_scheduler(opt, a, total_steps):
    """Both schedules are stepped once per mini-batch.
    Note: OneCycleLR keeps its default cycle_momentum=True, so SGD momentum (or AdamW beta1)
    is cycled between 0.85 and 0.95 during OneCycle runs. The reported results use this setting."""
    if a.schedule == "OneCycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps,
                                                   pct_start=cfg.ONE_CYCLE_WARMUP_FRACTION)
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)


@torch.inference_mode()
def per_class_accuracy(model, loader, dev, class_names):
    """Correct predictions and sample counts for each CIFAR-10 class (no confusion matrix)."""
    model.eval()
    correct, total = [0] * len(class_names), [0] * len(class_names)
    for images, targets in loader:
        predictions = model(images.to(dev)).argmax(dim=1).cpu()
        for cls in range(len(class_names)):
            mask = targets == cls
            total[cls] += int(mask.sum().item())
            correct[cls] += int((predictions[mask] == targets[mask]).sum().item())
    accuracy = {name: 100.0 * c / t if t else None for name, c, t in zip(class_names, correct, total)}
    return {"class_accuracy_pct": accuracy,
            "correct": dict(zip(class_names, correct)),
            "total": dict(zip(class_names, total))}


def main():
    print_device_info()
    a = get_args()
    wandb_run = None
    if a.wandb:
        import wandb
        wandb_run = wandb.init(project=a.wandb_project, job_type="baseline", config=vars(a))

    set_seed(a.seed)
    if a.deterministic:
        make_deterministic()
    dev = get_device()
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=a.batch_size, workers=a.workers, validation_fraction=a.val_fraction, seed=a.seed)

    model = build_mobilenetv2(cfg.NUM_CLASSES, a.width_mult, a.dropout, a.bn_eps, a.bn_momentum).to(dev)
    opt = make_optimizer(model, a)
    sched = make_scheduler(opt, a, a.epochs * len(train_loader))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None  # mixed precision on GPU only

    # ---- training loop: keep the checkpoint with the best VALIDATION accuracy
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = float("-inf"), 0
    for ep in range(1, a.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, opt, crit, dev, scaler, sched)
        vl, va = evaluate(model, val_loader, dev)
        for key, value in zip(hist, (tl, ta, vl, va)):
            hist[key].append(value)
        if va > best_val_acc:
            best_val_acc, best_epoch = va, ep
            torch.save(model.state_dict(), a.out)
            print(f"  New best validation accuracy: {va:.2f}% -> saved {a.out}")
        print(f"ep {ep:3d} | train {tl:.3f}/{ta:.2f}% | val {vl:.3f}/{va:.2f}%")
        if wandb_run is not None:
            wandb_run.log({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va,
                           "learning_rate": opt.param_groups[0]["lr"]})

    final_path = os.path.splitext(a.out)[0] + "_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final-epoch weights saved: {final_path}")
    print(f"Best validation checkpoint: {a.out} (epoch {best_epoch}, {best_val_acc:.2f}%)")

    # ---- test set is evaluated ONCE, on the validation-selected checkpoint
    model.load_state_dict(torch.load(a.out, map_location=dev, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, dev)
    print(f"Best-checkpoint test top-1: {test_acc:.2f}%")
    hist.update(config=vars(a), best_epoch=best_epoch, best_val_acc=best_val_acc,
                best_checkpoint_test_loss=test_loss, best_checkpoint_test_acc=test_acc)
    with open(a.history_out, "w") as f:
        json.dump(hist, f, indent=2)

    # ---- per-class test accuracy (material for the failure-mode discussion)
    classes = per_class_accuracy(model, test_loader, dev, cfg.CIFAR10_CLASSES)
    with open(a.class_accuracy_out, "w") as f:
        json.dump(classes, f, indent=2)
    print("Class-wise test accuracy (%):")
    for name, acc in classes["class_accuracy_pct"].items():
        print(f"  {name}: {acc:.2f}%" if acc is not None else f"  {name}: N/A")

    if wandb_run is not None:
        wandb_run.log({f"class_accuracy/{name}": acc
                       for name, acc in classes["class_accuracy_pct"].items() if acc is not None})
        wandb_run.log({
            "test_acc": test_acc,
            "test_loss": test_loss
        })
        wandb_run.summary.update({"best_epoch": best_epoch, "best_val_acc": best_val_acc,
                                  "test_loss": test_loss, "test_acc": test_acc})
        wandb_run.finish()


if __name__ == "__main__":
    main()