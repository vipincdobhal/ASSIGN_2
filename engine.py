"""Shared train / eval helpers."""
import random
import numpy as np
import torch
import config as cfg


def set_seed(seed=cfg.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, 100.0 * correct / n


def train_one_epoch(model, loader, opt, crit, device, scaler=None, sched=None):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    use_amp = scaler is not None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            loss = crit(out, y)
        if use_amp:
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(opt); scaler.update()  # step is skipped if grads overflowed
            stepped = scaler.get_scale() >= scale_before  # scale drops only when skipped
        else:
            loss.backward(); opt.step()
            stepped = True
        if sched is not None and stepped:  # avoids "scheduler.step() before optimizer.step()" warning
            sched.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, 100.0 * correct / n