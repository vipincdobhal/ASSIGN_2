"""CIFAR-10 data loading (Q1a).

The per-channel mean/std used by Normalize are computed from the CIFAR-10
TRAINING set only (never the test set) and are reused for train, test and the
compressed-model evaluation, because every script obtains its loaders from
get_loaders().
"""
import numpy as np
import torch
import torchvision
import config as cfg
import torchvision.transforms as transform
from torch.utils.data import DataLoader, Subset
from functools import partial


def calculate_mean_std(images):
    """Mean and std of each colour channel (R, G, B), with pixels scaled to 0..1.
    images: uint8 array of shape (N, 32, 32, 3). Returns two lists of 3 floats."""
    channels = [images[..., c].astype(np.float32) / 255.0 for c in range(3)]
    return [float(x.mean()) for x in channels], [float(x.std()) for x in channels]


def get_mean_std(root=cfg.DATA_ROOT):
    """Channel statistics of the raw training images (no transforms)."""
    raw = torchvision.datasets.CIFAR10(root, train=True, download=True)  # .data: (50000,32,32,3) uint8
    return calculate_mean_std(raw.data)


def get_loaders(root=cfg.DATA_ROOT, batch_size=cfg.BATCH_SIZE, workers=cfg.WORKERS, calibration=False,
                validation_fraction=0.0, seed=cfg.SEED):
    """Return the loaders, always in this order:
        train, [val if validation_fraction], test, [calibration if calibration=True]
    """
    if validation_fraction and not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    mean, std = get_mean_std(root)
    print(f"CIFAR-10 train mean={[round(m, 4) for m in mean]} std={[round(s, 4) for s in std]}")

    to_normalized_tensor = [transform.ToTensor(),              # uint8 0..255 -> float 0..1
                            transform.Normalize(mean, std)]    # (x - mean) / std per channel
    train_tf = transform.Compose([
        transform.RandomCrop(cfg.CROP_SIZE, padding=cfg.CROP_PADDING),        # random translation
        transform.RandomHorizontalFlip(p=cfg.HORIZONTAL_FLIP_PROBABILITY),    # mirror augmentation
        *to_normalized_tensor,
        transform.RandomErasing(p=cfg.RANDOM_ERASING_PROBABILITY),            # cutout-style regularisation
    ])
    test_tf = transform.Compose(to_normalized_tensor)                         # no augmentation

    cifar = partial(torchvision.datasets.CIFAR10, root, download=True)
    train_set = cifar(train=True, transform=train_tf)
    clean_train = cifar(train=True, transform=test_tf)  # same images, no augmentation (val + calibration)
    test_set = cifar(train=False, transform=test_tf)

    val_set = None
    if validation_fraction:
        # Seeded split of the 50,000 training images into train / validation.
        indices = np.random.default_rng(seed).permutation(len(train_set))
        n_val = int(round(len(train_set) * validation_fraction))
        val_idx, train_idx = indices[:n_val].tolist(), indices[n_val:].tolist()
        train_set = Subset(train_set, train_idx)
        val_set = Subset(clean_train, val_idx)
        clean_train = Subset(clean_train, train_idx)  # calibration never sees validation images

    loader = partial(DataLoader, num_workers=workers, pin_memory=torch.cuda.is_available())
    loaders = [loader(train_set, batch_size, shuffle=True)]
    if val_set is not None:
        loaders.append(loader(val_set, cfg.TEST_BATCH_SIZE, shuffle=False))
    loaders.append(loader(test_set, cfg.TEST_BATCH_SIZE, shuffle=False))
    if calibration:
        loaders.append(loader(clean_train, batch_size, shuffle=False))
    return tuple(loaders)