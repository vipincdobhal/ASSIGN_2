"""MobileNet-v2 written from scratch for 32x32 CIFAR-10 inputs (Q1b).
No torchvision model code is used (torchvision is only used for the dataset)."""
import torch.nn as nn
import config as cfg
from functools import partial

def conv_bn_relu(cin, cout, k, bn, stride=1, groups=1):
    """Conv -> BatchNorm -> ReLU6. The BN is later folded into the conv by compress.py."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
        bn(cout),
        nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    """1x1 expand (skipped when t == 1) -> 3x3 depthwise -> 1x1 linear projection (no ReLU)."""

    def __init__(self, cin, cout, stride, t, bn):
        super().__init__()
        hid = cin * t
        self.use_res = stride == 1 and cin == cout          # skip connection only if shapes match
        layers = [conv_bn_relu(cin, hid, 1, bn)] if t != 1 else []
        layers += [conv_bn_relu(hid, hid, 3, bn, stride, groups=hid),   # depthwise
                   nn.Conv2d(hid, cout, 1, bias=False), bn(cout)]      # linear projection
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out


def _make_divisible(v, d=8):
    """Round channel count to a multiple of d, never dropping more than 10%."""
    n = max(d, int(v + d / 2) // d * d)
    return n + d if n < 0.9 * v else n


class MobileNetV2(nn.Module):
    # (expansion t, out channels c, repeats n, first stride s).
    # CIFAR change: stem stride 1 and second stage stride 1 (both 2 in ImageNet MobileNet-v2),
    # so 32x32 inputs are downsampled 32 -> 16 -> 8 -> 4 instead of to 1x1.
    SETTING = [[1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 2], [6, 64, 4, 2],
               [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]

    def __init__(self, num_classes=cfg.NUM_CLASSES, width_mult=cfg.WIDTH_MULT, dropout=cfg.DROPOUT,
                 bn_eps=cfg.BN_EPS, bn_momentum=cfg.BN_MOMENTUM):
        super().__init__()
        bn = partial(nn.BatchNorm2d, eps=bn_eps, momentum=bn_momentum)  # same BN settings everywhere
        cin = _make_divisible(32 * width_mult)
        last = _make_divisible(1280 * max(1.0, width_mult))  # last layer never shrinks below 1280

        layers = [conv_bn_relu(3, cin, 3, bn)]               # stem
        for t, c, n, s in self.SETTING:
            cout = _make_divisible(c * width_mult)
            for i in range(n):
                layers.append(InvertedResidual(cin, cout, s if i == 0 else 1, t, bn))
                cin = cout
        layers.append(conv_bn_relu(cin, last, 1, bn))       # final 1x1 conv

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(last, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)


def build_mobilenetv2(num_classes=cfg.NUM_CLASSES, width_mult=cfg.WIDTH_MULT, dropout=cfg.DROPOUT,
                      bn_eps=cfg.BN_EPS, bn_momentum=cfg.BN_MOMENTUM):
    return MobileNetV2(num_classes, width_mult, dropout, bn_eps, bn_momentum)