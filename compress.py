"""
Own compression library (no compression APIs used).

Pipeline:
  1. Fold BatchNorm into the preceding conv (removes BN params/buffers).
  2. Weights: per-output-channel symmetric uniform quantisation to k bits,
     stored BIT-PACKED in a uint8 tensor (true in-memory size = k bits/weight).
  3. Activations: per-tensor asymmetric uniform quantisation of every conv/linear
     input to a bits (static ranges from calibration / QAT EMA).
  4. Optional QAT fine-tuning with straight-through estimator.
  5. Exceptions: first conv and classifier use `first_last_bits` (sensitive layers).

Inference after freeze():
  * Only packed codes, fp16 scales and fp16 biases stay resident.
  * Weights are dequantised in small output-channel chunks (cfg.WEIGHT_CHUNK_CODES
    codes at a time) immediately before use and freed after the chunk is computed,
    so a full FP32 copy of a layer's weights never exists in memory.
  * Standard PyTorch kernels compute in floating point, so each layer's input and
    output activation exist in FP32 while that layer runs (see README).
"""
import math
import torch
import config as cfg
import torch.nn as nn
import torch.nn.functional as F
from memory import tensor_bytes, measure_inference_memory  # noqa: F401  (re-exported)


# ----------------------------------------------------------------- bit packing
# Layout: LSB-first bit stream. Code i occupies stream bits [i*bits, (i+1)*bits).
# Every group of 8 codes therefore fills exactly `bits` bytes, so both functions
# work on (groups, 8) code / (groups, bits) byte matrices with uint8/int16 scratch
# only - O(N) bytes of temporary memory, no N x bits expansion and no int64 indices.

def _bit_positions(bits):
    """For code j (0..7) of a group: (byte index, shift, spills into next byte)."""
    return [((j * bits) // 8, (j * bits) % 8, (j * bits) % 8 + bits > 8) for j in range(8)]


def pack_bits(codes, bits):
    """codes: integer tensor in [0, 2^bits - 1] -> uint8 tensor of ceil(N*bits/8) bytes."""
    if not 2 <= bits <= 8:
        raise ValueError("bits must be between 2 and 8")
    codes = codes.reshape(-1)
    n = codes.numel()
    if bits == 8:
        return codes.to(torch.uint8).contiguous()
    groups = (n + 7) // 8
    c = torch.zeros(groups * 8, dtype=torch.int16, device=codes.device)
    c[:n] = codes.to(torch.int16)
    c = c.view(groups, 8)
    out = torch.zeros(groups, bits, dtype=torch.int16, device=codes.device)
    for j, (k, s, spill) in enumerate(_bit_positions(bits)):
        v = c[:, j] << s
        out[:, k] |= v & 0xFF
        if spill:
            out[:, k + 1] |= v >> 8
    return out.to(torch.uint8).view(-1)[: (n * bits + 7) // 8].contiguous()


def unpack_bits(packed, bits, n, start=0):
    """Return n uint8 codes starting at code index `start` (must be a multiple of 8)."""
    if not 2 <= bits <= 8:
        raise ValueError("bits must be between 2 and 8")
    if start % 8:
        raise ValueError("start must be a multiple of 8 codes (byte aligned)")
    b0 = start * bits // 8
    if bits == 8:
        return packed.reshape(-1)[b0:b0 + n]
    groups = (n + 7) // 8
    raw = packed.reshape(-1)[b0:b0 + groups * bits]
    if raw.numel() < groups * bits:                      # last partial group
        raw = F.pad(raw, (0, groups * bits - raw.numel()))
    raw = raw.view(groups, bits)
    mask = (1 << bits) - 1
    out = torch.empty(groups, 8, dtype=torch.uint8, device=packed.device)
    for j, (k, s, spill) in enumerate(_bit_positions(bits)):
        v = raw[:, k].to(torch.int32)               # one column: n/8 elements
        if spill:
            v |= raw[:, k + 1].to(torch.int32) << 8
        out[:, j] = (v >> s) & mask
    return out.view(-1)[:n]


# ----------------------------------------------------------------- quantisers
def ste_round(x):
    return (x.round() - x).detach() + x          # straight-through estimator


def quant_weight(w, bits):
    """Per-output-channel symmetric quantisation. Returns integer codes in
    [-qmax, qmax] (float tensor, STE-differentiable) and per-channel scale."""
    qmax = 2 ** (bits - 1) - 1
    scale = (w.detach().reshape(w.shape[0], -1).abs().amax(1).clamp(min=1e-8) / qmax)
    s = scale.view(-1, *[1] * (w.dim() - 1))
    q = torch.clamp(ste_round(w / s), -qmax, qmax)
    return q, scale


class ActQuant(nn.Module):
    """Per-tensor asymmetric activation quantiser with EMA min/max range."""

    def __init__(self, bits, momentum=cfg.ACTIVATION_EMA_MOMENTUM):
        super().__init__()
        self.bits, self.momentum, self.calibrating = bits, momentum, False
        self.last_numel = 0                       # per-sample elements (for measurement)
        self.last_packed_bytes = 0                # bytes of the real packed buffer (pack_inference)
        self.last_actual_output_bytes = 0
        self.pack_inference = False
        self.register_buffer("lo", torch.zeros(()))
        self.register_buffer("hi", torch.ones(()))
        self.register_buffer("init", torch.zeros((), dtype=torch.bool))

    def forward(self, x):
        self.last_numel = x[0].numel()
        self.last_actual_output_bytes = x.numel() * x.element_size()
        qmax = 2 ** self.bits - 1
        if self.training or self.calibrating:
            lo, hi = x.detach().min(), x.detach().max()
            if not self.init:
                self.lo.copy_(lo); self.hi.copy_(hi); self.init.fill_(True)
            else:
                m = self.momentum
                self.lo.copy_(m * self.lo + (1 - m) * lo)
                self.hi.copy_(m * self.hi + (1 - m) * hi)
            scale = ((self.hi - self.lo) / qmax).clamp(min=1e-8)
            q = torch.clamp(ste_round((x - self.lo) / scale), 0, qmax)
            return q * scale + self.lo
        # Inference: same arithmetic, one temporary tensor, all ops in place.
        scale = ((self.hi - self.lo) / qmax).clamp(min=1e-8)
        q = x - self.lo
        q.div_(scale).round_().clamp_(0, qmax)
        if not self.pack_inference:
            self.last_packed_bytes = 0
            return q.mul_(scale).add_(self.lo)
        # Measurement mode: materialise the real k-bit packed buffer, free the FP32
        # codes, and run the layer from the packed buffer (lossless round trip).
        packed = pack_bits(q.to(torch.uint8), self.bits)
        shape, dtype = q.shape, q.dtype
        del q
        self.last_packed_bytes = packed.numel() * packed.element_size()
        restored = unpack_bits(packed, self.bits, math.prod(shape)).to(dtype).view(shape)
        del packed
        return restored.mul_(scale).add_(self.lo)


class QLayer(nn.Module):
    """Wraps Conv2d / Linear. Before freeze(): fake-quant (trainable). After
    freeze(): float weight deleted, only packed ints + fp16 scales/bias remain,
    and the layer runs in output-channel chunks (see module docstring)."""

    def __init__(self, layer, w_bits, a_bits):
        super().__init__()
        self.layer, self.w_bits, self.act = layer, w_bits, ActQuant(a_bits)
        self.is_conv = isinstance(layer, nn.Conv2d)
        self.wshape = tuple(layer.weight.shape)
        self.has_bias = layer.bias is not None
        if self.is_conv:
            self.cfg = dict(stride=layer.stride, padding=layer.padding,
                            dilation=layer.dilation, groups=layer.groups)
            self.depthwise = layer.groups > 1
            if self.depthwise and not (layer.groups == layer.in_channels == layer.out_channels):
                raise NotImplementedError("only standard or depthwise convolutions are supported")
        else:
            self.depthwise = False
        self.packed = False
        # Output channels per chunk: a multiple of 8 (so every chunk starts on a byte
        # boundary of the packed stream) holding at most WEIGHT_CHUNK_CODES weights.
        per_channel = math.prod(self.wshape[1:])
        self.chunk = max(8, (cfg.WEIGHT_CHUNK_CODES // per_channel) // 8 * 8)

    def _weight_chunk(self, c0, c1):
        """Dequantised FP32 weights of output channels [c0, c1) only."""
        per_channel = math.prod(self.wshape[1:])
        qmax = 2 ** (self.w_bits - 1) - 1
        codes = unpack_bits(self.wq, self.w_bits, (c1 - c0) * per_channel, start=c0 * per_channel)
        w = codes.to(torch.float32).view(c1 - c0, *self.wshape[1:])
        shp = (-1,) + (1,) * (len(self.wshape) - 1)
        return w.sub_(qmax).mul_(self.scale[c0:c1].float().view(shp))

    def _packed_forward(self, x):
        out_ch = self.wshape[0]
        out = None
        for c0 in range(0, out_ch, self.chunk):
            c1 = min(out_ch, c0 + self.chunk)
            w = self._weight_chunk(c0, c1)
            b = self.bias_h[c0:c1].float() if self.has_bias else None
            if not self.is_conv:
                y = F.linear(x, w, b)
            elif self.depthwise:
                y = F.conv2d(x[:, c0:c1], w, b, self.cfg['stride'], self.cfg['padding'],
                             self.cfg['dilation'], c1 - c0)
            else:
                y = F.conv2d(x, w, b, self.cfg['stride'], self.cfg['padding'], self.cfg['dilation'], 1)
            del w, b
            if c0 == 0 and c1 == out_ch:          # single chunk: no copy needed
                return y
            if out is None:
                out = y.new_empty((y.shape[0], out_ch) + tuple(y.shape[2:]))
            out[:, c0:c1] = y
            del y
        return out

    def forward(self, x):
        x = self.act(x)
        if self.packed:
            return self._packed_forward(x)
        shp = (-1,) + (1,) * (len(self.wshape) - 1)
        q, s = quant_weight(self.layer.weight, self.w_bits)
        w = q * s.view(shp)
        if self.is_conv:
            return F.conv2d(x, w, self.layer.bias, **self.cfg)
        return F.linear(x, w, self.layer.bias)

    @torch.no_grad()
    def freeze(self):
        q, s = quant_weight(self.layer.weight, self.w_bits)
        qmax = 2 ** (self.w_bits - 1) - 1
        self.register_buffer("wq", pack_bits((q + qmax).round().to(torch.int32), self.w_bits))
        self.register_buffer("scale", s.half())
        if self.has_bias:
            self.register_buffer("bias_h", self.layer.bias.detach().half())
        self.layer = None
        self.packed = True


# ----------------------------------------------------------------- BN folding
@torch.no_grad()
def fold_bn(model):
    model.eval()  # use frozen running statistics
    for mod in list(model.modules()):
        if not isinstance(mod, nn.Sequential):
            continue
        keys = list(mod._modules.keys())
        for i in range(len(keys) - 1):
            conv, bn = mod._modules[keys[i]], mod._modules[keys[i + 1]]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                                conv.padding, conv.dilation, conv.groups, bias=True).to(conv.weight.device)
                new.weight.copy_(conv.weight * g.view(-1, 1, 1, 1))
                b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
                new.bias.copy_((b0 - bn.running_mean) * g + bn.bias)
                mod._modules[keys[i]] = new
                mod._modules[keys[i + 1]] = nn.Identity()
    return model


# ----------------------------------------------------------------- conversion
def convert(model, w_bits, a_bits, first_last_bits=cfg.FIRST_LAST_BITS, dw_bits=None, input_bits=cfg.INPUT_BITS):
    """Fold BN and replace every Conv2d/Linear with a QLayer."""
    dw_bits = dw_bits or w_bits
    for label, value in [('w_bits', w_bits), ('a_bits', a_bits),
                         ('first_last_bits', first_last_bits), ('dw_bits', dw_bits),
                         ('input_bits', input_bits)]:
        if not 2 <= value <= 8:
            raise ValueError(f'{label} must be in [2, 8], got {value}')
    fold_bn(model)
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for k, (name, m) in enumerate(targets):
        first, last = k == 0, k == len(targets) - 1
        depthwise = isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels
        wb = first_last_bits if (first or last) else (dw_bits if depthwise else w_bits)
        ab = input_bits if first else (first_last_bits if last else a_bits)
        parent = model
        *path, leaf = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, leaf, QLayer(m, wb, ab))
    return model


def qlayers(model):
    return [m for m in model.modules() if isinstance(m, QLayer)]


@torch.no_grad()
def calibrate(model, loader, device, batches=cfg.CALIBRATION_BATCHES):
    model.eval()
    acts = [m.act for m in qlayers(model)]
    for a in acts:
        a.calibrating = True
        a.init.fill_(False)
    for i, (x, _) in enumerate(loader):
        if i >= batches: break
        model(x.to(device))
    for a in acts: a.calibrating = False


def qat_finetune(model, loader, device, epochs, lr=cfg.QAT_LEARNING_RATE):
    from engine import train_one_epoch
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SGD_MOMENTUM, weight_decay=cfg.QAT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        l, a = train_one_epoch(model, loader, opt, crit, device, None, sched)
        print(f"  QAT ep {ep+1}/{epochs}: loss {l:.3f} acc {a:.2f}%")


def set_activation_packing(model, enabled):
    for m in qlayers(model):
        m.act.pack_inference = enabled


def freeze(model, pack_activations=cfg.PACK_ACTIVATIONS):
    for m in qlayers(model):
        m.freeze()
    set_activation_packing(model, pack_activations)
    model.eval()
    return model


# ----------------------------------------------------------------- measurement
@torch.no_grad()
def size_report(base_model, qmodel, device, batch_size=cfg.MEASUREMENT_BATCH_SIZE, meta_bytes_per_layer=cfg.METADATA_BYTES_PER_LAYER):
    """Persistent tensor storage and activation-buffer accounting (byte counts).

    The activation packed size is read from the REAL packed buffers created during
    one forward pass with activation packing switched on. Measured allocator
    memory (resident model and forward peak) comes from measure_inference_memory().
    """
    base_total = tensor_bytes(base_model)
    base_w = sum(p.numel() * p.element_size() for p in base_model.parameters() if p.dim() > 1)
    layers = qlayers(qmodel)
    payload = sum(l.wq.numel() * l.wq.element_size() for l in layers)
    scales = sum(l.scale.numel() * l.scale.element_size() for l in layers)
    biases = sum(l.bias_h.numel() * l.bias_h.element_size() for l in layers if l.has_bias)
    act_par = sum(sum(t.numel() * t.element_size() for t in l.act.buffers()) for l in layers)
    meta = meta_bytes_per_layer * len(layers)  # explicitly an estimate, not Python heap
    total = tensor_bytes(qmodel) + meta

    qmodel.eval()
    previous = [l.act.pack_inference for l in layers]
    set_activation_packing(qmodel, True)
    qmodel(torch.zeros(batch_size, 3, 32, 32, device=device))
    for l, p in zip(layers, previous):
        l.act.pack_inference = p
    act_fp32 = sum(l.act.last_actual_output_bytes for l in layers)
    act_packed = sum(l.act.last_packed_bytes for l in layers)
    peak_fp32 = max(l.act.last_actual_output_bytes for l in layers)
    peak_packed = max(l.act.last_packed_bytes for l in layers)
    MB = 1024 ** 2
    return {
        'baseline_size_mb': base_total / MB,
        'model_size_mb': total / MB,
        'compression_ratio': base_total / total,
        'weights_fp32_mb': base_w / MB,
        'weights_payload_mb': payload / MB,
        'weights_ratio_payload_only': base_w / payload,
        'weights_ratio_with_overhead': base_w / (payload + scales + biases + meta),
        'overhead_scales_kb': scales / 1024,
        'overhead_bias_kb': biases / 1024,
        'overhead_act_params_kb': act_par / 1024,
        'overhead_metadata_estimate_kb': meta / 1024,
        'other_tensor_bytes': tensor_bytes(qmodel) - (payload + scales + biases + act_par),
        'activation_measurement_batch_size': batch_size,
        'act_fp32_mb': act_fp32 / MB,
        'act_quant_mb': act_packed / MB,
        'activation_ratio_total': act_fp32 / act_packed,
        'activation_ratio_peak': peak_fp32 / peak_packed,
        'activation_ratio_note': ('Sum over all Conv/Linear inputs of FP32 bytes / bytes of the real k-bit '
                                  'packed buffer, one forward pass. Storage-format ratio: compute uses FP32 '
                                  'kernels, so the forward peak is reported separately (measured).'),
        'model_size_note': 'Persistent parameter/buffer tensor bytes plus estimated metadata (calculated).',
    }


def save_frozen(model, path, config):
    """Save a reloadable bit-packed model (disk size is NOT compression metric)."""
    torch.save({'state_dict': model.state_dict(), 'config': config}, path)


def load_frozen(path, device):
    """Reconstruct the packed module layout before loading its state_dict."""
    from model import build_mobilenetv2
    blob = torch.load(path, map_location=device, weights_only=True)
    conf = blob['config']
    model = build_mobilenetv2(width_mult=conf['width_mult'], dropout=conf['dropout'],
                              bn_eps=conf.get('bn_eps', 1e-5), bn_momentum=conf.get('bn_momentum', 0.1)).to(device)
    convert(model, conf['w_bits'], conf['a_bits'], conf['first_last_bits'], conf['dw_bits'])
    freeze(model, pack_activations=conf.get('pack_activations', False))
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model
