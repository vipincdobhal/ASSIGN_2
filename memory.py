"""Memory measurement for the baseline and the compressed model (Q4).

Two measured quantities, each taken in isolation for one model:
  resident      bytes the model's tensors occupy once loaded on the device
  forward peak  extra bytes allocated during one forward pass at the measurement
                batch size (activations, dequantised weight chunks, temporaries)
The sum is the memory the model needs to run inference.

CUDA: read from the caching allocator (memory_allocated / max_memory_allocated),
      so 512-byte block rounding and cuDNN workspaces are included.
CPU:  PyTorch has no allocator peak counter, so a tensor-level tracker counts every
      storage created during the forward and its lifetime (kernel-internal scratch
      is not visible); resident = exact tensor bytes.
"""
import gc
import weakref
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

MB = 1024 ** 2


def tensor_bytes(model):
    """Calculated in-memory size: every parameter + buffer tensor."""
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


class PeakTracker(TorchDispatchMode):
    """Peak bytes of tensor storages created while active (CPU measurement)."""

    def __init__(self, preexisting=()):
        super().__init__()
        self.skip = {t.untyped_storage().data_ptr() for t in preexisting}
        self.refs, self.live, self.peak = {}, 0, 0

    def _release(self, ptr, nbytes):
        self.refs[ptr] -= 1
        if self.refs[ptr] == 0:
            del self.refs[ptr]
            self.live -= nbytes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if not isinstance(t, torch.Tensor):
                continue
            storage = t.untyped_storage()
            ptr, nbytes = storage.data_ptr(), storage.nbytes()
            if ptr in self.skip or nbytes == 0:
                continue
            if ptr not in self.refs:
                self.refs[ptr] = 0
                self.live += nbytes
                self.peak = max(self.peak, self.live)
            self.refs[ptr] += 1
            weakref.finalize(t, self._release, ptr, nbytes)
        return out


@torch.no_grad()
def measure_inference_memory(model, device, batch_size=1, repeats=3):
    """Measured resident size and forward peak of ONE model. Leaves it on `device`."""
    model.eval()
    x = torch.zeros(batch_size, 3, 32, 32)
    if device.type == "cuda":
        model.cpu()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        model.to(device)
        torch.cuda.synchronize(device)
        resident = torch.cuda.memory_allocated(device) - before
        x = x.to(device)
        model(x)                                            # warm-up (cuDNN setup)
        torch.cuda.synchronize(device)
        gc.collect(); torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
        for _ in range(repeats):
            model(x)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - start
        method = "CUDA caching allocator (memory_allocated / max_memory_allocated)"
    else:
        model.to(device)
        resident = tensor_bytes(model)
        model(x)                                            # warm-up
        pre = list(model.parameters()) + list(model.buffers()) + [x]
        with PeakTracker(pre) as tracker:
            model(x)
        peak = tracker.peak
        method = "CPU tensor-level tracker (TorchDispatchMode); resident = tensor bytes"
    return {"resident_mb": resident / MB, "forward_peak_mb": peak / MB,
            "inference_total_mb": (resident + peak) / MB,
            "memory_method": method, "memory_batch_size": batch_size}
