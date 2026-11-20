"""Isolated CUDA inference-memory profiler for the EXISTING MobileNetV2 compression code.

Usage (run from project root, with existing modules on PYTHONPATH):
 python profile_isolated.py --baseline best_baseline.pt --compressed results/w4_a6_dw8_qat5_packed.pt
 python profile_isolated.py --baseline best_baseline.pt --compressed results/*_packed.pt --output results/isolated_memory.csv

Each model is loaded and measured in its own subprocess. No quantization code is changed.
Values are PyTorch CUDA allocated bytes, not whole-device VRAM or load-time peak.
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

MIB = 1024 ** 2

def measure_worker(kind, checkpoint, batch_size, warmup, repeats):
    import torch
    import compress
    from model import build_mobilenetv2
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for isolated CUDA memory measurements')
    device = torch.device('cuda:0')
    # Construct on CPU first. Only the selected model moves to GPU.
    if kind == 'baseline':
        import config as cfg
        selected_cfg = {}
        config_path = Path(getattr(cfg, 'BEST_BASELINE_CONFIG', 'best_baseline_config.json'))
        if config_path.exists():
            obj = json.loads(config_path.read_text())
            selected_cfg = obj.get('config', obj)
        model = build_mobilenetv2(
            width_mult=selected_cfg.get('width_mult', getattr(cfg, 'WIDTH_MULT', 1.0)),
            dropout=selected_cfg.get('dropout', getattr(cfg, 'DROPOUT', 0.2)),
            bn_eps=selected_cfg.get('bn_eps', getattr(cfg, 'BN_EPS', 1e-5)),
            bn_momentum=selected_cfg.get('bn_momentum', getattr(cfg, 'BN_MOMENTUM', 0.1)))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    else:
        model = compress.load_frozen(checkpoint, torch.device('cpu'))
    model = model.to(device).eval()
    x = torch.zeros((batch_size, 3, 32, 32), device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            result = model(x)
            del result
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        end = torch.cuda.memory_allocated(device)
    # Report unique parameter/buffer storages, avoiding shared-storage double count.
    seen = set()
    resident = 0
    for t in list(model.parameters()) + list(model.buffers()):
        storage = t.untyped_storage()
        key = (str(t.device), storage.data_ptr())
        if key not in seen:
            resident += storage.nbytes()
            seen.add(key)
    return dict(kind=kind, checkpoint=str(Path(checkpoint).resolve()),
                gpu_name=torch.cuda.get_device_name(device),
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                batch_size=batch_size, warmup_passes=warmup, measured_passes=repeats,
                model_resident_tensor_bytes=resident,
                start_allocated_bytes=start, peak_allocated_bytes=peak,
                incremental_forward_peak_bytes=max(0, peak-start),
                end_allocated_bytes=end,
                start_allocated_mib=start/MIB, peak_allocated_mib=peak/MIB,
                incremental_forward_peak_mib=max(0, peak-start)/MIB,
                model_resident_tensor_mib=resident/MIB,
                note='PyTorch CUDA allocated memory; excludes driver/other processes and model-loading peak')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=False)
    p.add_argument('--compressed', nargs='*', default=[])
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output', default='results/isolated_memory.csv')
    p.add_argument('--worker-kind', choices=('baseline','compressed'), help=argparse.SUPPRESS)
    p.add_argument('--worker-checkpoint', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_kind:
        print('ISOLATED_MEMORY_JSON=' + json.dumps(measure_worker(args.worker_kind, args.worker_checkpoint,
                                args.batch_size, args.warmup, args.repeats)), flush=True)
        return
    if not args.baseline or not args.compressed:
        p.error('Both --baseline and at least one --compressed checkpoint are required')
    if args.batch_size < 1 or args.warmup < 1 or args.repeats < 1:
        p.error('batch-size, warmup, and repeats must be positive')
    rows = []
    for kind, checkpoint in [('baseline', args.baseline)] + [('compressed', path) for path in args.compressed]:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker-kind', kind,
               '--worker-checkpoint', str(Path(checkpoint).resolve()),
               '--batch-size', str(args.batch_size), '--warmup', str(args.warmup),
               '--repeats', str(args.repeats)]
        completed = subprocess.run(cmd, text=True, capture_output=True, cwd=os.getcwd())
        if completed.returncode:
            raise RuntimeError(f'Profiling failed for {checkpoint}:\n{completed.stderr[-4000:]}')
        matches = [line.split('=',1)[1] for line in completed.stdout.splitlines()
                   if line.startswith('ISOLATED_MEMORY_JSON=')]
        if not matches:
            raise RuntimeError(f'No profile result for {checkpoint}:\n{completed.stdout[-2000:]}')
        row = json.loads(matches[-1])
        rows.append(row)
        print(f'{kind}: {Path(checkpoint).name} | resident={row["model_resident_tensor_mib"]:.3f} MiB | '
              f'peak={row["peak_allocated_mib"]:.3f} MiB | forward={row["incremental_forward_peak_mib"]:.3f} MiB')
    base = rows[0]
    for row in rows:
        for key, numerator, denominator in [
            ('isolated_peak_ratio_vs_fp32', base['peak_allocated_bytes'], row['peak_allocated_bytes']),
            ('isolated_resident_ratio_vs_fp32', base['model_resident_tensor_bytes'], row['model_resident_tensor_bytes']),
            ('isolated_forward_increment_ratio_vs_fp32', base['incremental_forward_peak_bytes'], row['incremental_forward_peak_bytes'])]:
            row[key] = numerator/denominator if denominator else None
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    destination.with_suffix('.json').write_text(json.dumps(rows, indent=2))
    print('Saved:', destination, 'and', destination.with_suffix('.json'))

if __name__ == '__main__':
    main()
