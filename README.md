# MobileNet-v2 on CIFAR-10 with Custom Quantization-Based Compression

CS6886W – Systems Engineering for Deep Learning, Assignment 2 (IIT Madras).

This repository trains MobileNet-v2 from scratch on CIFAR-10 and compresses it with a
hand-written quantization pipeline: BatchNorm folding, bit-packed k-bit weights, k-bit
activations and quantization-aware training (QAT). **No compression or quantization APIs
are used.** Every compression step is built from elementary tensor operations in `compress.py`.

Repository: `<https://github.com/vipincdobhal/ASSIGN_2>`

---

## Results at a glance

| | FP32 baseline | Final compressed model (w4 a6, 5 QAT epochs) |
|---|---|---|
| Test top-1 (CIFAR-10, 10,000 images) | **94.60%** | **94.40%** (−0.20 pp) |
| Model size in memory | 8.663 MiB | **1.154 MiB** (1,209,813 bytes) |
| Model compression ratio | 1× | **7.51×** |
| Weight compression ratio (incl. scales, biases, metadata) | 1× | **7.29×** (7.73× codes only) |
| Activation compression ratio | 1× | **5.33×** |
| Storage overhead | – | 67.96 KB = 5.75% of the model |
| Model tensors on the GPU (measured) | 8.663 MiB | 1.153 MiB (7.51×) |
| Peak GPU memory during inference, batch 1 (measured) | 20.909 MiB | 13.541 MiB (1.54×) |

All sizes are **in-memory** (RAM/VRAM) tensor sizes, not file sizes. MiB = 1,048,576 bytes;
the code's `*_mb` fields are MiB. The full tables are in `RESULTS.md` and the report.

---

## 1. Environment and dependency versions

Results were produced with:

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 5000 Ada Generation Laptop GPU |
| CUDA | 13.0 |
| Python | `<ADD YOUR PYTHON VERSION HERE>` |
| torch | 2.14.1 (+cu130) |
| torchvision | 0.29.1 |
| numpy | 2.5.4 |
| wandb | 0.31.0 |
| matplotlib | 3.11.2 (figures only) |

`requirements.txt` pins these versions.

```bash
git clone https://github.com/vipincdobhal/ASSIGN_2
cd ASSIGN_2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
wandb login                     # needed for --wandb runs and the sweeps
```

CIFAR-10 is downloaded automatically into `./data` on first use.

---

## 2. Seed configuration and reproducibility

| Setting | Value | Where |
|---|---|---|
| Global seed | `SEED = 42` | `config.py` (override with `--seed N`) |
| Seeded generators | Python `random`, NumPy, `torch`, all CUDA devices | `engine.set_seed()` |
| Train / validation split | 45,000 / 5,000, from `numpy.random.default_rng(42).permutation` | `data.get_loaders()` |
| Deterministic kernels | off by default (`DETERMINISTIC = False`) | `config.py` / `--deterministic` |

- The test set is never used for model selection. The baseline is chosen by validation accuracy.
- Normalization statistics, calibration and QAT use only training images.
- For bit-exact reruns, add `--deterministic` to `train.py`. It sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True` and `cudnn.benchmark=False`,
  and calls `torch.use_deterministic_algorithms(True, warn_only=True)`. Training is slower in
  this mode. Without it, reruns can differ by a few hundredths of a point.

---

## 3. Implementation overview

### Data (`data.py`)

- Per-channel mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616) are computed
  from the **training images only**.
- Train transforms: `RandomCrop(32, padding=4)` → `RandomHorizontalFlip(0.5)` → `ToTensor` → `Normalize` → `RandomErasing(0.25)`.
- Validation, test and calibration transforms: `ToTensor` → `Normalize`.

### Model (`model.py`)

- MobileNet-v2 is written from scratch, without torchvision model code.
- CIFAR changes: the stem stride and the second-stage stride are both 1, so 32×32 inputs
  are downsampled 32 → 16 → 8 → 4.
- Options: width multiplier, dropout before the classifier, and BN eps/momentum.

### Training (`train.py`, `engine.py`)

- Optimizer: SGD (Nesterov, momentum 0.9) or AdamW. With OneCycle, PyTorch's default
  `cycle_momentum=True` cycles the momentum between 0.85 and 0.95.
- LR schedule: OneCycle (10% warm-up) or Cosine, stepped every iteration.
- Label smoothing 0.1 and mixed precision (AMP) on GPU.
- The best checkpoint by validation accuracy is kept. The test set is evaluated once, on that checkpoint.
- The 11 baseline configurations are in `BASELINE_RUNS` in `config.py`. **Run 05 was selected**:
  width 1.0, dropout 0.1, SGD + OneCycle, max LR 0.08, weight decay 1e-4, batch 64, 150 epochs
  (validation 94.82%, test 94.60%).

### Compression (`compress.py`)

1. **BN folding:** each Conv→BN pair is merged into a single conv with a bias.
2. **Weights:** per-output-channel symmetric k-bit quantization, stored **bit-packed** in a
   `uint8` buffer (any width from 2 to 8 bits; N codes take ⌈N·k/8⌉ bytes).
3. **Activations:** the input of every Conv/Linear is quantized per tensor, asymmetrically,
   to a bits. The range comes from an EMA of the min/max during calibration.
4. **Calibration** on 20 training batches, then optional **QAT** with a straight-through estimator.
5. **Freeze:** the float weights are deleted. Each layer keeps only packed codes, FP16 scales,
   an FP16 bias and its activation range, and dequantizes its weights in output-channel chunks
   of at most 16,384 codes just before use.

Per-layer precision:

| Layer group | Weight bits | Input activation bits |
|---|---|---|
| Stem conv (`features.0.0`) | `--first_last_bits` (8) | 8 (input image) |
| 17 depthwise 3×3 convs | `--dw_bits` (default **8**) | `--a_bits` |
| 34 pointwise 1×1 convs | `--w_bits` | `--a_bits` |
| Classifier | `--first_last_bits` (8) | `--first_last_bits` (8) |

Residual additions, ReLU6 and global pooling stay in FP32.

### Measurement (`compress.size_report`, `memory.py`, `profile_isolated.py`)

- **Model size:** bytes (numel × element size) of every parameter and buffer, plus an
  estimate of 16 bytes of metadata per layer.
- **Activation ratio:** one forward pass of one image with activation packing switched on.
  For each of the 53 conv/linear inputs, the FP32 bytes are compared with the bytes of the
  real bit-packed buffer. This is a storage-format ratio: computation still uses FP32
  kernels, so the forward-pass peak is measured separately.
- **GPU memory:** `torch.cuda.memory_allocated` / `max_memory_allocated`. `profile_isolated.py`
  measures each model in its own process.

---

## 4. Reproducing every result

### Step 1: train the baselines (Q1)

```bash
python baseline_sweep.py --all --wandb          # all 11 configs; completed runs are skipped (resumable)
python baseline_sweep.py --run 5 --wandb        # only run 5 (1..11)
python baseline_sweep.py --run 5 --wandb --force  # repeat a completed run
```

### Step 2: select the best baseline by validation accuracy

```bash
python select_best_model.py                     # requires all 11 runs
python select_best_model.py --allow-partial     # preview using only finished runs
```

This writes `best_baseline.pt` and `best_baseline_config.json`, which every compression run
uses. It also writes `best_test_baseline.*` (for diagnostics only) and `baseline_results.json`.

### Step 3: compression sweeps (Q3)

Three W&B grid sweeps of 15 runs each (weight bits {2, 3, 4, 6, 8} × activation bits {4, 6, 8}),
with 0, 3 and 5 QAT epochs: 45 runs in total.

```bash
python run_compress.py --sweep --sweep_qat_epochs 0 --out_dir results/qat0
python run_compress.py --sweep --sweep_qat_epochs 3 --out_dir results/qat3
python run_compress.py --sweep --sweep_qat_epochs 5 --out_dir results/qat5
```

### Step 4: the final model (Q4)

```bash
python run_compress.py --w_bits 4 --a_bits 6 --qat_epochs 5 --save_model --wandb
python profile_isolated.py --baseline best_baseline.pt --compressed results/w4_a6_dw8_qat5_packed.pt
```

The first command saves the packed model to `results/w4_a6_dw8_qat5_packed.pt` and checks
that it reloads with identical accuracy. The second writes `results/isolated_memory.{csv,json}`.

The template repository's command line also works:

```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8
python test.py --weight_quant_bits 4 --activation_quant_bits 6 --qat_epochs 5
```

### Step 5: tables and figures

```bash
python summarize_results.py     # RESULTS.md: all baseline runs and all compression runs
python plot_baseline.py         # figures/best_run_curves.png, figures/all_runs_val_acc.png
python plot_compression.py      # figures/compression_tradeoff.png, figures/parallel_coordinates.png
```

### Wandb Parallel Coordinates chart

Open a sweep in W&B (project `MBNET_VCD`) → **Add panel → Parallel coordinates**, then choose
the axes `activation_quant_bits`, `weight_quant_bits`, `compression_ratio`, `model_size_mb`
and `quantized_acc`.

### `run_compress.py` arguments

| Argument | Default | Meaning |
|---|---|---|
| `--ckpt` | `best_baseline.pt` | Baseline checkpoint. Width, dropout, BN settings, seed and batch size are read from `best_baseline_config.json` |
| `--w_bits` / `--weight_quant_bits` | 4 | Weight bits for the pointwise convs (2–8) |
| `--a_bits` / `--activation_quant_bits` | 8 | Activation bits for the inner layers (2–8) |
| `--dw_bits` | 8 | Weight bits for the depthwise convs |
| `--first_last_bits` | 8 | Weight and activation bits for the stem conv and the classifier |
| `--qat_epochs` | 5 | QAT epochs (0 = calibration only) |
| `--calib_batches` | 20 | Calibration batches |
| `--measure_batch_size` | 1 | Batch size for activation and memory measurement |
| `--no_pack_activations` | off | Disable packed activation buffers at inference (measurement always packs) |
| `--save_model` | off | Save the frozen packed model and check that it reloads with identical accuracy |
| `--out_dir` | `results` | Output folder |
| `--wandb` | off | Log a single run to W&B (project `MBNET_VCD`) |
| `--sweep` | off | Run the W&B grid sweep instead of a single run |
| `--sweep_qat_epochs` | 5 | QAT epochs for every run of `--sweep` |

---

## 5. Outputs included in this repository

| Path | Contents |
|---|---|
| `baseline_runs/run_NN_history.json` | Per-epoch losses and accuracies, config, best epoch and test accuracy of each of the 11 runs |
| `baseline_runs/run_NN_class_accuracy.json` | Per-class test accuracy |
| `results/qat0/`, `results/qat3/`, `results/qat5/` | The three sweeps: one JSON report per run (with per-layer precision) and `results.csv` |
| `results/w4_a6_dw8_qat5.json`, `results/results.csv` | The final model (and the w4 a8 model saved earlier), with `reload_verified = true` |
| `results/isolated_memory.{csv,json}` | GPU memory of the baseline and the final model, each in its own process |
| `RESULTS.md` | All measured accuracies, sizes and compression ratios (from `summarize_results.py`) |
| `figures/` | Training curves, accuracy-vs-compression chart, parallel-coordinates chart |

Checkpoints (`*.pt`) are not committed (see `.gitignore`); the commands above regenerate them.

---

## 6. Repository layout

| File | Purpose |
|---|---|
| `config.py` | All defaults, the 11 baseline configurations and the sweep grid |
| `data.py` | CIFAR-10 loaders, training-set statistics, seeded split, calibration loader |
| `model.py` | MobileNet-v2 for CIFAR-10, written from scratch |
| `engine.py` | Seeding, device selection, train and evaluate loops |
| `train.py` | Trains one baseline |
| `baseline_sweep.py` | Runs baseline configurations 1–11 |
| `select_best_model.py` | Selects the best model by validation accuracy |
| `compress.py` | The compression method: packing, quantizers, BN folding, calibration, QAT, freezing, size report, save/load |
| `memory.py` | In-memory size and forward-peak measurement |
| `run_compress.py` | Single compression experiment, or a W&B sweep with `--sweep` |
| `test.py` | Same as `run_compress.py`, with the template repository's command line |
| `profile_isolated.py` | GPU memory of each model in its own process |
| `summarize_results.py` | Collects all results into `RESULTS.md` |
| `plot_baseline.py`, `plot_compression.py` | Report figures |
| `requirements.txt` | Pinned dependencies |
