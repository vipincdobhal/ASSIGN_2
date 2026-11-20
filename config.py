"""Central configuration for the entire MobileNet-v2 assignment.
Edit values here; command-line arguments can still override run settings.
"""
# Dataset and reproducibility
DATA_ROOT = "./data"
SEED = 42
WORKERS = 4
BATCH_SIZE = 128
VALIDATION_FRACTION = 0.1
CALIBRATION_BATCHES = 20
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CROP_SIZE = 32
CROP_PADDING = 4
HORIZONTAL_FLIP_PROBABILITY = 0.5
RANDOM_ERASING_PROBABILITY = 0.25
TEST_BATCH_SIZE = 500

# Model
NUM_CLASSES = 10
WIDTH_MULT = 1.0
DROPOUT = 0.2
BN_EPS = 1e-5
BN_MOMENTUM = 0.1

# Baseline training
EPOCHS = 150
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
SGD_MOMENTUM = 0.9
NESTEROV = True
ONE_CYCLE_WARMUP_FRACTION = 0.1
LABEL_SMOOTHING = 0.1
DETERMINISTIC = False
CHECKPOINT = "mobilenetv2_cifar10.pt"
TRAIN_HISTORY = "train_history.json"
CLASS_ACCURACY = "class_accuracy.json"

# Compression
WEIGHT_BITS = 4
ACTIVATION_BITS = 8
FIRST_LAST_BITS = 8
DEPTHWISE_BITS = 8  # depthwise convs hold 2.9% of weights but are the most sensitive (None = use WEIGHT_BITS)
INPUT_BITS = 8
ACTIVATION_EMA_MOMENTUM = 0.9
QAT_EPOCHS = 5
QAT_LEARNING_RATE = 0.005
QAT_WEIGHT_DECAY = 1e-4
PACK_ACTIVATIONS = False  # packing is lossless; size_report() switches it on for its own measurement pass
WEIGHT_CHUNK_CODES = 16384  # max weights dequantised at once (output-channel chunks, multiple of 8 channels)
MEASUREMENT_BATCH_SIZE = 1
RUNTIME_REPEATS = 3
METADATA_BYTES_PER_LAYER = 16
RESULTS_DIRECTORY = "results"
FIGURES_DIRECTORY = "figures"
SAVE_COMPRESSED_MODEL = False

# W&B
WANDB_ENABLED = False
WANDB_PROJECT = "MBNET_VCD"
SWEEP_WEIGHT_BITS = [2, 3, 4, 6, 8]
SWEEP_ACTIVATION_BITS = [4, 6, 8]
SWEEP_QAT_EPOCHS = 5  # default for --sweep; override with --sweep_qat_epochs (report uses 0, 3 and 5)

# Baseline experiment selection (validation accuracy; test set is held out)
BASELINE_RUN_DIR = "baseline_runs"
BEST_BASELINE_CHECKPOINT = "best_baseline.pt"
BEST_BASELINE_CONFIG = "best_baseline_config.json"
BASELINE_RESULTS = "baseline_results.json"
BEST_TEST_CHECKPOINT = "best_test_baseline.pt"
BEST_TEST_CONFIG = "best_test_baseline_config.json"
BASELINE_SELECTION_METRIC = "best_val_acc"
BASELINE_RUNS = [
    dict(width=1.0, dropout=.2, bn=.10, bn_eps=1e-5, optimizer="SGD", schedule="OneCycle", lr=.10, wd=5e-4, epochs=100, batch=128),
    dict(width=.75, dropout=.2, bn=.10, bn_eps=1e-5, optimizer="SGD", schedule="Cosine", lr=.05, wd=5e-4, epochs=100, batch=128),
    dict(width=1.25, dropout=.2, bn=.10, bn_eps=1e-5, optimizer="SGD", schedule="OneCycle", lr=.10, wd=5e-4, epochs=150, batch=128),
    dict(width=1.0, dropout=.3, bn=.05, bn_eps=1e-3, optimizer="AdamW", schedule="Cosine", lr=.001, wd=1e-4, epochs=100, batch=128),
    dict(width=1.0, dropout=.1, bn=.10, bn_eps=1e-5, optimizer="SGD", schedule="OneCycle", lr=.08, wd=1e-4, epochs=150, batch=64),
    dict(width=.75, dropout=.3, bn=.05, bn_eps=1e-3, optimizer="AdamW", schedule="Cosine", lr=.001, wd=5e-4, epochs=150, batch=128),
    dict(width=1.25, dropout=.1, bn=.10, bn_eps=1e-5, optimizer="SGD", schedule="Cosine", lr=.05, wd=5e-4, epochs=150, batch=64),
    dict(width=1.0, dropout=.2, bn=.05, bn_eps=1e-3, optimizer="AdamW", schedule="OneCycle", lr=.002, wd=1e-4, epochs=100, batch=64),
    dict(width=1.25, dropout=.3, bn=.05, bn_eps=1e-3, optimizer="AdamW", schedule="Cosine", lr=.0005, wd=5e-4, epochs=150, batch=128),
    dict(width=.75, dropout=.1, bn=.10, bn_eps=1e-5, optimizer="SGD", schedule="OneCycle", lr=.10, wd=1e-4, epochs=150, batch=64),
    dict(width=1.0, dropout=.2, bn=.10, bn_eps=1e-5, optimizer="SGD", schedule="OneCycle", lr=.10, wd=5e-4, epochs=150, batch=128),
]
