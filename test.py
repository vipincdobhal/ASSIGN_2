"""Template-compatible entry point (same command line as the template repository):

    python test.py --weight_quant_bits 8 --activation_quant_bits 8
    python test.py --weight_quant_bits 4 --activation_quant_bits 6 --qat_epochs 5 --save_model

Runs one compression experiment on best_baseline.pt (MobileNet-v2) through run_compress.py.
Every other run_compress.py option (--dw_bits, --qat_epochs, --wandb, ...) is accepted too.
"""
from run_compress import main

if __name__ == "__main__":
    main()
