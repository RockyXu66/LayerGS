#!/usr/bin/env python3
#
# LayerGS Unified Training Script
#

import sys
import argparse
import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 0) -> None:
    """Set seeds for python, numpy, torch (cpu & cuda) and make cudnn deterministic.
    
    Also resets module-level random generators in the codebase for full reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed when torch.use_deterministic_algorithms(True) with CuBLAS on CUDA>=10.2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # from layergs.data.humangaussian_dataset import reset_random_generators as reset_hg_generators
    # from layergs.guidance.gala_sds_controlnet_utils import reset_random_generators as reset_gala_generators
    # reset_hg_generators(seed)
    # reset_gala_generators(seed)


def main():
    # set_global_seed(0)

    # Create top-level parser
    parser = argparse.ArgumentParser(
        description="LayerGS Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "stage",
        choices=["singlelayer", "inner", "outer"],
        help="Training stage to run"
    )
    
    # Parse just the stage argument
    args, remaining = parser.parse_known_args()
    
    # Import and run the appropriate training module
    if args.stage == "singlelayer":
        from layergs.train.singlelayer import main as train_main
    elif args.stage == "inner":
        from layergs.train.inner import main as train_main
    elif args.stage == "outer":
        from layergs.train.outer import main as train_main
    
    # Replace sys.argv with remaining arguments for the training script
    sys.argv = [sys.argv[0]] + remaining
    train_main()


if __name__ == "__main__":
    main()

