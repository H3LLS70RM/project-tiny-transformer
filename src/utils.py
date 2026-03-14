import random
import numpy as np
import torch

def set_seed(seed=1337):
    """
    Sets the seed for random, numpy, and torch (CPU & CUDA) to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For strict reproducibility, benchmarking can be disabled and deterministic can be enabled
        # However, this might slow down training.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed}")
