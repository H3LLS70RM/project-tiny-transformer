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
    
def get_task_vocab(task=None):
    """
    Returns a consistent stoi and itos for all tasks.
    A unified vocabulary ensures that models are robust to out-of-distribution
    characters (like symbols or digits) and that data is compatible across tasks.
    """
    # Unified alphabet: digits + uppercase letters + lowercase letters + operators + OOD symbols + separators
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()+-=>: \n"
    vocab = sorted(list(set(alphabet)))
    
    stoi = {ch: i+1 for i, ch in enumerate(vocab)}
    itos = {i+1: ch for i, ch in enumerate(vocab)}
    stoi['<pad>'] = 0
    itos[0] = '<pad>'
    
    return stoi, itos
