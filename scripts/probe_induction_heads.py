import os
import sys
# Add project root to sys.path for robust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
import glob
import re
from src.model.tiny_transformer import TinyTransformer
from src.configs.model_configs import config
from src.evaluation.metrics import probe_induction_heads # import scoring function
from src.utils import get_task_vocab

from src.plots.visualize import plot_induction_heads

# No need for duplicate visualize_induction_heads logic here anymore

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Auto-discover tasks and scales from checkpoints directory
    if args.tasks:
        tasks = args.tasks
    elif os.path.exists("checkpoints"):
        tasks = [d for d in os.listdir("checkpoints") if os.path.isdir(os.path.join("checkpoints", d))]
    else:
        tasks = ['decoding', 'addition', 'mapping']
        
    for task in tasks:
        task_dir = os.path.join("checkpoints", task)
        if args.configs:
            scales = args.configs
        else:
            scales = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        # Sort scales numerically if possible
        def sort_key(s):
            m = re.search(r'(\d+)([km]?)', s)
            if not m: return 0
            v = int(m.group(1))
            if m.group(2) == 'k': v *= 1000
            elif m.group(2) == 'm': v *= 1000000
            return v
        scales.sort(key=sort_key)
        
        for scale in scales:
            try:
                cfg = config(scale)
                stoi, itos = get_task_vocab(task)
                
                model = TinyTransformer(
                    vocab_size=len(stoi), dim=cfg['dim'], depth=cfg['depth'], 
                    n_heads=cfg['n_heads'], stoi=stoi, itos=itos, 
                    configkey=scale, mlp_dim=cfg['mlp_dim'],
                    max_len=cfg.get('max_len', 256), use_rope=True
                ).to(device)
                
                ckpt_dir = f"checkpoints/{task}/{scale}/"
                options = [os.path.join(ckpt_dir, "model_best_icl.pt"), os.path.join(ckpt_dir, "model_best.pt"), os.path.join(ckpt_dir, "model_latest.pt")]
                target_ckpt = None
                for opt in options:
                    if os.path.exists(opt): target_ckpt = opt; break
                
                if not target_ckpt:
                    ckpts = glob.glob(f"{ckpt_dir}model-step-*.pt")
                    if ckpts: target_ckpt = max(ckpts, key=lambda x: int(re.search(r"model-step-(\d+)", x).group(1)))
                
                if target_ckpt:
                    print(f"Loading {target_ckpt}")
                    model.load_state_dict(torch.load(target_ckpt, map_location=device))
                    plot_induction_heads(model, scale, task, device)
                else:
                    print(f"No checkpoint for {scale} on {task}, skipping.")
            except Exception as e:
                print(f"Error processing {scale} on {task}: {e}")
