import os
import sys
# Add project root to sys.path for robust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import re
import argparse
import glob
import json
from src.model.tiny_transformer import TinyTransformer
from src.configs.model_configs import config
from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.evaluation.metrics import calculate_lcs, probe_induction_heads
from src.evaluation.probes import evaluate_label_flipping, generate_analysis_summary
from src.utils import get_task_vocab

def run_emergence_evaluation(configs, tasks, step=None, results_dir="results"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {task: {} for task in tasks}
    
    for task in tasks:
        print(f"\n--- Evaluating Emergence: Task {task} ---")
        for model_scale in configs:
            print(f"  Model: {model_scale}")
            cfg = config(model_scale)
            
            # Setup vocab
            stoi, itos = get_task_vocab(task)
            
            model = TinyTransformer(
                vocab_size=len(stoi), dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'],
                stoi=stoi, itos=itos, configkey=model_scale, mlp_dim=cfg['mlp_dim'],
                max_len=cfg.get('max_len', 256), use_rope=True
            ).to(device)
            
            ckpt_dir = f"checkpoints/{task}/{model_scale}/"
            target_ckpt = None
            if step:
                target_ckpt = os.path.join(ckpt_dir, f"model-step-{step}.pt")
            else:
                options = [os.path.join(ckpt_dir, "model_best.pt"), os.path.join(ckpt_dir, "model_latest.pt")]
                for opt in options:
                    if os.path.exists(opt): target_ckpt = opt; break
                if not target_ckpt:
                    ckpts = glob.glob(f"{ckpt_dir}model-step-*.pt")
                    if ckpts: target_ckpt = max(ckpts, key=lambda x: int(re.search(r"model-step-(\d+)", x).group(1)))

            if not target_ckpt or not os.path.exists(target_ckpt):
                print(f"    No suitable checkpoint found in {ckpt_dir}, skipping.")
                continue
                
            print(f"    Loading {target_ckpt}")
            model.load_state_dict(torch.load(target_ckpt, map_location=device))
            
            flip_score, flip_preds = evaluate_label_flipping(model, task, device=device)
            lcs_score = calculate_lcs(model, task, device=device)
            induction_score = probe_induction_heads(model, task, device)
            
            all_results[task][model_scale] = {
                "flip_score": flip_score,
                "flip_predictions": flip_preds,
                "max_induction_score": induction_score,
                "lcs_score": lcs_score
            }

    # Save Results
    res_path = os.path.join(results_dir, "icl_emergence_results.json")
    with open(res_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate Analysis Summary
    generate_analysis_summary(all_results, results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=['tt-8k', 'tt-26k', 'tt-150k'])
    parser.add_argument("--tasks", nargs="+", default=['addition', 'arithmetic_symbolic', 'mapping', 'decoding'])
    parser.add_argument("--step", type=int, default=None, help="Evaluate a specific step.")
    args = parser.parse_args()
    
    run_emergence_evaluation(args.configs, args.tasks, step=args.step)
