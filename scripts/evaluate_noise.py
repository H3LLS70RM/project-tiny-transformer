import os
import json
import sys
# Add project root to sys.path for robust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import re
import argparse
import matplotlib.pyplot as plt
import glob

from src.configs.model_configs import config
from src.model.tiny_transformer import TinyTransformer
from src.evaluation.probes import evaluate_noise_robustness
from src.utils import get_task_vocab

def plot_noise_robustness(task_results, task):
    """Plot noise robustness results."""
    plt.figure(figsize=(10, 6))
    
    for configkey, results in task_results.items():
        # Results keys may be strings like 'noise_0.2_accuracy' (from probes).
        pairs = []
        for k, v in results.items():
            try:
                import re
                m = re.search(r"noise_([0-9.]+)", k)
                if m:
                    r = float(m.group(1))
                else:
                    r = float(k)
            except Exception:
                # Fallback: try direct float conversion
                try:
                    r = float(k)
                except Exception:
                    continue
            pairs.append((r, float(v)))

        pairs.sort(key=lambda x: x[0])
        if not pairs:
            continue
        ratios, accuracies = zip(*pairs)
        plt.plot(ratios, accuracies, marker='o', linewidth=2, label=configkey)
    
    plt.title(f"ICL Noise Robustness: {task.capitalize()}", fontsize=14)
    plt.xlabel("Noise Ratio (Proportion of context examples with wrong answers)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Model Scale", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    task_dir = f"results/plots/{task}"
    os.makedirs(task_dir, exist_ok=True)
    plt.savefig(f"{task_dir}/noise_robustness.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {task_dir}/noise_robustness.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, nargs="+", default=['tt-8k', 'tt-26k', 'tt-150k'])
    parser.add_argument("--tasks", type=str, nargs="+", default=['addition'])
    parser.add_argument("--step", type=int, default=None, help="Specific step to evaluate.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--digit_level", action="store_true", help="Use digit-level formatting for addition (spaces between digits).")
    parser.add_argument("--rule_diversity", action="store_true", help="Enable rule diversity for mapping tasks.")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for task in args.tasks:
        all_results = {}
        for configkey in args.configs:
            print(f"\nEvaluating {configkey} on {task}...")
            model_cfg = config(configkey)
            
            # Setup vocab
            stoi, itos = get_task_vocab(task)
            
            model = TinyTransformer(
                vocab_size=len(stoi), dim=model_cfg['dim'], depth=model_cfg['depth'], n_heads=model_cfg['n_heads'],
                stoi=stoi, itos=itos, configkey=configkey, mlp_dim=model_cfg['mlp_dim'],
                max_len=model_cfg.get('max_len', 256), use_rope=True
            ).to(device)
            
            ckpt_dir = f"checkpoints/{task}/{configkey}/"
            target_ckpt = None
            if args.step:
                target_ckpt = os.path.join(ckpt_dir, f"model-step-{args.step}.pt")
            else:
                options = [os.path.join(ckpt_dir, "model_best.pt"), os.path.join(ckpt_dir, "model_latest.pt")]
                for opt in options:
                    if os.path.exists(opt): target_ckpt = opt; break
                if not target_ckpt:
                    ckpts = glob.glob(f"{ckpt_dir}model-step-*.pt")
                    if ckpts: target_ckpt = max(ckpts, key=lambda x: int(re.search(r"model-step-(\d+)", x).group(1)))

            if not target_ckpt or not os.path.exists(target_ckpt):
                print(f"Skipping: No checkpoint found in {ckpt_dir}"); continue
                
            print(f"Loading checkpoint: {target_ckpt}")
            model.load_state_dict(torch.load(target_ckpt, map_location=device))
            
            raw_results = evaluate_noise_robustness(
                model, task, noise_ratios=noise_ratios, n_samples=args.samples,
                n_context=5, device=device, digit_level=args.digit_level, rule_diversity=args.rule_diversity
            )
            print(f"Raw noise results for {configkey}: {raw_results}")
            all_results[configkey] = raw_results
            
        if all_results:
            save_path = f"results/evaluation/noise_robustness_{task}.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f: json.dump(all_results, f, indent=2)
            plot_noise_robustness(all_results, task)