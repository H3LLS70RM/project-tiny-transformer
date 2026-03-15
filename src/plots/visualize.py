import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import argparse

# Ordering for consistent visualization
SCALE_ORDER = ['tt-1k', 'tt-2k', 'tt-4k', 'tt-8k', 'tt-14k', 'tt-26k', 'tt-40k', 'tt-50k', 'tt-150k', 'tt-800k', 'tt-3000k']

def get_sorted_models(model_dict):
    sorted_models = [s for s in SCALE_ORDER if s in model_dict]
    sorted_models += [s for s in model_dict.keys() if s not in SCALE_ORDER]
    return sorted_models

def plot_training_progress(log_dir="results/logs", output_dir="results/plots"):
    """Plots Training Loss, Accuracy, and Learning Rate."""
    os.makedirs(output_dir, exist_ok=True)
    log_files = glob.glob(os.path.join(log_dir, "**", "training_log*.json"), recursive=True)
    if not log_files: return
    
    data = {}
    for log_file in log_files:
        path_parts = log_file.split(os.sep)
        if len(path_parts) >= 3:
            scale, task = path_parts[-2], path_parts[-3]
            if task not in data: data[task] = {}
            with open(log_file, 'r') as f:
                logs = json.load(f)
                if scale in data[task]: data[task][scale].extend(logs)
                else: data[task][scale] = logs
                data[task][scale].sort(key=lambda x: x['step'])

    for task, scales in data.items():
        sorted_scales = get_sorted_models(scales)
        
        # Loss Plot
        plt.figure(figsize=(10, 6))
        for scale in sorted_scales:
            logs = scales[scale]
            plt.plot([e['step'] for e in logs], [e['loss'] for e in logs], label=scale)
        plt.title(f"Training Loss - {task}"); plt.xlabel("Steps"); plt.ylabel("Loss")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{task}_training_loss.png")); plt.close()

        # Accuracy Plot
        plt.figure(figsize=(10, 6))
        for scale in sorted_scales:
            logs = scales[scale]
            
            # Train Accuracy
            acc_steps = [e['step'] for e in logs if 'accuracy' in e]
            acc_vals = [e['accuracy'] for e in logs if 'accuracy' in e]
            if acc_vals:
                p = plt.plot(acc_steps, acc_vals, label=f"{scale} (Train)")
                color = p[0].get_color()
                
                # Test Accuracy (Same color)
                test_acc_steps = [e['step'] for e in logs if 'test_accuracy' in e]
                test_acc_vals = [e['test_accuracy'] for e in logs if 'test_accuracy' in e]
                if test_acc_vals:
                    plt.plot(test_acc_steps, test_acc_vals, linestyle='--', color=color, label=f"{scale} (Test)")
        
        plt.title(f"Accuracy Progress - {task}"); plt.xlabel("Steps"); plt.ylabel("Accuracy")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{task}_training_accuracy.png"), bbox_inches='tight'); plt.close()

def plot_icl_results(results_path="results/evaluation/suite_*.json", output_dir="results/plots"):
    """Plots LCS scores and ICL Scaling data from the evaluation suite."""
    os.makedirs(output_dir, exist_ok=True)
    res_files = glob.glob(results_path)
    for res_file in res_files:
        task = re.search(r'suite_(.*).json', os.path.basename(res_file)).group(1)
        with open(res_file, 'r') as f: data = json.load(f)
        sorted_scales = get_sorted_models(data)

        # 1. Scaling Curve (Accuracy vs Shots)
        plt.figure(figsize=(8, 6))
        for scale in sorted_scales:
            res = data[scale]
            shots = [0, 1, 3, 5]
            accs = [res.get(f"{s}_shot_accuracy", 0) for s in shots]
            plt.plot(shots, accs, marker='o', label=scale)
        plt.title(f"ICL Scaling - {task.capitalize()}"); plt.xlabel("Shots"); plt.ylabel("Accuracy (%)")
        plt.legend(); plt.grid(True); plt.savefig(os.path.join(output_dir, f"icl_scaling_{task}.png")); plt.close()

        # 2. LCS Scores
        plt.figure(figsize=(10, 6))
        lcs_vals = [data[s].get('lcs_score', 0) for s in sorted_scales]
        bars = plt.bar(sorted_scales, lcs_vals, color='mediumpurple')
        plt.title(f"LCS Score - {task.capitalize()}"); plt.ylabel("LCS Score"); plt.xticks(rotation=45)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, h, f'{h:.2f}', ha='center', va='bottom')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"icl_lcs_{task}.png")); plt.close()

def plot_emergence_results(results_path="results/icl_emergence_results.json", output_dir="results/plots"):
    """Plots Flip Scores, LCS, and Induction scores together to show emergence."""
    if not os.path.exists(results_path):
        print(f"Emergence results not found at {results_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "r") as f:
        data = json.load(f)
        
    for task, scales in data.items():
        sorted_scales = get_sorted_models(scales)
        if not sorted_scales: continue
        
        # 1. Individual Task Plot: Flip Score vs Model Scale
        plt.figure(figsize=(10, 6))
        flip_scores = [scales[s].get('flip_score', 0) * 100 for s in sorted_scales]
        bars = plt.bar(sorted_scales, flip_scores, color='salmon')
        plt.title(f"ICL Flip Score (Context Following) - {task.capitalize()}")
        plt.ylabel("Flip Score Accuracy (%)")
        plt.xticks(rotation=45)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, h, f'{h:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"icl_emergence_{task}.png"))
        plt.close()

        # 2. Multi-metric Plot (Normalized comparison)
        plt.figure(figsize=(12, 7))
        x = np.arange(len(sorted_scales))
        width = 0.25
        
        flip_vals = [scales[s].get('flip_score', 0) for s in sorted_scales]
        lcs_vals = [scales[s].get('lcs_score', 0) for s in sorted_scales]
        # Normalize LCS if it's very large, but usually it's around 0-2
        ind_vals = [scales[s].get('max_induction_score', 0) for s in sorted_scales]
        
        plt.bar(x - width, flip_vals, width, label='Flip Score', color='salmon')
        plt.bar(x, lcs_vals, width, label='LCS Score', color='mediumpurple')
        plt.bar(x + width, ind_vals, width, label='Induction Score', color='skyblue')
        
        plt.title(f"ICL Emergence Metrics - {task.capitalize()}")
        plt.xticks(x, sorted_scales, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"icl_emergence_combined_{task}.png"))
        plt.close()

def plot_model_scaling_metrics(results_path="results/evaluation/suite_*.json", output_dir="results/plots"):
    """Plots Accuracy vs Model Scale and Edit Distance vs Model Scale."""
    os.makedirs(output_dir, exist_ok=True)
    res_files = glob.glob(results_path)
    
    # Helper to parse parameter count from scale string
    def parse_params(scale):
        match = re.search(r'tt-(\d+)([km]?)', scale)
        if not match: return 0
        val = int(match.group(1))
        unit = match.group(2)
        if unit == 'k': val *= 1000
        elif unit == 'm': val *= 1000000
        return val

    for res_file in res_files:
        task = re.search(r'suite_(.*).json', os.path.basename(res_file)).group(1)
        with open(res_file, 'r') as f: data = json.load(f)
        
        sorted_scales = get_sorted_models(data)
        if not sorted_scales: continue
        
        param_counts = [parse_params(s) for s in sorted_scales]
        
        # 1. Accuracy vs Model Scale (5-shot)
        plt.figure(figsize=(10, 6))
        acc_vals = [data[s].get('5_shot_accuracy', 0) for s in sorted_scales]
        plt.semilogx(param_counts, acc_vals, marker='o', linewidth=2, color='royalblue')
        plt.title(f"Model Scaling: Accuracy vs Parameters - {task.capitalize()}")
        plt.xlabel("Model Parameters (Log Scale)")
        plt.ylabel("5-Shot Accuracy (%)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        # Label points with scale name
        for i, s in enumerate(sorted_scales):
             plt.text(param_counts[i], acc_vals[i], s, fontsize=9, ha='right', va='bottom')
        plt.savefig(os.path.join(output_dir, f"scaling_accuracy_{task}.png"), bbox_inches='tight')
        plt.close()

        # 2. Token Edit Distance vs Model Scale (5-shot)
        plt.figure(figsize=(10, 6))
        # Only plot if we have edit distance data (check for key existence)
        valid_indices = [i for i, s in enumerate(sorted_scales) if '5_shot_edit_distance' in data[s]]
        if valid_indices:
            p_subset = [param_counts[i] for i in valid_indices]
            d_subset = [data[sorted_scales[i]]['5_shot_edit_distance'] for i in valid_indices]
            s_subset = [sorted_scales[i] for i in valid_indices]
            
            plt.semilogx(p_subset, d_subset, marker='s', linewidth=2, color='crimson')
            plt.title(f"Model Scaling: Edit Distance vs Parameters - {task.capitalize()}")
            plt.xlabel("Model Parameters (Log Scale)")
            plt.ylabel("Average Token Edit Distance (Lower is Better)")
            plt.grid(True, which="both", ls="-", alpha=0.5)
            for i, s in enumerate(s_subset):
                 plt.text(p_subset[i], d_subset[i], s, fontsize=9, ha='right', va='bottom')
            plt.savefig(os.path.join(output_dir, f"scaling_dist_{task}.png"), bbox_inches='tight')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize project results.")
    parser.add_argument("--logs", action="store_true", help="Plot training progress.")
    parser.add_argument("--icl", action="store_true", help="Plot ICL evaluation results.")
    parser.add_argument("--scaling", action="store_true", help="Plot model scaling metrics.")
    parser.add_argument("--all", action="store_true", help="Plot everything.")
    args = parser.parse_args()

    if args.logs or args.all:
        print("Plotting training progress...")
        plot_training_progress()
    if args.icl or args.all:
        print("Plotting ICL results...")
        plot_icl_results()
        print("Plotting emergence results...")
        plot_emergence_results()
    if args.scaling or args.all:
        print("Plotting model scaling metrics...")
        plot_model_scaling_metrics()

if __name__ == "__main__":
    main()
