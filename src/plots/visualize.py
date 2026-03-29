import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import glob
import re
import argparse

# Ordering for consistent visualization
SCALE_ORDER = ['tt-2k', 'tt-5k', 'tt-9k', 'tt-14k', 'tt-26k', 'tt-49k', 'tt-141k', 'tt-808k', 'tt-3M']

def parse_params(scale):
    match = re.search(r'tt-(\d+)([kKmM]?)', scale)
    if not match: return 0
    val = int(match.group(1))
    unit = match.group(2).lower()
    if unit == 'k': val *= 1000
    elif unit == 'm': val *= 1000000
    return val

def get_sorted_models(model_dict):
    return sorted(list(model_dict.keys()), key=parse_params)

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
        
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        
        # Loss Plot
        plt.figure(figsize=(10, 6))
        for scale in sorted_scales:
            logs = scales[scale]
            plt.plot([e['step'] for e in logs], [e['loss'] for e in logs], label=scale)
        plt.title(f"Training Loss - {task}"); plt.xlabel("Steps"); plt.ylabel("Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True)
        plt.savefig(os.path.join(task_dir, f"training_loss.png"), bbox_inches='tight'); plt.close()

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
        plt.savefig(os.path.join(task_dir, f"training_accuracy.png"), bbox_inches='tight'); plt.close()

def plot_icl_results(results_path="results/evaluation/suite_*.json", output_dir="results/plots"):
    """Plots LCS scores and ICL Scaling data from the evaluation suite."""
    os.makedirs(output_dir, exist_ok=True)
    res_files = glob.glob(results_path)
    for res_file in res_files:
        task = re.search(r'suite_(.*).json', os.path.basename(res_file)).group(1)
        with open(res_file, 'r') as f: data = json.load(f)
        sorted_scales = get_sorted_models(data)
        
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        # 1. Scaling Curve (Accuracy vs Shots)
        plt.figure(figsize=(8, 6))
        for scale in sorted_scales:
            res = data[scale]
            shots = [0, 1, 2, 3, 4, 5]
            accs = [res.get(f"{s}_shot_accuracy", 0) for s in shots]
            plt.plot(shots, accs, marker='o', label=scale)
        plt.title(f"ICL Scaling - {task.capitalize()}"); plt.xlabel("Shots"); plt.ylabel("Accuracy (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True)
        plt.savefig(os.path.join(task_dir, f"icl_scaling.png"), bbox_inches='tight'); plt.close()

        # 2. LCS Scores
        plt.figure(figsize=(10, 6))
        lcs_vals = [data[s].get('lcs_score', 0) for s in sorted_scales]
        bars = plt.bar(sorted_scales, lcs_vals, color='mediumpurple')
        plt.title(f"LCS Score - {task.capitalize()}"); plt.ylabel("LCS Score"); plt.xticks(rotation=45)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, h, f'{h:.2f}', ha='center', va='bottom')
        plt.tight_layout(); plt.savefig(os.path.join(task_dir, f"icl_lcs.png")); plt.close()

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
        
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        
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
        plt.savefig(os.path.join(task_dir, f"icl_emergence.png"))
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, f"icl_emergence_combined.png"), bbox_inches='tight')
        plt.close()

def plot_model_scaling_metrics(results_path="results/evaluation/suite_*.json", output_dir="results/plots"):
    """Plots Accuracy vs Model Scale and Edit Distance vs Model Scale."""
    os.makedirs(output_dir, exist_ok=True)
    res_files = glob.glob(results_path)

    for res_file in res_files:
        task = re.search(r'suite_(.*).json', os.path.basename(res_file)).group(1)
        with open(res_file, 'r') as f: data = json.load(f)
        
        # Determine the task from the data (top level keys are models)
        # Exclude experimental / non-standard configs from scaling plots
        EXCLUDED_CONFIGS = {'tt-50k-balanced'}
        sorted_scales = sorted([s for s in data.keys() if s not in EXCLUDED_CONFIGS], key=parse_params)
        # Log-scale plots cannot represent zero/unknown parameter counts.
        sorted_scales = [s for s in sorted_scales if parse_params(s) > 0]
        if not sorted_scales: continue
        
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        
        param_counts = [parse_params(s) for s in sorted_scales]
        
        # 1. Accuracy vs Model Scale (ICL/5-shot)
        plt.figure(figsize=(10, 6))
        # Try new 'exact_match_accuracy', then old '5_shot_accuracy'
        acc_vals = [data[s].get('exact_match_accuracy', data[s].get('5_shot_accuracy', 0)) for s in sorted_scales]
        plt.semilogx(param_counts, acc_vals, marker='o', linewidth=2, color='royalblue', label=f'ICL Accuracy')
        
        # Also plot Token Accuracy if available
        if 'token_accuracy' in data[sorted_scales[0]]:
            tok_acc_vals = [data[s].get('token_accuracy', 0) for s in sorted_scales]
            plt.semilogx(param_counts, tok_acc_vals, marker='x', linestyle='--', linewidth=1.5, color='darkblue', alpha=0.6, label='Token Accuracy')
            
        plt.title(f"Model Scaling: Accuracy vs Parameters - {task.capitalize()}")
        plt.xlabel("Model Parameters (Log Scale)")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # Label points with scale name
        for i, s in enumerate(sorted_scales):
             plt.text(param_counts[i], acc_vals[i], s, fontsize=9, ha='right', va='bottom')
        plt.savefig(os.path.join(task_dir, f"scaling_accuracy.png"), bbox_inches='tight')
        plt.close()

        # 2. Token Edit Distance vs Model Scale
        plt.figure(figsize=(10, 6))
        
        dist_key = 'avg_edit_distance' if 'avg_edit_distance' in data[sorted_scales[0]] else '5_shot_edit_distance'
        
        valid_indices = [i for i, s in enumerate(sorted_scales) if dist_key in data[s]]
        if valid_indices:
            p_subset = [param_counts[i] for i in valid_indices]
            d_subset = [data[sorted_scales[i]][dist_key] for i in valid_indices]
            s_subset = [sorted_scales[i] for i in valid_indices]
            
            plt.semilogx(p_subset, d_subset, marker='s', linewidth=2, color='crimson')
            plt.title(f"Model Scaling: Token Distance vs Parameters - {task.capitalize()}")
            plt.xlabel("Model Parameters (Log Scale)")
            plt.ylabel("Avg Token Distance (Lower is Better)")
            plt.grid(True, which="both", ls="-", alpha=0.5)
            for i, s in enumerate(s_subset):
                 plt.text(p_subset[i], d_subset[i], s, fontsize=9, ha='right', va='bottom')
            plt.savefig(os.path.join(task_dir, f"scaling_dist.png"), bbox_inches='tight')
            plt.close()

def visualize_head_attention(model, layer, head, tokens, configkey, task, target_pos=None, output_dir="results/plots/induction_heads"):
    """
    Saves a 2D attention matrix plot with highlighting for the expected 
    induction target and the actual maximum attention.
    """
    import torch
    import seaborn as sns
    from matplotlib.patches import Rectangle
    
    model.eval()
    stoi, itos = model.stoi, model.itos
    # Clean character labels (requested by user to remove position index)
    token_labels = [itos.get(t, '?') for t in tokens]
    
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(next(model.parameters()).device)
    with torch.no_grad():
        model(input_tensor)
        attentions = model.get_attention_maps()
    
    # Layer attention is [B, H, T, T]
    matrix = attentions[layer][0, head].cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(matrix, annot=False, cmap="Blues", xticklabels=token_labels, yticklabels=token_labels)
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Highlight the ACTUAL maximum attention in the last row (the query token)
    query_idx = matrix.shape[0] - 1
    # Only consider key positions before the query
    visible_row = matrix[query_idx, :query_idx]
    if len(visible_row) > 0:
        actual_max_pos = np.argmax(visible_row)
        # Red rectangle for actual max
        rect_max = Rectangle((actual_max_pos, query_idx), 1, 1, fill=False, edgecolor='red', lw=3, label="Actual Max")
        ax.add_patch(rect_max)
    
    # Green rectangle for EXPECTED target (if provided)
    if target_pos is not None and 0 <= target_pos < matrix.shape[1]:
        rect_exp = Rectangle((target_pos, query_idx), 1, 1, fill=False, edgecolor='green', lw=3, linestyle='--', label="Expected")
        ax.add_patch(rect_exp)

    plt.title(f"Attention Matrix: L{layer} H{head} - {configkey} ({task})")
    plt.xlabel("Key Tokens"); plt.ylabel("Query Tokens")
    plt.tight_layout()
    
    os.makedirs(os.path.join(output_dir, task), exist_ok=True)
    save_path = os.path.join(output_dir, task, f"attn_matrix_L{layer}H{head}_{configkey}.png")
    plt.savefig(save_path)
    plt.close()

def plot_induction_heads(model, configkey, task, device, output_dir="results/plots/induction_heads"):
    """
    Probes for induction heads, saves a summary heatmap with the top head highlighted,
    and generates matrices for all heads above a score threshold.
    """
    import torch
    import seaborn as sns
    from matplotlib.patches import Rectangle
    from src.evaluation.metrics import probe_induction_heads
    
    os.makedirs(os.path.join(output_dir, task), exist_ok=True)
    
    model.eval()
    stoi = model.stoi
    if task in ("addition", "mapping"):
        keys = [k for k in stoi.keys() if re.match(r'^\d$', k)]
    else:
        keys = [k for k in stoi.keys() if re.match(r'^[A-Z]$', k)]
    
    if len(keys) < 4:
        keys = [k for k in stoi.keys() if len(k) == 1 and k not in ('<pad>', '\n', ' ', '>', '-', '=')]
    if len(keys) < 2: return None

    trigger_char = keys[0]
    target_char = keys[1]
    other_chars = keys[2:] if len(keys) > 2 else keys
    
    if task == "addition":
        common_oper = random.choice(other_chars)
        template_prev = f"{trigger_char} + {common_oper} = {target_char}\n"
        template_junk = f"{random.choice(other_chars)} + {random.choice(other_chars)} = {random.choice(other_chars)}\n"
        template_query = f"{trigger_char} + {common_oper} = "
    elif task == "mapping":
        template_prev = f"{trigger_char} -> {target_char}\n"
        template_junk = f"{random.choice(other_chars)} -> {random.choice(other_chars)}\n"
        template_query = f"{trigger_char} -> "
    else: # decoding
        template_prev = f"{trigger_char} -> {target_char}\n"
        template_junk = f"{random.choice(other_chars)} -> {random.choice(other_chars)}\n"
        template_query = f"{trigger_char} -> "

    full_text = template_prev + template_junk + template_query
    tokens = [stoi.get(c, 0) for c in full_text]
    target_pos = full_text.find(target_char)

    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        model(input_tensor)
        attentions = model.get_attention_maps()

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    scores = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        scores[l] = attentions[l][0, :, -1, target_pos].cpu().numpy()

    # Save summary heatmap with Best Head Highlighted
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(scores, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1)
    
    # Highlight the absolute best head on the heatmap
    max_idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    best_layer, best_head = max_idx
    rect_best = Rectangle((best_head, best_layer), 1, 1, fill=False, edgecolor='blue', lw=4, label="Best Head")
    ax.add_patch(rect_best)

    plt.title(f"Induction Head Scores (Task-Aware) - {configkey} ({task})")
    plt.xlabel("Head Index"); plt.ylabel("Layer Index")
    save_path = os.path.join(output_dir, task, f"induction_heatmap_{configkey}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # Visualize TOP Heads (all heads > 0.4 or at least the top one)
    threshold = 0.4
    high_heads = np.argwhere(scores > threshold)
    if len(high_heads) == 0:
        high_heads = [max_idx]
    
    # Cap at top 5 to avoid explosion
    top_indices = sorted(high_heads, key=lambda x: scores[x[0], x[1]], reverse=True)[:5]
    
    for layer, head in top_indices:
        visualize_head_attention(model, layer, head, tokens, configkey, task, target_pos, output_dir)

    return scores

def plot_noise_robustness(results_path="results/evaluation/suite_*.json", output_dir="results/plots"):
    """Plots Noise Robustness curves (Accuracy vs Noise Ratio)."""
    os.makedirs(output_dir, exist_ok=True)
    res_files = glob.glob(results_path)
    
    for res_file in res_files:
        task_match = re.search(r'suite_(.*).json', os.path.basename(res_file))
        if not task_match: continue
        task = task_match.group(1)
        
        with open(res_file, 'r') as f: data = json.load(f)
        sorted_scales = get_sorted_models(data)
        if not sorted_scales: continue
        
        plt.figure(figsize=(10, 6))
        plotted_any = False
        
        for scale in sorted_scales:
            res = data[scale]
            # Find keys like noise_0.0_accuracy, noise_0.2_accuracy, etc.
            noise_keys = sorted([k for k in res.keys() if k.startswith('noise_') and k.endswith('_accuracy')])
            if not noise_keys: continue
            
            ratios = [float(re.search(r'noise_(.*)_accuracy', k).group(1)) for k in noise_keys]
            accuracies = [res[k] for k in noise_keys]
            
            plt.plot(ratios, accuracies, marker='o', linewidth=2, label=scale)
            plotted_any = True
            
        if plotted_any:
            plt.title(f"ICL Noise Robustness: {task.capitalize()}", fontsize=14)
            plt.xlabel("Noise Ratio (Proportion of context examples with wrong answers)", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.ylim(-5, 105)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="Model Scale", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            task_dir = os.path.join(output_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            plt.savefig(os.path.join(task_dir, f"noise_robustness.png"), dpi=300, bbox_inches="tight")
        plt.close()

def plot_ood_accuracy(results_path="results/evaluation/suite_*.json", output_dir="results/plots"):
    """Plots OOD exact-match accuracy vs model scale."""
    os.makedirs(output_dir, exist_ok=True)
    res_files = glob.glob(results_path)

    for res_file in res_files:
        task_match = re.search(r'suite_(.*).json', os.path.basename(res_file))
        if not task_match:
            continue
        task = task_match.group(1)

        with open(res_file, 'r') as f:
            data = json.load(f)

        sorted_scales = sorted(data.keys(), key=parse_params)
        sorted_scales = [s for s in sorted_scales if parse_params(s) > 0 and 'ood_accuracy' in data[s]]
        if not sorted_scales:
            continue

        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        param_counts = [parse_params(s) for s in sorted_scales]
        ood_acc_vals = [data[s].get('ood_accuracy', 0) for s in sorted_scales]

        plt.figure(figsize=(10, 6))
        plt.semilogx(param_counts, ood_acc_vals, marker='o', linewidth=2, color='darkgreen', label='OOD Accuracy')
        plt.title(f"OOD Accuracy vs Parameters - {task.capitalize()}")
        plt.xlabel("Model Parameters (Log Scale)")
        plt.ylabel("OOD Exact Match Accuracy (%)")
        plt.ylim(-5, 105)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        for i, s in enumerate(sorted_scales):
            plt.text(param_counts[i], ood_acc_vals[i], s, fontsize=9, ha='right', va='bottom')
        plt.savefig(os.path.join(task_dir, "ood_accuracy.png"), bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize project results.")
    parser.add_argument("--logs", action="store_true", help="Plot training progress.")
    parser.add_argument("--icl", action="store_true", help="Plot ICL evaluation results.")
    parser.add_argument("--scaling", action="store_true", help="Plot model scaling metrics.")
    parser.add_argument("--noise", action="store_true", help="Plot noise robustness curves.")
    parser.add_argument("--ood", action="store_true", help="Plot OOD accuracy curves.")
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
    if args.noise or args.all:
        print("Plotting noise robustness...")
        plot_noise_robustness()
    if args.ood or args.all:
        print("Plotting OOD accuracy...")
        plot_ood_accuracy()

if __name__ == "__main__":
    main()
