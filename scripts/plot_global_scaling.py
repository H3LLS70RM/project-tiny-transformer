import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import os
import re

# Use utility from project if possible, otherwise redefine
def parse_params(scale):
    match = re.search(r'tt-(\d+)([kKmM]?)', scale)
    if not match: return 0
    val = int(match.group(1))
    unit = match.group(2).lower()
    if unit == 'k': val *= 1000
    elif unit == 'm': val *= 1000000
    return val

def plot_global_scaling(results_dir="results/evaluation", output_dir="results/plots", metric="accuracy"):
    """Aggregates all task families into one scaling plot for the given metric."""
    os.makedirs(output_dir, exist_ok=True)
    res_files = glob.glob(os.path.join(results_dir, "suite_*.json"))
    
    if not res_files:
        print(f"No suite files found in {results_dir}")
        return

    plt.figure(figsize=(12, 8))
    
    for res_file in sorted(res_files):
        task_match = re.search(r'suite_(.*).json', os.path.basename(res_file))
        if not task_match: continue
        task = task_match.group(1)
        
        with open(res_file, 'r') as f:
            data = json.load(f)
            
        # Exclude experimental / non-standard configs
        EXCLUDED_CONFIGS = {'tt-50k-balanced'}
        sorted_scales = sorted([s for s in data.keys() if s not in EXCLUDED_CONFIGS], key=parse_params)
        sorted_scales = [s for s in sorted_scales if parse_params(s) > 0]
        
        if not sorted_scales: continue
        
        param_counts = [parse_params(s) for s in sorted_scales]
        
        if metric == "accuracy":
            # Prefer exact_match_accuracy, fallback to 5_shot_accuracy
            vals = [data[s].get('exact_match_accuracy', data[s].get('5_shot_accuracy', 0)) for s in sorted_scales]
            ylabel = "Accuracy (%)"
            title = "Scaling Laws: ICL Accuracy vs Model Size"
            ylim = (-5, 105)
            y_text = -4
        else: # edit_distance
            # Prefer avg_edit_distance, fallback to 5_shot_edit_distance
            vals = [data[s].get('avg_edit_distance', data[s].get('5_shot_edit_distance', 0)) for s in sorted_scales]
            ylabel = "Avg Token Edit Distance (Lower is Better)"
            title = "Scaling Laws: Token Edit Distance vs Model Size"
            # Limit y to a reasonable range for edit distance (usually 0-3)
            max_v = max(vals) if vals else 2
            ylim = (-0.1, max_v + 0.5)
            y_text = -0.15

        plt.semilogx(param_counts, vals, marker='o', linewidth=2.5, label=task.replace('_', ' ').capitalize())

    plt.title(title, fontsize=16)
    plt.xlabel("Model Parameters (Log Scale)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(title="Task Family", fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(ylim)
    
    # Add vertical lines for ALL model scales for clarity
    SCALE_ORDER = ['tt-2k', 'tt-5k', 'tt-9k', 'tt-14k', 'tt-26k', 'tt-49k', 'tt-141k', 'tt-808k', 'tt-3M']
    for scale in SCALE_ORDER:
        p = parse_params(scale)
        if p > 0:
            plt.axvline(x=p, color='gray', linestyle='--', alpha=0.15)
            plt.text(p, y_text, scale, rotation=45, fontsize=9, color='dimgray', ha='center', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"global_scaling_{metric}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Global scaling {metric} plot saved to {output_path}")

if __name__ == "__main__":
    # Also check the dated results folder from early logs
    target_dir = "results/evaluation"
    if not glob.glob(os.path.join(target_dir, "suite_*.json")):
        dated_dir = "results_20260328/evaluation"
        if os.path.exists(dated_dir):
            target_dir = dated_dir
            print(f"Using dated results directory: {target_dir}")
            
    plot_global_scaling(results_dir=target_dir, metric="accuracy")
    plot_global_scaling(results_dir=target_dir, metric="edit_distance")

if __name__ == "__main__":
    # Also check the dated results folder from early logs
    target_dir = "results/evaluation"
    if not glob.glob(os.path.join(target_dir, "suite_*.json")):
        dated_dir = "results_20260328/evaluation"
        if os.path.exists(dated_dir):
            target_dir = dated_dir
            print(f"Using dated results directory: {target_dir}")
            
    plot_global_scaling(results_dir=target_dir)
