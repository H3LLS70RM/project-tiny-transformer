import torch
import json
import os
import re
import argparse
import glob
import sys
import subprocess

from src.model.tiny_transformer import TinyTransformer
from src.configs.model_configs import config
from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.plots.visualize import (
    plot_training_progress, plot_icl_results, plot_emergence_results, 
    plot_model_scaling_metrics, plot_noise_robustness, plot_ood_accuracy, plot_induction_heads,
    parse_params
)

from src.evaluation.metrics import (
    calculate_edit_distance,
    calculate_lcs, 
    probe_induction_heads, 
    evaluate_icl_scaling
)
from src.evaluation.probes import (
    evaluate_label_flipping, 
    evaluate_noise_robustness, 
    evaluate_generalization, 
    generate_analysis_summary
)
from src.utils import get_task_vocab


# --- Core Evaluation Functions ---

def evaluate_model(model, dataset, task, batch_size=32, max_len=256, device='cpu', model_scale=None, json_path=None):
    """Standard Exact Match and Token Accuracy evaluation."""
    stoi, itos = model.stoi, model.itos
    model.eval()
    
    total_exact_matches = 0
    total_tokens, correct_tokens = 0, 0
    total_edit_distance = 0
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            full_seqs, prompt_lens = [], []
            for item in batch:
                p_toks, a_toks = [stoi.get(c, 0) for c in item['prompt']], [stoi.get(c, 0) for c in item['answer']]
                full_seqs.append((p_toks + a_toks)[:max_len])
                prompt_lens.append(len(p_toks))
            
            max_batch_len = max(len(s) for s in full_seqs)
            padded = torch.tensor([s + [0]*(max_batch_len - len(s)) for s in full_seqs]).to(device)
            logits = model(padded[:, :-1])
            tgt = padded[:, 1:]
            
            for b_idx in range(len(batch)):
                start, end = max(0, prompt_lens[b_idx] - 1), len(full_seqs[b_idx]) - 1
                if end > start:
                    pred = logits[b_idx, start:end].argmax(dim=-1)
                    actual = tgt[b_idx, start:end]
                    correct_tokens += (pred == actual).sum().item()
                    total_tokens += (end - start)
                    
                    # Greedy generation for exact match and edit distance
                    gen_toks = padded[b_idx, :prompt_lens[b_idx]].tolist()
                    max_gen = min(20, model.max_len - len(gen_toks))
                    stop_id = stoi.get('\n', -1)
                    for _ in range(max_gen):
                        out = model(torch.tensor([gen_toks]).to(device))
                        nxt = out[0, -1, :].argmax().item()
                        gen_toks.append(nxt)
                        if nxt == 0 or nxt == stop_id or len(gen_toks) >= model.max_len: break
                    
                    gen_answer = ''.join([itos.get(t, '') for t in gen_toks[prompt_lens[b_idx]:] if t != 0])
                    true_answer = batch[b_idx]['answer'].strip()
                    # Universal extraction: everything up to the first newline
                    extracted_answer = gen_answer.split('\n')[0].strip()
                    
                    # Calculate edit distance on the extracted answer part
                    total_edit_distance += calculate_edit_distance(extracted_answer, true_answer)
                    
                    if extracted_answer == true_answer:
                        total_exact_matches += 1
            
    results = {
        "exact_match_accuracy": (total_exact_matches / len(dataset)) * 100,
        "token_accuracy": (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0,
        "avg_edit_distance": total_edit_distance / len(dataset) if len(dataset) > 0 else 0,
    }

    if json_path and model_scale:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        all_data = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try: all_data = json.load(f)
                except: all_data = {}
        
        if task not in all_data: all_data[task] = {}
        all_data[task][model_scale] = results
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=2)

    return results

def save_raw_predictions(model, dataset, task, model_scale, device='cpu', n_samples=20, suffix=""):
    """Saves raw prompts and model outputs to a JSON file for manual inspection."""
    stoi, itos = model.stoi, model.itos
    model.eval()
    
    samples = dataset[:n_samples]
    predictions = []
    
    desc = f"{model_scale} {suffix}".strip()
    print(f"  Logging {len(samples)} raw predictions for {desc}...")
    
    with torch.no_grad():
        for item in samples:
            prompt_text = item['prompt']
            if task == "mapping":
                rule_family_idx = item['rule_family_idx']
            expected_answer = item['answer'].strip()
            
            p_toks = [stoi.get(c, 0) for c in prompt_text]
            gen_toks = list(p_toks)
            
            # Greedy generation
            max_gen = min(20, model.max_len - len(p_toks))
            stop_id = stoi.get('\n', -1)
            for _ in range(max_gen):
                out = model(torch.tensor([gen_toks]).to(device))
                nxt = out[0, -1, :].argmax().item()
                gen_toks.append(nxt)
                if nxt == 0 or nxt == stop_id or len(gen_toks) >= model.max_len: break
                
            gen_text = ''.join([itos.get(t, '') for t in gen_toks[len(p_toks):] if t != 0])
            
            # Universal extraction: everything up to the first newline
            extracted_answer = gen_text.split('\n')[0].strip()
            prediction = {
                "prompt": prompt_text,
                "expected_answer": expected_answer,
                "generated_text": gen_text,
                "extracted_answer": extracted_answer,
                "is_correct": extracted_answer == expected_answer,
            }
            if task == "mapping":
                prediction["rule_family_idx"] = rule_family_idx
            predictions.append(prediction)
            
    save_dir = "results/evaluation/raw_predictions"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{task}_{model_scale}_{suffix}.json" if suffix else f"{task}_{model_scale}.json"
    save_path = f"{save_dir}/{filename}"
    
    with open(save_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Raw predictions logged to {save_path}")


# --- Main Entry Point ---

def run_suite(model_scale, task, device='cpu', step=None, 
              digit_level=False, rule_diversity=False, 
              mapping_ood_type='extrapolation', decoding_reversal=False,
              hard_icl=False, exclude_family_idx=None):
    print(f"\n--- Running Evaluation Suite: {model_scale} | Task: {task} ---")
    cfg = config(model_scale)
    
    # Vocabulary setup (consistent with training)
    stoi, itos = get_task_vocab(task)
    
    model = TinyTransformer(
        vocab_size=len(stoi), dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'],
        stoi=stoi, itos=itos, configkey=model_scale, mlp_dim=cfg['mlp_dim'],
        max_len=cfg.get('max_len', 256), use_rope=True
    ).to(device)
    
    ckpt_dir = f"checkpoints/{task}/{model_scale}/"
    
    # Smart Checkpoint Loading
    target_ckpt = None
    if step:
        target_ckpt = os.path.join(ckpt_dir, f"model-step-{step}.pt")
    else:
        # Check for model_best_icl.pt, then model_best.pt, then model_latest.pt, then highest step
        options = [
            os.path.join(ckpt_dir, "model_best_icl.pt"),
            os.path.join(ckpt_dir, "model_best.pt"),
            os.path.join(ckpt_dir, "model_latest.pt")
        ]
        for opt in options:
            if os.path.exists(opt):
                target_ckpt = opt
                break
        
        if not target_ckpt:
            ckpts = glob.glob(f"{ckpt_dir}model-step-*.pt")
            if ckpts:
                target_ckpt = max(ckpts, key=lambda x: int(re.search(r"model-step-(\d+)", x).group(1)))

    if not target_ckpt or not os.path.exists(target_ckpt):
        print(f"Skipping: No suitable checkpoint found in {ckpt_dir}"); return
    
    print(f"Loading checkpoint: {target_ckpt}")
    model.load_state_dict(torch.load(target_ckpt, map_location=device))
    
    flip_score, _ = evaluate_label_flipping(model, task, device=device, hard_icl=hard_icl)
    max_induction = probe_induction_heads(model, task, device)
    # Automatically generate induction head heatmap
    plot_induction_heads(model, model_scale, task, device=device)
    lcs = calculate_lcs(model, task, device=device, digit_level=digit_level)
    noise_results = evaluate_noise_robustness(model, task, device=device, digit_level=digit_level, rule_diversity=rule_diversity, hard_icl=hard_icl)
    generalization_results = evaluate_generalization(
        model, task, evaluate_model, device=device,
        digit_level=digit_level, rule_diversity=rule_diversity,
        mapping_ood_type=mapping_ood_type, decoding_reversal=decoding_reversal,
        hard_icl=hard_icl, exclude_family_idx=exclude_family_idx
    )
    
    results = {
        "model_scale": model_scale,
        "task": task,
        "checkpoint": os.path.basename(target_ckpt),
        "lcs_score": lcs,
        "induction_score": max_induction,
        **evaluate_icl_scaling(model, task, evaluate_model, device=device, digit_level=digit_level, rule_diversity=rule_diversity, hard_icl=hard_icl),
        "flip_score": flip_score,
        **noise_results,
        **generalization_results
    }

    # --- Logging Raw Predictions ---
    # 1. In-Distribution (ID) samples
    logging_ds_id = SyntheticICLDataset(
        task=task, n_samples=50, n_context=5, is_ood=False,
        digit_level=digit_level, rule_diversity=rule_diversity,
        mapping_ood_type=mapping_ood_type, decoding_reversal=decoding_reversal,
        hard_icl=hard_icl, exclude_family_idx=exclude_family_idx
    ).build_dataset(return_answer=True)
    save_raw_predictions(model, logging_ds_id, task, model_scale, device=device, suffix="id")
    
    # 2. Out-of-Distribution (OOD) samples
    logging_ds_ood = SyntheticICLDataset(
        task=task, n_samples=50, n_context=5, is_ood=True,
        digit_level=digit_level, rule_diversity=rule_diversity,
        mapping_ood_type=mapping_ood_type, decoding_reversal=decoding_reversal,
        hard_icl=hard_icl, rule_family_idx=exclude_family_idx
    ).build_dataset(return_answer=True)
    save_raw_predictions(model, logging_ds_ood, task, model_scale, device=device, suffix="ood")
    
    # 1. Standard Suite Results
    save_path = f"results/evaluation/suite_{task}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_data = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f: all_data = json.load(f)
    all_data[model_scale] = results
    # Sort correctly by parameter count before saving
    all_data = {k: all_data[k] for k in sorted(all_data.keys(), key=parse_params)}
    with open(save_path, "w") as f: json.dump(all_data, f, indent=2)
    
    # 2. Emergence Results (consolidated for all tasks/scales as expected by visualize.py)
    emergence_path = "results/icl_emergence_results.json"
    emerg_data = {}
    if os.path.exists(emergence_path):
        with open(emergence_path, "r") as f: emerg_data = json.load(f)
    
    if task not in emerg_data: emerg_data[task] = {}
    emerg_data[task][model_scale] = {
        "flip_score": flip_score,
        "lcs_score": lcs,
        "max_induction_score": max_induction
    }
    # Sort tasks and scales correctly by parameter count
    emerg_data[task] = {k: emerg_data[task][k] for k in sorted(emerg_data[task].keys(), key=parse_params)}
    with open(emergence_path, "w") as f: json.dump(emerg_data, f, indent=2)
    
    # 3. Generate analysis summary
    generate_analysis_summary(emerg_data)
    
    print(f"Results saved to {save_path}")
    for k, v in results.items(): print(f"  {k}: {v}")
    
    # Automatically update scaling plots
    print("  Updating scaling plots...")
    plot_model_scaling_metrics()

    # Update consolidated noise JSON for this task with the current model's noise results
    try:
        noise_out = f"results/evaluation/noise_robustness_{task}.json"
        consolidated = {}
        if os.path.exists(noise_out):
            with open(noise_out, 'r') as f:
                try: consolidated = json.load(f)
                except: consolidated = {}

        # noise_results contains keys like 'noise_0.0_accuracy'
        if isinstance(noise_results, dict) and noise_results:
            model_map = {}
            for k, v in noise_results.items():
                m = re.search(r'noise_([0-9.]+)', k)
                if m:
                    model_map[m.group(1)] = v
                else:
                    try:
                        model_map[str(float(k))] = v
                    except Exception:
                        continue

            if model_map:
                consolidated[model_scale] = {kk: model_map[kk] for kk in sorted(model_map.keys(), key=lambda x: float(x))}
                # Sort consolidated by model scale correctly
                consolidated = {k: consolidated[k] for k in sorted(consolidated.keys(), key=parse_params)}
                os.makedirs(os.path.dirname(noise_out), exist_ok=True)
                with open(noise_out, 'w') as f:
                    json.dump(consolidated, f, indent=2)
                print(f"  Updated noise JSON: {noise_out}")
    except Exception as e:
        print(f"  Warning: failed to update noise JSON for {task}/{model_scale}: {e}")

    # Generate noise robustness plot for this task (safe to call repeatedly)
    try:
        print("\nGenerating plots...")
        plot_training_progress()
        plot_icl_results()
        plot_emergence_results()
        plot_model_scaling_metrics()
        plot_noise_robustness()
        plot_ood_accuracy()
    except Exception as e:
        print(f"  Warning: failed to generate plots: {e}")

if __name__ == "__main__":
    # Automatically switch to venv if not already inside it
    if not sys.prefix.endswith('.venv'):
        venv_exe = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(os.getcwd(), '.venv', 'bin', 'python')
        if os.path.exists(venv_exe):
            print(f"  --> Redirecting to Virtual Environment: {venv_exe}")
            os.environ['PYTHONPATH'] = os.getcwd()
            subprocess.run([venv_exe] + sys.argv)
            sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=['tt-8k', 'tt-26k', 'tt-150k'])
    parser.add_argument("--tasks", nargs="+", default=['addition', 'arithmetic_symbolic', 'mapping', 'decoding'])
    parser.add_argument("--step", type=int, default=None, help="Evaluate a specific step.")
    
    # Generalization-V2 Flags
    parser.add_argument("--digit_level", action="store_true", help="Use digit-level formatting for addition.")
    parser.add_argument("--rule_diversity", action="store_true", help="Enable rule diversity for mapping.")
    parser.add_argument("--mapping_ood_type", type=str, choices=['exponential', 'extrapolation', 'modulo'], default='extrapolation', help="OOD type for mapping.")
    parser.add_argument("--decoding_reversal", action="store_true", help="Enable soft reversal for decoding OOD.")
    parser.add_argument("--hard_icl", action="store_true", help="Enable strict ICL mode with symbolic ops and jitter.")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for task in args.tasks:
        for cfg in args.configs:
            run_suite(
                cfg, task, device, step=args.step,
                digit_level=args.digit_level, rule_diversity=args.rule_diversity,
                mapping_ood_type=args.mapping_ood_type, decoding_reversal=args.decoding_reversal,
                hard_icl=args.hard_icl
            )
