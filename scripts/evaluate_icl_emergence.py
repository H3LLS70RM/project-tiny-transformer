import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import re
import argparse
import glob
from tqdm import tqdm

from src.model.tiny_transformer import TinyTransformer
from src.configs.model_configs import config
from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.evaluation.suite import calculate_lcs, probe_induction_heads, evaluate_icl_scaling

def evaluate_label_flipping(model, task, n_samples=20, device='cpu'):
    """
    Evaluates if the model can follow 'flipped' labels in the context.
    Example (Addition): "2 + 2 = 5\n2 + 2 = " -> Model should predict '5'.
    """
    stoi, itos = model.stoi, model.itos
    model.eval()
    
    correct_flips = 0
    predictions = []
    
    # Task-specific logic for generating flipped examples
    for _ in range(n_samples):
        if task == "addition":
            a, b = random.randint(0, 5), random.randint(0, 5)
            true_ans = a + b
            flip_ans = random.randint(0, 9)
            while flip_ans == true_ans:
                flip_ans = random.randint(0, 9)
            prompt = f"{a} + {b} = {flip_ans}\n{a} + {b} = "
            target = str(flip_ans)
        elif task == "mapping":
            x = random.randint(1, 10)
            true_y = x # simplified identity or similar
            flip_y = random.randint(0, 9)
            while flip_y == true_y:
                flip_y = random.randint(0, 9)
            prompt = f"{x} -> {flip_y}\n{x} -> "
            target = str(flip_y)
        elif task == "decoding":
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            k = random.choice(chars)
            true_v = k # identity mapping usually
            flip_v = random.choice([c for c in chars if c != true_v])
            prompt = f"{k} -> {flip_v}\n{k} -> "
            target = flip_v
        else:
            continue

        p_toks = [stoi.get(c, 0) for c in prompt]
        curr = p_toks[:]
        
        with torch.no_grad():
            for _ in range(5):
                inp = torch.tensor([curr]).to(device)
                out = model(inp)
                nxt = out[0, -1, :].argmax().item()
                curr.append(nxt)
                if nxt == 0: break
        
        gen_text = ''.join([itos.get(t, '') for t in curr[len(p_toks):] if t != 0])
        match = re.search(r'\d+', gen_text) if task in ('addition', 'mapping') else re.search(r'[A-Z]', gen_text)
        pred = match.group(0) if match else gen_text.strip()
        
        if pred == target:
            correct_flips += 1
            
        predictions.append({
            "flip_prompt": prompt,
            "flip_target": target,
            "flip_pred": pred
        })
        
    return (correct_flips / n_samples), predictions

def run_emergence_evaluation(configs, tasks, results_dir="results"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {task: {} for task in tasks}
    
    for task in tasks:
        print(f"\n--- Evaluating Emergence: Task {task} ---")
        for model_scale in configs:
            print(f"  Model: {model_scale}")
            cfg = config(model_scale)
            
            # Setup vocab
            dummy_ds = SyntheticICLDataset(task=task, n_samples=100)
            vocab_data = dummy_ds.build_dataset(return_answer=True)
            all_text = ''.join([item['prompt'] + item['answer'] for item in vocab_data])
            vocab = sorted(set(all_text))
            stoi = {ch: i+1 for i, ch in enumerate(vocab)}; itos = {i+1: ch for i, ch in enumerate(vocab)}
            stoi['<pad>'] = 0; itos[0] = '<pad>'
            
            model = TinyTransformer(
                vocab_size=len(stoi), dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'],
                stoi=stoi, itos=itos, configkey=model_scale, mlp_dim=cfg['mlp_dim'],
                max_len=cfg.get('max_len', 256), use_rope=True
            ).to(device)
            
            # Load latest checkpoint
            ckpt_dir = f"checkpoints/{task}/{model_scale}/"
            ckpts = glob.glob(f"{ckpt_dir}model-step-*.pt")
            if not ckpts:
                print(f"    No checkpoints found for {model_scale}, skipping.")
                continue
                
            latest_ckpt = max(ckpts, key=lambda x: int(re.search(r"model-step-(\d+)", x).group(1)))
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
            
            # 1. Flip Score
            flip_score, flip_preds = evaluate_label_flipping(model, task, device=device)
            
            # 2. LCS Score
            lcs_score = calculate_lcs(model, task, device=device)
            
            # 3. Induction Score
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
    print(f"\nSaved results to {res_path}")

    # Generate Analysis Summary
    generate_analysis_summary(all_results, results_dir)

def generate_analysis_summary(data, results_dir):
    summary_path = os.path.join(results_dir, "icl_analysis.txt")
    with open(summary_path, "w") as f:
        f.write("\n" + "="*70 + "\n")
        f.write("IN-CONTEXT LEARNING EMERGENCE ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        for task, scales in data.items():
            f.write("="*70 + f"\nTask: {task.upper()}\n" + "="*70 + "\n\n")
            for scale, res in scales.items():
                f.write(f"  Model: {scale}\n")
                f.write(f"  " + "-"*66 + "\n")
                
                # We don't have scaling data here, so we focus on what we have
                flip = res['flip_score']
                lcs = res['lcs_score']
                ind = res['max_induction_score']
                
                f.write(f"  Flip Score:      {flip*100:6.2f}%\n")
                f.write(f"  LCS Score:       {lcs:8.4f}\n")
                f.write(f"  Induction:       {ind:8.4f}\n")
                
                status = "NO ICL"
                if flip > 0.4 or lcs > 1.0: status = "STRONG ICL"
                elif flip > 0.15 or lcs > 0.3: status = "MODERATE ICL"
                elif flip > 0.05 or lcs > 0.1: status = "WEAK ICL"
                
                f.write(f"  Status:              {status}\n\n")
            f.write("\n")
            
    print(f"Saved analysis summary to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=['tt-8k', 'tt-14k', 'tt-26k', 'tt-50k', 'tt-150k', 'tt-800k', 'tt-3000k'])
    parser.add_argument("--tasks", nargs="+", default=['addition', 'mapping', 'decoding'])
    args = parser.parse_args()
    
    run_emergence_evaluation(args.configs, args.tasks)
