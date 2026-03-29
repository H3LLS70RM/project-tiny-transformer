import torch
import random
import os
from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.plots.visualize import parse_params

def evaluate_label_flipping(model, task, n_samples=20, device='cpu', hard_icl=False):
    """Evaluates if the model can follow 'flipped' labels in the context."""
    stoi, itos = model.stoi, model.itos
    model.eval()
    
    dec_sep = "->" if task in ("mapping", "decoding") else "="
    if hard_icl:
        dec_sep = random.choice(["->", ":", "==", "=>", "="])
    sep = f" {dec_sep} " if dec_sep != ":" else ":"
    
    correct_flips = 0
    predictions = []
    
    for _ in range(n_samples):
        if task == "addition":
            a, b = random.randint(0, 5), random.randint(0, 5)
            true_ans = a + b
            flip_ans = random.randint(0, 9)
            while flip_ans == true_ans:
                flip_ans = random.randint(0, 9)
            prompt = f"{a} + {b}{sep}{flip_ans}\n{a} + {b}{sep}"
            target = str(flip_ans)
        elif task == "mapping":
            x = random.randint(1, 20)
            true_y = x 
            flip_y = random.randint(0, 9)
            while flip_y == true_y:
                flip_y = random.randint(0, 9)
            prompt = f"{x}{sep}{flip_y}\n{x}{sep}"
            target = str(flip_y)
        elif task == "decoding":
            # f
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            k = random.choice(chars)
            true_v = k 
            flip_v = random.choice([c for c in chars if c != true_v])
            prompt = f"{k}{sep}{flip_v}\n{k}{sep}"
            target = flip_v
        elif task in ("arithmetic", "arithmetic_symbolic", "arithmetic_shuffled"):
            if task == "arithmetic":
                symbol = "+"
            elif task == "arithmetic_shuffled":
                symbol = random.choice(["+", "-", "*", "max", "min"])
            else: # arithmetic_symbolic
                symbol = random.choice(list("@#$%^&*?"))
                
            a, b = random.randint(0, 9), random.randint(0, 9)
            true_ans = a + b # Specific value doesn't matter for the flip logic
            flip_ans = random.randint(-9, 30)
            while flip_ans == true_ans:
                flip_ans = random.randint(-9, 30)
            prompt = f"{a} {symbol} {b}{sep}{flip_ans}\n{a} {symbol} {b}{sep}"
            target = str(flip_ans)
        else:
            continue

        p_toks = [stoi.get(c, 0) for c in prompt]
        curr = p_toks[:]
        stop_id = stoi.get('\n', -1)
        
        with torch.no_grad():
            for _ in range(10):
                inp = torch.tensor([curr]).to(device)
                out = model(inp)
                nxt = out[0, -1, :].argmax().item()
                curr.append(nxt)
                if nxt == 0 or nxt == stop_id: break
        
        gen_text = ''.join([itos.get(t, '') for t in curr[len(p_toks):] if t != 0])
        extracted = gen_text.split('\n')[0].strip()
        
        if extracted == target:
            correct_flips += 1
            
        predictions.append({
            "flip_prompt": prompt,
            "flip_target": target,
            "flip_pred": extracted
        })
        
    return (correct_flips / n_samples), predictions

def evaluate_noise_robustness(model, task, noise_ratios=[0.0, 0.2, 0.5, 0.8], n_samples=100, n_context=5, device='cpu', digit_level=False, rule_diversity=False, hard_icl=False):
    """Evaluates model performance with noisy in-context examples."""
    stoi, itos = model.stoi, model.itos
    model.eval()
    results = {}
    
    for ratio in noise_ratios:
        test_data = SyntheticICLDataset(
            task=task, n_samples=n_samples, n_context=n_context, noise_ratio=ratio, 
            digit_level=digit_level, rule_diversity=rule_diversity, hard_icl=hard_icl
        ).build_dataset(return_answer=True)
        total_correct = 0
        
        with torch.no_grad():
            for item in test_data:
                p_toks = [stoi.get(c, 0) for c in item['prompt']]
                curr = p_toks[:]
                stop_id = stoi.get('\n', -1)
                
                for _ in range(20):
                    inp = torch.tensor([curr]).to(device)
                    out = model(inp)
                    nxt = out[0, -1, :].argmax().item()
                    curr.append(nxt)
                    if nxt == 0 or nxt == stop_id: break
                
                gen_text = ''.join([itos.get(t, '') for t in curr[len(p_toks):] if t != 0])
                # Universal extraction: everything up to the first newline
                predicted = gen_text.split('\n')[0].strip()
                
                if predicted == item['answer'].strip():
                    total_correct += 1
        
        results[f"noise_{ratio:.1f}_accuracy"] = round((total_correct / n_samples) * 100, 2)
        
    return results

def evaluate_generalization(model, task, evaluate_model_fn, n_samples=100, device='cpu', 
                            digit_level=False, rule_diversity=False, mapping_ood_type='extrapolation', decoding_reversal=False, hard_icl=False, exclude_family_idx=None):
    """Evaluates model performance on out-of-distribution (OOD) data."""
    print(f"  Evaluating Generalization (OOD) for {task}...")
    
    # Generate OOD dataset using provided V2 flags
    ood_ds = SyntheticICLDataset(
        task=task, n_samples=n_samples, n_context=5, is_ood=True,
        digit_level=digit_level, rule_diversity=rule_diversity,
        mapping_ood_type=mapping_ood_type, decoding_reversal=decoding_reversal,
        hard_icl=hard_icl, rule_family_idx=exclude_family_idx
    ).build_dataset(return_answer=True)
    if not ood_ds:
        return {}
        
    res = evaluate_model_fn(model, ood_ds, task, device=device)
    return {
        "ood_accuracy": round(res['exact_match_accuracy'], 2),
        "ood_avg_edit_distance": round(res['avg_edit_distance'], 2)
    }

def generate_analysis_summary(emerg_data, results_dir="results"):
    """Generates a text-based analysis of ICL emergence levels."""
    summary_path = os.path.join(results_dir, "icl_analysis.txt")
    os.makedirs(results_dir, exist_ok=True)
    
    summary_path = os.path.join(results_dir, "icl_analysis.txt")
    os.makedirs(results_dir, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("IN-CONTEXT LEARNING EMERGENCE ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        for task, scales in emerg_data.items():
            f.write("="*70 + f"\nTask: {task.upper()}\n" + "="*70 + "\n\n")
            # Sort dynamically by parameter count
            sorted_scales = sorted(list(scales.keys()), key=parse_params)
            
            for scale in sorted_scales:
                res = scales[scale]
                f.write(f"  Model: {scale}\n")
                f.write(f"  " + "-"*66 + "\n")
                
                flip = res.get('flip_score', 0)
                lcs = res.get('lcs_score', 0)
                ind = res.get('max_induction_score', 0)
                
                f.write(f"  Flip Score:      {flip*100:6.2f}%\n")
                f.write(f"  LCS Score:       {lcs:8.4f}\n")
                f.write(f"  Induction:       {ind:8.4f}\n")
                
                status = "NO ICL"
                if flip > 0.4 or lcs > 1.0: status = "STRONG ICL"
                elif flip > 0.15 or lcs > 0.3: status = "MODERATE ICL"
                elif flip > 0.05 or lcs > 0.1: status = "WEAK ICL"
                
                f.write(f"  Status:              {status}\n\n")
            f.write("\n")
    
    print(f"Analysis summary saved to {summary_path}")
