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

# --- Utility Functions ---

def calculate_edit_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]
    for x in range(size_x): matrix[x][0] = x
    for y in range(size_y): matrix[0][y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = matrix[x-1][y-1]
            else:
                matrix[x][y] = min(matrix[x-1][y] + 1, matrix[x-1][y-1] + 1, matrix[x][y-1] + 1)
    return matrix[size_x-1][size_y-1]

def calculate_conditional_likelihood(model, context_ids, target_ids, device):
    if len(target_ids) == 0: return 0.0
    full_seq = context_ids + target_ids
    input_ids = torch.tensor([full_seq], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids)
    start_idx = max(0, len(context_ids) - 1)
    target_len = len(target_ids)
    selected_logits = logits[0, start_idx : start_idx + target_len, :]
    selected_targets = input_ids[0, len(context_ids) : len(context_ids) + target_len]
    min_len = min(len(selected_logits), len(selected_targets))
    loss = F.cross_entropy(selected_logits[:min_len], selected_targets[:min_len], reduction='mean')
    return torch.exp(-loss).item()

# --- Core Evaluation Functions ---

def evaluate_model(model, dataset, task, batch_size=32, max_len=256, device='cpu', model_scale=None):
    """Standard Exact Match and Token Accuracy evaluation."""
    stoi, itos = model.stoi, model.itos
    model.eval()
    
    total_exact_matches = 0
    total_tokens, correct_tokens = 0, 0
    total_loss, num_batches = 0, 0
    
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
                    
                    # Greedy generation for exact match
                    gen_toks = padded[b_idx, :prompt_lens[b_idx]].tolist()
                    for _ in range(20):
                        out = model(torch.tensor([gen_toks]).to(device))
                        nxt = out[0, -1, :].argmax().item()
                        gen_toks.append(nxt)
                        if nxt == 0: break
                    gen_text = ''.join([itos.get(t, '') for t in gen_toks[prompt_lens[b_idx]:] if t != 0])
                    # Task-specific extraction
                    match = re.search(r'\d+', gen_text) if task in ('addition', 'mapping') else re.search(r'[A-Z]', gen_text)
                    if (match.group(0) if match else gen_text.strip()) == batch[b_idx]['answer'].strip():
                        total_exact_matches += 1
            
            num_batches += 1
            
    results = {
        "exact_match_accuracy": (total_exact_matches / len(dataset)) * 100,
        "token_accuracy": (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0,
    }
    return results

def calculate_lcs(model, task, n_samples=250, device='cpu'):
    """Calculates Learning-to-Context Slope (LCS)."""
    stoi = model.stoi
    dataset = SyntheticICLDataset(task=task, n_samples=n_samples, n_context=1)
    s_vals, t_vals = [], []
    nl_id = [stoi.get("\n", 0)]
    
    for _ in range(n_samples):
        # Generate Q, X, D parts manually for clarity
        if task == "addition":
            da, db = random.randint(0, 99), random.randint(0, 99)
            d_str = f"{da} + {db} = {da+db}"
            qa, qb = random.randint(0, 99), random.randint(0, 99)
            q_str, x_str = f"{qa} + {qb} = ", str(qa + qb)
        elif task == "mapping":
            ma, mb = random.randint(1, 10), random.randint(0, 50)
            dx = random.randint(0, 99)
            d_str = f"{dx} -> {dataset.mapping_fn(ma, mb, dx)}"
            qx = random.randint(0, 99)
            q_str, x_str = f"{qx} -> ", str(dataset.mapping_fn(ma, mb, qx))
        else: # decoding fallback
            motif = [random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)]
            d_str = f"{motif[0]} -> {motif[1]}" # simplified relevance
            q_str, x_str = f"{motif[0]} -> ", motif[1]

        q_ids, x_ids, d_ids = [stoi.get(c, 0) for c in q_str], [stoi.get(c, 0) for c in x_str], [stoi.get(c, 0) for c in d_str]
        
        p_x_q = calculate_conditional_likelihood(model, q_ids, x_ids, device)
        p_x_qd = calculate_conditional_likelihood(model, d_ids + nl_id + q_ids, x_ids, device)
        p_d_q = calculate_conditional_likelihood(model, q_ids, d_ids, device)
        p_d_qx = calculate_conditional_likelihood(model, q_ids + x_ids, d_ids, device)
        
        s_vals.append(p_x_qd - p_x_q)
        t_vals.append(p_d_qx - p_d_q)
        
    s_mean, t_mean = np.mean(s_vals), np.mean(t_vals)
    if abs(s_mean) < 0.001: return 0.0
    num = sum((t - t_mean)**2 for t in t_vals)
    den = sum((t - t_mean) * (s - s_mean) for t, s in zip(t_vals, s_vals))
    return float(den / num) if abs(num) > 1e-9 else 0.0

def probe_induction_heads(model, task, device):
    """Measures Induction Head strength across any task."""
    stoi = model.stoi
    # Pick valid tokens based on what the model was likely trained on
    if task in ("addition", "mapping"):
        valid_toks = [v for k, v in stoi.items() if re.match(r'^\d$', k)]
    else:
        valid_toks = [v for k, v in stoi.items() if re.match(r'^[A-Z]$', k)]
    
    # Fallback if task-specific regex fails
    if len(valid_toks) < 2:
        valid_toks = [v for k, v in stoi.items() if len(k) == 1 and k != '<pad>']
        
    if len(valid_toks) < 2: return 0.0
    
    seq_len = 20
    trigger, target = valid_toks[0], valid_toks[1]
    pool = [t for t in valid_toks if t not in (trigger, target)]
    seq = [random.choice(pool) for _ in range(seq_len)]
    
    t_pos = random.randint(0, seq_len-5)
    seq[t_pos], seq[t_pos+1], seq[-1] = trigger, target, trigger
    
    with torch.no_grad():
        model(torch.tensor([seq]).to(device))
        attns = model.get_attention_maps() # [L, B, H, T, T]
    
    scores = []
    for layer_attn in attns:
        # Last token (Query) attends to the token at t_pos+1 (Key/Value)
        scores.append(layer_attn[0, :, -1, t_pos+1].cpu().numpy())
    return float(np.max(scores)) if scores else 0.0

def evaluate_icl_scaling(model, task, n_samples=100, device='cpu'):
    """Evaluates accuracy across literal few-shot counts."""
    results = {}
    for n_shots in [0, 1, 3, 5]:
        ds = SyntheticICLDataset(task=task, n_samples=n_samples, n_context=n_shots).build_dataset(return_answer=True)
        res = evaluate_model(model, ds, task, device=device)
        results[f"{n_shots}_shot_accuracy"] = res['exact_match_accuracy']
    return results

# --- Main Entry Point ---

def run_suite(model_scale, task, device='cpu'):
    print(f"\n--- Running Evaluation Suite: {model_scale} | Task: {task} ---")
    cfg = config(model_scale)
    
    # Vocabulary setup (consistent with training)
    dummy_ds = SyntheticICLDataset(task=task, n_samples=1000)
    vocab_data = dummy_ds.build_dataset(return_answer=True)
    all_text = ''.join([item['prompt'] + item['answer'] for item in vocab_data])
    vocab = sorted(set(all_text))
    stoi = {ch: i+1 for i, ch in enumerate(vocab)}
    itos = {i+1: ch for i, ch in enumerate(vocab)}
    stoi['<pad>'] = 0; itos[0] = '<pad>'
    
    model = TinyTransformer(
        vocab_size=len(stoi), dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'],
        stoi=stoi, itos=itos, configkey=model_scale, mlp_dim=cfg['mlp_dim'],
        max_len=cfg.get('max_len', 256), use_rope=True
    ).to(device)
    
    ckpt_dir = f"checkpoints/{task}/{model_scale}/"
    ckpts = glob.glob(f"{ckpt_dir}model-step-*.pt")
    if not ckpts:
        print(f"Skipping: No checkpoints in {ckpt_dir}"); return
    
    latest_ckpt = max(ckpts, key=lambda x: int(re.search(r"model-step-(\d+)", x).group(1)))
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    
    results = {
        "model_scale": model_scale,
        "task": task,
        "checkpoint": os.path.basename(latest_ckpt),
        "lcs_score": calculate_lcs(model, task, device=device),
        "induction_score": probe_induction_heads(model, task, device),
        **evaluate_icl_scaling(model, task, device=device)
    }
    
    save_path = f"results/evaluation/suite_{task}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_data = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f: all_data = json.load(f)
    all_data[model_scale] = results
    with open(save_path, "w") as f: json.dump(all_data, f, indent=2)
    
    print(f"Results saved to {save_path}")
    for k, v in results.items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=['tt-8k', 'tt-26k', 'tt-150k'])
    parser.add_argument("--tasks", nargs="+", default=['addition', 'mapping', 'decoding'])
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for task in args.tasks:
        for cfg in args.configs:
            run_suite(cfg, task, device)
