import torch
import torch.nn.functional as F
import numpy as np
import random
import re

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

def calculate_lcs(model, task, n_samples=250, device='cpu', digit_level=False):
    """Calculates Learning-to-Context Slope (LCS)."""
    stoi = model.stoi
    from src.dataset.synthetic_dataset import SyntheticICLDataset
    dataset = SyntheticICLDataset(task=task, n_samples=n_samples, n_context=1, digit_level=digit_level)
    s_vals, t_vals = [], []
    nl_id = [stoi.get("\n", 0)]
    
    def fmt(v):
        if digit_level:
            return " ".join(str(v))
        return str(v)
    
    for _ in range(n_samples):
        if task in ("addition", "arithmetic", "arithmetic_shuffled", "arithmetic_symbolic"):
            da, db = random.randint(0, 99), random.randint(0, 99)
            if task == "arithmetic_shuffled":
                op = random.choice(["+", "-", "*", "max", "min"])
            else:
                op = "+" if task != "arithmetic_symbolic" else random.choice(list("@#$%^&*?"))
            d_str = f"{fmt(da)} {op} {fmt(db)} = {fmt(da+db)}"
            qa, qb = random.randint(0, 99), random.randint(0, 99)
            q_str, x_str = f"{fmt(qa)} {op} {fmt(qb)} = ", fmt(qa + qb)
        elif task == "mapping":
            # Note: mapping_fn might need to be imported or passed if it's external, 
            # but SyntheticICLDataset has a default one.
            ma, mb = random.randint(1, 10), random.randint(0, 50)
            dx = random.randint(0, 99)
            d_str = f"{dx} -> {dataset.mapping_fn(ma, mb, dx)}"
            qx = random.randint(0, 99)
            q_str, x_str = f"{qx} -> ", str(dataset.mapping_fn(ma, mb, qx))
        else: # decoding fallback
            motif = [random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)]
            d_str = f"{motif[0]} -> {motif[1]}"
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
    """
    Measures Induction Head strength using a task-aware template.

    """
    stoi = model.stoi
    if task in ("addition", "arithmetic", "arithmetic_shuffled", "arithmetic_symbolic", "mapping"):
        keys = [k for k in stoi.keys() if re.match(r'^\d$', k)]
    else:
        keys = [k for k in stoi.keys() if re.match(r'^[A-Z]$', k)]
    
    if len(keys) < 4:
        keys = [k for k in stoi.keys() if len(k) == 1 and k not in ('<pad>', '\n', ' ', '>', '-', '=')]
        
    if len(keys) < 2: return 0.0
    
    # Template: "A -> B \n C -> D \n A -> "
    trigger_char = keys[0]
    target_char = keys[1]
    other_chars = keys[2:] if len(keys) > 2 else keys
    
    if task in ("addition", "arithmetic", "arithmetic_shuffled", "arithmetic_symbolic"):
        if task == "arithmetic_shuffled":
            op = random.choice(["+", "-", "*", "max", "min"])
        else:
            op = "+" if task != "arithmetic_symbolic" else random.choice(list("@#$%^&*?"))
        template_prev = f"{trigger_char} {op} {random.choice(other_chars)} = {target_char}\n"
        template_junk = f"{random.choice(other_chars)} {op} {random.choice(other_chars)} = {random.choice(other_chars)}\n"
        template_query = f"{trigger_char} {op} {random.choice(other_chars)} = "
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
    
    try:
        target_pos = full_text.find(target_char)
    except:
        return 0.0

    with torch.no_grad():
        model(torch.tensor([tokens]).to(device))
        attns = model.get_attention_maps()
    
    scores = []
    for layer_attn in attns:
        scores.append(layer_attn[0, :, -1, target_pos].cpu().numpy())
    
    return float(np.max(scores)) if scores else 0.0

def evaluate_icl_scaling(model, task, evaluate_model_fn, n_samples=100, device='cpu', digit_level=False, rule_diversity=False, hard_icl=False, exclude_family_idx=None):
    """Evaluates accuracy across literal few-shot counts."""
    from src.dataset.synthetic_dataset import SyntheticICLDataset
    results = {}
    for n_shots in [0, 1, 2, 3, 4, 5]:
        ds = SyntheticICLDataset(
            task=task, n_samples=n_samples, n_context=n_shots, 
            digit_level=digit_level, rule_diversity=rule_diversity, 
            hard_icl=hard_icl, exclude_family_idx=exclude_family_idx
        ).build_dataset(return_answer=True)
        res = evaluate_model_fn(model, ds, task, device=device)
        results[f"{n_shots}_shot_accuracy"] = res['exact_match_accuracy']
        results[f"{n_shots}_shot_edit_distance"] = res['avg_edit_distance']
        
        # Per-family breakout for mapping (Diversity Analysis)
        if task == "mapping" and rule_diversity:
            # SyntheticICLDataset.rule_families contains the lambdas
            for f_idx in range(len(SyntheticICLDataset.rule_families)):
                f_ds = SyntheticICLDataset(
                    task=task, n_samples=n_samples, n_context=n_shots, 
                    digit_level=digit_level, rule_diversity=True, 
                    hard_icl=hard_icl, rule_family_idx=f_idx
                ).build_dataset(return_answer=True)
                f_res = evaluate_model_fn(model, f_ds, task, device=device)
                results[f"{n_shots}_shot_family_{f_idx}_accuracy"] = f_res['exact_match_accuracy']
    return results
