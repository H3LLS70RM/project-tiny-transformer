import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
import glob
import re
from src.utils import set_seed
from src.dataset.synthetic_dataset import SyntheticICLDataset
from matplotlib.patches import Rectangle

def probe_induction_heads(model, configkey, task, device):
    """
    Probes the model for induction heads and visualizes them.
    
    Induction Head Logic:
    Sequence: [RND] ... [A] [B] ... [A] -> Model should predict [B]
    Attention: The last [A] (query) should attend to the [B] (key/value) that followed the first [A].
    """
    if task != "decoding":
        print(f"Skipping induction head probing for task '{task}'...")
        return
    print(f"Probing induction heads for scale '{configkey}'...")

    model.eval()
    
    # 2. Generate Induction Sequence
    # Pattern: R1 R2 ... A B ... R3 ... A -> Predict B
    seq_len = 30
    
    stoi = model.stoi

    # Use valid non-special tokens
    valid_tokens = [stoi[c] for c in stoi.keys() if c.strip() and re.match(r'^[A-Z]$', c) ]
    valid_chars = [model.itos[t] for t in valid_tokens]
    print(f"Valid chars: {valid_chars}")
    if len(valid_tokens) < 5:
         # Fallback to digits
         valid_tokens = [stoi[str(i)] for i in range(10) if str(i) in stoi]
    
    print(f"Using valid tokens: {valid_tokens}")

    # Select Trigger and Target tokens first
    trigger_token = valid_tokens[0]
    target_token = valid_tokens[1]
    
    # Create a pool of tokens excluding trigger and target to avoid ambiguity
    pool = [t for t in valid_tokens if t != trigger_token and t != target_token]
    
    # Unique random tokens from the restricted pool
    seq = [random.choice(pool) for _ in range(seq_len)]
    
    print(f"Trigger token: {model.itos[trigger_token]}, Target token: {model.itos[target_token]}")
    
    # Randomize Trigger Pair Position
    trigger_start_idx = random.randint(0, seq_len - 5)
    
    seq[trigger_start_idx] = trigger_token
    seq[trigger_start_idx+1] = target_token
    
    target_pos = trigger_start_idx + 1
    
    # Place 'A' at the very end (Query position)
    seq[-1] = trigger_token

    gen_text = ''.join([model.itos.get(t, '') for t in seq])
    print(f"Generated sequence: {gen_text}")
    
    # Query is the last token
    query_pos = seq_len - 1
    
    print(f"Sequence created. Trigger '{model.itos[trigger_token]}' at pos {trigger_start_idx} and {query_pos}. Target '{model.itos[target_token]}' at pos {target_pos}.")
    
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    
    # 3. Forward Pass & Capture Attention
    with torch.no_grad():
        output = model(input_tensor)
        output_char = model.itos[output[0, -1, :].argmax().item()]
        print(f"Output char: {output_char}")
        attentions = model.get_attention_maps() # List of [B, H, T, T]
    
    # 4. Calculate Scores
    # Score = Attention weight from Query Pos to Target Pos
    
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    
    induction_scores = np.zeros((n_layers, n_heads))
    
    for l in range(n_layers):
        # [B(1), H, T, T]
        attn = attentions[l][0] 
        # shape [H, T, T] -> [H, query_pos, target_pos]
        scores = attn[:, query_pos, target_pos].cpu().numpy()
        induction_scores[l] = scores

    # 5. Plot Heatmap
    save_dir = f"results/plots/induction_heads/{task}"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(induction_scores, annot=True, fmt=".2f", cmap="Reds", yticklabels=[f"L{i}" for i in range(n_layers)], xticklabels=[f"H{i}" for i in range(n_heads)])
    plt.title(f"Induction Head Scores (Scale: {configkey})")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.savefig(f"{save_dir}/induction_head_heatmap_{configkey}.png")
    plt.close()
    print(f"Saved heatmap to {save_dir}/induction_head_heatmap_{configkey}.png")
    
    # 6. Plot Specific Attention Pattern for Max Head
    # Find max head
    max_idx = np.unravel_index(np.argmax(induction_scores), induction_scores.shape)
    best_layer, best_head = max_idx
    print(f"Strongest induction head: Layer {best_layer}, Head {best_head} (Score: {induction_scores[best_layer, best_head]:.4f})")
    
    # Get attention matrix for this head
    # [T, T]
    attn_matrix = attentions[best_layer][0, best_head].cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap="Blues")
    plt.title(f"Attention Pattern - Layer {best_layer} Head {best_head}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    
    # Highlight the induction step
    plt.gca().add_patch(Rectangle((target_pos, query_pos), 1, 1, fill=False, edgecolor='red', lw=2))

    max_attn_idx = np.argmax(attn_matrix[query_pos, :])
    max_attn_val = attn_matrix[query_pos, max_attn_idx]
    
    print(f"  > Expected target pos: {target_pos} (Token: '{model.itos[seq[target_pos]]}')")
    print(f"  > Actual max attention at pos {max_attn_idx} (Token: '{model.itos[seq[max_attn_idx]]}') with value {max_attn_val:.4f}")
    
    # Highlight actual max
    plt.gca().add_patch(Rectangle((max_attn_idx, query_pos), 1, 1, fill=False, edgecolor='green', lw=2, linestyle='--'))
    
    plt.savefig(f"{save_dir}/induction_head_attention_{configkey}.png")
    plt.close()
    print(f"Saved attention map to {save_dir}/induction_head_attention_{configkey}.png")
