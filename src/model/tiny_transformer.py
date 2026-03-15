import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import random

from src.dataset.synthetic_dataset import SyntheticICLDataset

# RoPE Implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        inv_freq = 1.0 / (max_seq_len ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.curr_seq_len = 0
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
            
        if self.cached_cos is None or seq_len > self.curr_seq_len:
            self.curr_seq_len = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()[None, None, :, :] # [1, 1, T, D] for broadcasting with [B, H, T, D]
            self.cached_sin = emb.sin()[None, None, :, :]
            
        return self.cached_cos[:, :, :seq_len, :], self.cached_sin[:, :, :seq_len, :]

def rotate_half(x):
    # Split at the last dimension (head_dim)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, T, D]
    # cos, sin: [1, 1, T, D]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# Transformer with RoPE

class TinySelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # store attention weights for interpretability
        self.attn_weights = None

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, D = x.shape

        qkv = self.qkv(x)                        # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        self.attn_weights = att.detach()    # save for external inspection

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)

class TinyTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = TinySelfAttention(dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.dropout(self.attn(self.ln1(x), rope_cos, rope_sin))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, n_heads, stoi=None, itos=None, configkey=None, mlp_dim=512, max_len=128, dropout=0.1, use_rope=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.stoi = stoi
        self.itos = itos
        self.configkey = configkey
        self.rope = None
        if(use_rope):
            self.rope = RotaryEmbedding(dim // n_heads, max_seq_len=max_len)
        else:
            self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim) * 0.01)
        
        self.vocab_size = vocab_size
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(dim, n_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) # [B, T, D]
        if self.rope is None:
            # Use T from idx for slicing, ensuring it matches the sequence dimension of x
            x = x + self.pos_emb[:, :T, :]
                
        # Calculate RoPE embeddings for this sequence length
        if self.rope:
            cos, sin = self.rope(x, seq_len=T)
        else:
            cos, sin = None, None

        for blk in self.blocks:
            x = blk(x, rope_cos=cos, rope_sin=sin)

        x = self.ln_f(x)
        return self.head(x)

    def get_attention_maps(self):
        """Returns attention weights for all layers."""
        maps = []
        for blk in self.blocks:
            maps.append(blk.attn.attn_weights)
        return maps

    def quick_icl_eval(self, task, stoi, itos, n_samples=50, n_shots=3, max_len=256, device='cpu'):
        """
        Lightweight ICL evaluation: generates n_samples few-shot prompts,
        does greedy decoding, and returns exact-match accuracy (0.0 - 1.0).
        """
        self.eval()
        ds = SyntheticICLDataset(task=task, n_samples=n_samples, n_context=n_shots)
        data = ds.build_dataset(return_answer=True)

        correct = 0
        with torch.no_grad():
            for item in data:
                p_tokens = [stoi.get(ch, 0) for ch in item['prompt']]
                curr = p_tokens[:]
                stop_id = stoi.get('\n', -1)
                for _ in range(20):
                    if len(curr) >= max_len:
                        break
                    inp = torch.tensor([curr], dtype=torch.long).to(device)
                    logits = self(inp)
                    nxt = logits[0, -1, :].argmax().item()
                    curr.append(nxt)
                    if nxt == 0 or nxt == stop_id:
                        break
                gen = ''.join([itos.get(t, '') for t in curr[len(p_tokens):] if t != 0])
                pred = gen.split('\n')[0].strip()
                if pred == item['answer'].strip():
                    correct += 1
        self.train()
        return correct / max(len(data), 1)

    def fit(self, data_fn, steps, batch_size, lr=3e-4, weight_decay=0.1, device='cpu', 
            checkpoint_path=None, checkpoint_interval=1000, start_step=0, log_file=None, 
            test_data_fn=None, icl_early_stopping=False, icl_eval_fn=None, 
            eval_interval=500, log_interval=250):
        """
        Core training loop with logging, early stopping, and checkpointing.
        """
        self.to(device)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
            
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        warmup_steps = 1000
        def get_lr(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)
        metrics = []

        # Early stopping init
        best_loss = float('inf')
        loss_patience = 0
        patience_limit = 2000
        min_delta = 1e-4

        # ICL-based early stopping init
        best_icl_acc = -1.0
        icl_patience = 0
        icl_patience_limit = max(1, 3000 // eval_interval)
        icl_check_interval = eval_interval
        icl_min_step = 2000 
        
        if start_step > 0 and log_file and os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    metrics = json.load(f)
                print(f"Resumed logging from step {start_step}, loaded {len(metrics)} points.")
            except Exception as e:
                print(f"Warning: Could not load existing log file: {e}")

        current_icl_acc = 0.0
        progress = tqdm(range(start_step, steps), desc="Training", total=steps, initial=start_step, leave=True)
        early_stop = False
        early_stop_step = steps

        for step in progress:
            batch_data = data_fn(batch_size)
            if len(batch_data) == 3:
                inp, tgt, mask = batch_data
                mask = mask.to(device)
            else:
                inp, tgt = batch_data
                mask = None
                
            inp, tgt = inp.to(device), tgt.to(device)
            logits = self(inp)
            
            if mask is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=0, label_smoothing=0.1)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    correct = (pred == tgt) & mask
                    total_masked = mask.sum()
                    accuracy = correct.sum().float() / total_masked if total_masked > 0 else torch.tensor(0.0, device=device)
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=0, label_smoothing=0.1)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    correct = (pred == tgt)
                    accuracy = correct.float().mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            scheduler.step()

            monitor_loss = loss.item()
            monitor_acc = accuracy.item()
            
            if test_data_fn and step % eval_interval == 0:
                self.eval()
                with torch.no_grad():
                    t_data = test_data_fn(batch_size)
                    if len(t_data) == 3:
                        t_inp, t_tgt, t_mask = t_data
                        t_mask = t_mask.to(device)
                    else:
                        t_inp, t_tgt = t_data
                        t_mask = None
                    
                    t_inp, t_tgt = t_inp.to(device), t_tgt.to(device)
                    t_logits = self(t_inp)
                    test_loss = F.cross_entropy(t_logits.reshape(-1, t_logits.size(-1)), t_tgt.reshape(-1), ignore_index=0, label_smoothing=0.1)
                    monitor_loss = test_loss.item()
                    
                    t_pred = t_logits.argmax(dim=-1)
                    if t_mask is not None:
                        t_correct = (t_pred == t_tgt) & t_mask
                        t_total = t_mask.sum()
                        monitor_acc = (t_correct.sum().float() / t_total).item() if t_total > 0 else 0.0
                    else:
                        monitor_acc = (t_pred == t_tgt).float().mean().item()

                self.train()
                
                if monitor_loss < best_loss - min_delta:
                    best_loss = monitor_loss
                    loss_patience = 0
                else:
                    loss_patience += 1
                    
                if loss_patience >= (patience_limit // eval_interval):
                    early_stop = True
                    early_stop_step = step
                    break

            if icl_early_stopping and icl_eval_fn and step >= icl_min_step and step % icl_check_interval == 0:
                current_icl_acc = icl_eval_fn()
                if current_icl_acc > best_icl_acc:
                    best_icl_acc = current_icl_acc
                    icl_patience = 0
                else:
                    icl_patience += 1
                    
                if icl_patience >= icl_patience_limit:
                    early_stop = True
                    early_stop_step = step
                    break

            if log_file and step % log_interval == 0 and step > 0:
                metric_entry = {
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "accuracy": round(accuracy.item(), 4),
                    "icl_accuracy": round(current_icl_acc, 4),
                    "lr": round(scheduler.get_last_lr()[0], 6)
                }
                if test_data_fn:
                    metric_entry["test_loss"] = round(monitor_loss, 4)
                    metric_entry["test_accuracy"] = round(monitor_acc, 4)
                metrics.append(metric_entry)
                with open(log_file, "w") as f:
                    json.dump(metrics, f)

            if checkpoint_path and step % checkpoint_interval == 0 and step > 0:
                torch.save(self.state_dict(), checkpoint_path + f"model-step-{step}.pt")

            if step % 50 == 0:    
                progress.set_postfix({
                    "loss": f"{loss.item():.3f}", 
                    "acc": f"{accuracy.item():.3f}", 
                    "icl": f"{current_icl_acc:.3f}", 
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

        if early_stop:
            print(f"\nEarly stopping triggered at step {early_stop_step}.")
        if log_file:
            with open(log_file, "w") as f:
                json.dump(metrics, f)
        return self