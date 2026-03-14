import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

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
    def __init__(self, vocab_size, dim, depth, n_heads, stoi, itos, configkey, mlp_dim=512, max_len=128, dropout=0.1, use_rope=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.stoi = stoi
        self.itos = itos
        self.configkey = configkey
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
        if not self.rope:
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




def train(model, data_fn, steps, batch_size, lr=3e-4, weight_decay=0.1, device='cpu', checkpoint_path=None, checkpoint_interval=1000, start_step=0, log_file=None, test_data_fn=None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Linear warmup + cosine decay
    warmup_steps = 1000
    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)

    progress = tqdm(range(start_step, steps), desc="Training", total=steps, initial=start_step, ncols=100, leave=True)
    
    metrics = []

    # Early stopping might hinder generalization due to grokking (especially in larger models)
    # # Early stopping init
    # best_loss = float('inf')
    # loss_patience = 0
    # patience_limit = 1000
    # min_delta = 1e-4
    
    # Load existing logs if resuming
    if start_step > 0 and log_file and os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                metrics = json.load(f)
            print(f"Resumed logging from step {start_step}, loaded {len(metrics)} metrics points.")
        except Exception as e:
            print(f"Warning: Could not load existing log file: {e}")

    
    for step in progress:
        inp, tgt = data_fn(batch_size)
        inp, tgt = inp.to(device), tgt.to(device)

        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=0, label_smoothing=0.1)
        if step == start_step:
            progress.set_postfix({"starting_loss": f"{loss.item():.4f}"})
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        opt.step()
        scheduler.step()

        # # Early Stopping check
        # current_loss = loss.item()
        # if current_loss < best_loss - min_delta:
        #     best_loss = current_loss
        #     loss_patience = 0
        # else:
        #     loss_patience += 1
            
        # if loss_patience >= patience_limit:
        #     print(f"Early stopping triggered at step {step}. Loss did not improve by {min_delta} for {patience_limit} steps.")
        #     break

        # Log metrics for visualization later on
        if log_file:
            metrics.append({
                "step": step,
                "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0]
            })
            
            if test_data_fn and step % 100 == 0:
                model.eval()
                with torch.no_grad():
                    t_inp, t_tgt = test_data_fn(batch_size)
                    t_inp, t_tgt = t_inp.to(device), t_tgt.to(device)
                    t_logits = model(t_inp)
                    test_loss = F.cross_entropy(t_logits.reshape(-1, t_logits.size(-1)), t_tgt.reshape(-1), ignore_index=0, label_smoothing=0.1)
                    metrics[-1]["test_loss"] = test_loss.item()
                model.train()
            
            # metrics.append(metrics.pop())
            if step % 100 == 0 and step > 0:
                 with open(log_file, "w") as f:
                    json.dump(metrics, f)

        # Save model every checkpoint_interval steps
        if checkpoint_path is not None and step % checkpoint_interval == 0 and step > 0:
            directory = os.path.dirname(checkpoint_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path + f"model-step-{step}.pt")

        if step % 100 == 0:    
            progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

    # Save final logs
    if log_file:
        with open(log_file, "w") as f:
            json.dump(metrics, f)
            
    return model
