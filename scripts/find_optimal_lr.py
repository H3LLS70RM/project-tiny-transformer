from src.model.tiny_transformer import TinyTransformer
from src.configs.model_configs import config
from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.lr_finder import LRFinder
import torch
import random
import os
from src.utils import set_seed

# Setup
set_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scale = 'tt-50k' # Using a small scale for quick LR finding
task = 'addition'
print(f"Using device: {device}")

# 1. Prepare Data
print("Preparing dataset...")
# Using a smaller dataset for LR finding is usually fine
train_data = SyntheticICLDataset(task=task, n_samples=10000, n_context=5).build_dataset(return_answer=True)

# Build vocab
all_text = ''.join([item['prompt'] + item['answer'] for item in train_data])
vocab = sorted(set(all_text))
stoi = {ch: i+1 for i, ch in enumerate(vocab)}
stoi['<pad>'] = 0
max_len = 256

def tokenize(s):
    return [stoi.get(ch, 0) for ch in s]

def data_fn(batch_size):
    batch = random.choices(train_data, k=batch_size)
    full_sequences = [tokenize(item['prompt'] + item['answer'])[:max_len] for item in batch]
    full_sequences = [seq + [0]*(max_len - len(seq)) if len(seq) < max_len else seq for seq in full_sequences]
    tensor = torch.tensor(full_sequences, dtype=torch.long)
    inp = tensor[:, :-1]
    tgt = tensor[:, 1:]
    return inp, tgt

# 2. Initialize Model
print(f"Initializing model (scale: {scale})...")
model_cfg = config(scale)
model = TinyTransformer(
    vocab_size=len(stoi),
    dim=model_cfg['dim'],
    depth=model_cfg['depth'],
    n_heads=model_cfg['n_heads'],
    stoi=stoi,
    itos={i: ch for ch, i in stoi.items()},
    mlp_dim=model_cfg['mlp_dim'],
    dropout=model_cfg.get('dropout', 0.1),
    max_len=max_len
)
model.to(device)

# 3. Setup Optimizer
# Using standard AdamW as in training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)

# 4. Run LR Finder
print("Running LR Finder...")
lr_finder = LRFinder(model, optimizer, device=device)
lr_finder.range_test(
    data_fn=data_fn,
    start_lr=1e-6,
    end_lr=10,
    num_iter=100,
    batch_size=model_cfg['batch_size']
)

# 5. Plot
save_path = "results/lr_finder_plot.png"
lr_finder.plot(save_path=save_path, show=False)
print(f"Done! Check {save_path}")
