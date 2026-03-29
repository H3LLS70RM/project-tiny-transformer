import os
import sys
# Add project root to sys.path for robust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.model.tiny_transformer import TinyTransformer
from src.configs.model_configs import config
from src.evaluation.suite import evaluate_model, save_raw_predictions
from src.evaluation.probes import evaluate_generalization
from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.utils import get_task_vocab
import glob
import re

def run_generalization_test(model_scale='tt-3000k', task='addition', checkpoint_dir=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    cfg = config(model_scale)
    
    # Setup vocab
    stoi, itos = get_task_vocab(task)
    
    model = TinyTransformer(
        vocab_size=len(stoi), dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'],
        stoi=stoi, itos=itos, configkey=model_scale, mlp_dim=cfg['mlp_dim'],
        max_len=cfg.get('max_len', 256), use_rope=True
    ).to(device)
    
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/{task}/{model_scale}/"
    ckpts = glob.glob(f"{checkpoint_dir}model-step-*.pt")
    if not ckpts:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    latest_ckpt = max(ckpts, key=lambda x: int(re.search(r"model-step-(\d+)", x).group(1)))
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))

    # Run modular generalization test
    results = evaluate_generalization(model, task, evaluate_model, n_samples=500, device=device)
    print(f"\nGeneralization Results for {model_scale}:")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")

    # Log OOD raw predictions for debugging (V2 Defaults)
    if task == "addition":
        ood_logging_ds = SyntheticICLDataset(task=task, n_samples=20, n_context=5, is_ood=True, digit_level=True).build_dataset(return_answer=True)
    elif task == "mapping":
        ood_logging_ds = SyntheticICLDataset(task=task, n_samples=20, n_context=5, is_ood=True, mapping_ood_type='extrapolation').build_dataset(return_answer=True)
    elif task == "decoding":
        ood_logging_ds = SyntheticICLDataset(task=task, n_samples=20, n_context=5, is_ood=True, decoding_reversal=True).build_dataset(return_answer=True)
    else:
        ood_logging_ds = SyntheticICLDataset(task=task, n_samples=20, n_context=5, is_ood=True).build_dataset(return_answer=True)
    save_raw_predictions(model, ood_logging_ds, task, model_scale, device=device, suffix="ood")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=['addition', 'decoding', 'mapping'])
    parser.add_argument("--configs", nargs="+", default=['tt-1k', 'tt-4k', 'tt-8k', 'tt-14k', 'tt-26k', 'tt-50k', 'tt-150k', 'tt-800k', 'tt-3000k'])
    args = parser.parse_args()
    
    for task in args.tasks:
        for model_scale in args.configs:
            run_generalization_test(model_scale=model_scale, task=task)
