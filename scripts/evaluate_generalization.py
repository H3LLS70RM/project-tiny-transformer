
import torch
from src.model.tiny_transformer_rope import TinyTransformerRoPE
from src.configs.model_configs import config
from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.evaluation.evaluate_model import evaluate_model
import glob
import os
import re

def evaluate_generalization(model_scale='tt-3000k', task='addition', checkpoint_dir=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Model Configuration
    print(f"Loading config for scale '{model_scale}'...")
    cfg = config(model_scale)
    
    # Initialize Model
    print("Building vocabulary from standard distribution...")
    dummy_data = SyntheticICLDataset(task=task, n_samples=1000).build_dataset(return_answer=True)
    all_text = ''.join([item['prompt'] + item['answer'] for item in dummy_data])
    vocab = sorted(set(all_text))
    stoi = {ch: i+1 for i, ch in enumerate(vocab)}
    itos = {i+1: ch for i, ch in enumerate(vocab)}
    stoi['<pad>'] = 0
    itos[0] = '<pad>'
    
    model = TinyTransformerRoPE(
        vocab_size=len(stoi),
        dim=cfg['dim'],
        depth=cfg['depth'],
        n_heads=cfg['n_heads'],
        mlp_dim=cfg['mlp_dim'],
        max_len=cfg['max_len'],
        stoi=stoi,
        itos=itos
    ).to(device)
    
    # Load Checkpoint
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/{task}/{model_scale}/"
        
    pattern = f"{checkpoint_dir}model-step-*.pt"
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # Find latest checkpoint
    def extract_step(ckpt):
        m = re.search(r"model-step-(\d+)", ckpt)
        return int(m.group(1)) if m else 0
    latest_ckpt = max(checkpoints, key=extract_step)
    print(f"Loading checkpoint: {latest_ckpt}")
    
    try:
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Generate datasets
    print("\nGenerating In-Distribution Dataset...")
    in_dist_data = SyntheticICLDataset(
        task=task, 
        n_samples=500, 
        n_context=5,
    ).build_dataset(return_answer=True)
    
    print("Generating Out-of-Distribution Dataset...")
    if task == "addition":
        ood_data = SyntheticICLDataset(
            task=task, 
            n_samples=500, 
            n_context=5,
            addition_range=(100, 110)
        ).build_dataset(return_answer=True)
    elif task == "mapping":
        ood_data = SyntheticICLDataset(
            task=task, 
            n_samples=500, 
            n_context=5,
            mapping_fn = lambda a,b,x: a * b * x + 11,
            mapping_range=(1, 10),
            mapping_b_range=(1, 30)
        ).build_dataset(return_answer=True)
    elif task == "decoding":
        ood_data = SyntheticICLDataset(
            task=task, 
            n_samples=500, 
            n_context=5,
            motif_range=(11, 15)
        ).build_dataset(return_answer=True)

    # Evaluate
    print(f"\nEvaluating In-Distribution (0-99)...")
    evaluate_model(model, in_dist_data, task, device=device, model_scale=model_scale, json_path=f"results/evaluation/gen_in_dist_{task}.json")
    
    print(f"\nEvaluating Out-of-Distribution in {task}...")
    evaluate_model(model, ood_data, task, device=device, model_scale=model_scale, json_path=f"results/evaluation/gen_ood_{task}.json")

if __name__ == "__main__":
    # You can change the scale here or use argparse
    for task in ['mapping']:
        for model_scale in ['tt-26k', 'tt-40k', 'tt-50k', 'tt-150k', 'tt-800k', 'tt-3000k']:
            evaluate_generalization(model_scale=model_scale, task=task)
