
import torch
import os
import json
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.model.tiny_transformer import TinyTransformer
from src.evaluation.evaluate_model import evaluate_model

def verify():
    print("Starting verification...")
    
    # 1. Setup Dummy Model and Data
    vocab_size = 20
    dim = 32
    stoi = {str(i): i for i in range(1, vocab_size)}
    stoi['<pad>'] = 0
    itos = {i: str(i) for i in range(1, vocab_size)}
    itos[0] = '<pad>'
    
    model = TinyTransformer(
        vocab_size=vocab_size,
        dim=dim,
        stoi=stoi,
        itos=itos,
        depth=2,
        n_heads=2,
        mlp_dim=64,
        max_len=20
    )
    
    # Dummy dataset
    dataset = [
        {'prompt': '1 2', 'answer': '3 4'},
        {'prompt': '5 6', 'answer': '7 8'}
    ]
    
    # 2. Run Evaluation
    json_path = "results/evaluation/eval_results_verification.json"
    if os.path.exists(json_path):
        os.remove(json_path)
        
    print(f"Running evaluate_model -> {json_path}")
    evaluate_model(
        model=model,
        dataset=dataset,
        task="verification",
        batch_size=2,
        max_len=10,
        device="cpu",
        json_path=json_path,
        model_scale="test_scale"
    )
    
    # 3. Check Metric
    if not os.path.exists(json_path):
        print("FAILED: Result file not created.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    metric = data.get("verification", {}).get("test_scale", {}).get("avg_edit_distance")
    if metric is None:
        print("FAILED: avg_edit_distance not found in results.")
        print(data)
    else:
        print(f"SUCCESS: avg_edit_distance found: {metric}")

    # 4. Run Plotting Script
    print("\nRunning plotting script...")
    exit_code = os.system("python plot_token_distance_vs_params.py")
    if exit_code == 0:
        print("SUCCESS: Plotting script ran successfully.")
        # Check if plot exists
        if os.path.exists("results/evaluation/token_distance_vs_params.png"):
             print("SUCCESS: Plot file created.")
        else:
             print("FAILED: Plot file not found.")
    else:
        print("FAILED: Plotting script returned error code.")

if __name__ == "__main__":
    verify()
