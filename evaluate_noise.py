import os
import json
import torch
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.configs.model_configs import config
from src.model.tiny_transformer import TinyTransformer


def evaluate_noise_robustness(model, task, stoi, itos, noise_ratios=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                              n_samples=500, n_context=5, batch_size=32, max_len=128, device='cpu'):
    """
    Evaluate model performance with different ratios of noisy/wrong examples in the context.
    
    Args:
        model: The transformer model to evaluate
        task: Task name ('addition', 'mapping', 'decoding')
        stoi: String to index mapping
        itos: Index to string mapping
        noise_ratios: List of noise ratios to test (0.0 to 1.0)
        n_samples: Number of test samples per noise ratio
        n_context: Number of in-context examples (shots)
        batch_size: Batch size for evaluation
        max_len: Maximum sequence length
        device: Device to run on
        
    Returns:
        Dictionary mapping noise ratio to accuracy
    """
    model.eval()
    
    def tokenize(s):
        return [stoi.get(ch, 0) for ch in s]
    
    results = {}
    
    print(f"\nEvaluating Noise Robustness for '{task}' with {n_context}-shot")
    print(f"Noise Ratios: {noise_ratios}")
    print(f"Samples per ratio: {n_samples}")
    
    for ratio in noise_ratios:
        print(f"\n--- Testing Noise Ratio: {ratio:.1f} ---")
        
        # Generate test data with noisy context examples
        test_data = SyntheticICLDataset(
            task=task, 
            n_samples=n_samples, 
            n_context=n_context,
            noise_ratio=ratio
        ).build_dataset(return_answer=True)

        total_correct = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc=f"Evaluating Ratio {ratio:.1f}"):
                batch = test_data[i:i+batch_size]
                if not batch:
                    break
                
                # Generate predictions for accuracy
                for j in range(len(batch)):
                    prompt_str = batch[j]['prompt']
                    p_tokens = tokenize(prompt_str)
                    
                    # Greedy generation
                    curr_tokens = p_tokens[:]
                    for _ in range(20):
                        if len(curr_tokens) >= max_len:
                            break
                        
                        inp_tensor = torch.tensor([curr_tokens], dtype=torch.long).to(device)
                        out = model(inp_tensor)
                        next_token = out[0, -1, :].argmax().item()
                        curr_tokens.append(next_token)
                        if next_token == 0:
                            break
                    
                    # Decode generated text
                    gen_text = ''.join([itos.get(t, '') for t in curr_tokens[len(p_tokens):] if t != 0])
                    
                    # Extract answer based on task
                    if task == "addition" or task == "mapping":
                        match = re.search(r'\d+', gen_text)
                    elif task == "decoding":
                        match = re.search(r'[A-Z]', gen_text)
                    else:
                        match = None
                    
                    predicted = match.group(0) if match else gen_text.strip()
                    expected = batch[j]['answer'].strip()
                    
                    total_predictions += 1
                    if predicted == expected:
                        total_correct += 1
        
        # Calculate metrics
        accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0.0
        
        results[ratio] = accuracy
        print(f"Ratio {ratio:.1f}: Accuracy = {accuracy:.2f}% ({total_correct}/{total_predictions})")
    
    return results

def plot_noise_robustness(task_results, task):
    """Plot noise robustness results."""
    plt.figure(figsize=(10, 6))
    
    for configkey, results in task_results.items():
        ratios = list(results.keys())
        accuracies = list(results.values())
        
        plt.plot(ratios, accuracies, marker='o', linewidth=2, label=configkey)
    
    plt.title(f"ICL Noise Robustness: {task.capitalize()}", fontsize=14)
    plt.xlabel("Noise Ratio (Proportion of context examples with wrong answers)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Model Scale")
    
    # Save the plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/noise_robustness_{task}.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to results/plots/noise_robustness_{task}.png")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configuration
    task = 'mapping'
    # Evaluate across a few scales to see if larger models are more robust
    configs_to_test = ['tt-1k', 'tt-14k', 'tt-150k', 'tt-800k']
    noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_samples = 200
    n_context = 5
    
    all_results = {}
    
    for configkey in configs_to_test:
        print(f"\n{'='*60}")
        print(f"Testing Config: {configkey}")
        print(f"{'='*60}")
        
        model_cfg = config(configkey)
        max_len = model_cfg.get('max_len', 256)
        
        # Build vocabulary from training data
        print("Building vocabulary...")
        dummy_ds = SyntheticICLDataset(task=task, n_samples=1000, n_context=5)
        train_data = dummy_ds.build_dataset(return_answer=True)
        all_text = ''.join([item['prompt'] + ' ' + item['answer'] for item in train_data])
        vocab = sorted(set(all_text))
        stoi = {ch: i+1 for i, ch in enumerate(vocab)}
        itos = {i+1: ch for i, ch in enumerate(vocab)}
        stoi['<pad>'] = 0
        itos[0] = '<pad>'
        
        # Initialize model
        model = TinyTransformer(
            vocab_size=len(stoi),
            dim=model_cfg['dim'],
            depth=model_cfg['depth'],
            n_heads=model_cfg['n_heads'],
            stoi=stoi,
            itos=itos,
            configkey=configkey,
            mlp_dim=model_cfg['mlp_dim'],
            dropout=0.0,
            max_len=max_len,
            use_rope=True
        ).to(device)
        
        # Load best checkpoint
        checkpoint_dir = f"checkpoints/{task}/{configkey}/"
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory {checkpoint_dir} not found. Skipping {configkey}.")
            continue
            
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}. Skipping {configkey}.")
            continue
            
        def extract_step(ckpt):
            m = re.search(r"model-step-(\d+)", ckpt)
            return int(m.group(1)) if m else 0
            
        latest_ckpt = max(checkpoints, key=extract_step)
        ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
        
        print(f"Loading checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # Evaluate
        results = evaluate_noise_robustness(
            model=model,
            task=task,
            stoi=stoi,
            itos=itos,
            noise_ratios=noise_ratios,
            n_samples=n_samples,
            n_context=n_context,
            max_len=max_len,
            device=device
        )
        
        all_results[configkey] = results
        
    print(f"\n{'='*60}")
    print("Summary of Results:")
    for ck, res in all_results.items():
        print(f"{ck}: {res}")
        
    if all_results:
        # Save results to JSON
        os.makedirs("results/evaluation", exist_ok=True)
        with open(f"results/evaluation/noise_robustness_{task}.json", "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved results to results/evaluation/noise_robustness_{task}.json")
            
        plot_noise_robustness(all_results, task)
