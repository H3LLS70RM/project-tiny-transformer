import argparse
import os
import glob
import re
import torch
import random
import math
import json
import torch.nn.functional as F
from tqdm import tqdm

from src.model.tiny_transformer import TinyTransformer
from src.configs.model_configs import config
from src.dataset.synthetic_dataset import SyntheticICLDataset
from src.utils import set_seed
from src.lr_finder import LRFinder

# Optional imports for evaluation
try:
    from src.evaluation.suite import evaluate_model, run_suite, probe_induction_heads
    from src.plots.visualize import plot_training_progress, plot_icl_results
    HAS_EVAL = True
except ImportError:
    HAS_EVAL = False

def main():
    parser = argparse.ArgumentParser(description="Train TinyTransformer models on synthetic ICL tasks.")
    parser.add_argument("--configs", type=str, nargs="+", default=['tt-8k', 'tt-14k', 'tt-26k'], help="Model scales to train.")
    parser.add_argument("--tasks", type=str, nargs="+", default=['addition', 'mapping', 'decoding'], help="Tasks to train on.")
    parser.add_argument("--infinite", action="store_true", help="Use infinite on-the-fly data generation.")
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embeddings.")
    parser.add_argument("--no_eval", action="store_true", help="Disable automated evaluation after training.")
    parser.add_argument("--no_viz", action="store_true", help="Disable plotting after training.")
    parser.add_argument("--icl_early_stopping", action="store_true", default=True, help="Use ICL accuracy for early stopping.")
    parser.add_argument("--steps", type=int, default=None, help="Override default training steps.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(1337)
    print(f"Device: {device}")

    for configkey in args.configs:
        model_cfg = config(configkey)
        steps = args.steps if args.steps else model_cfg['train_steps']
        batch_size = model_cfg['batch_size']
        lr = model_cfg['learning_rate']
        max_len = model_cfg.get('max_len', 256)
        
        for task in args.tasks:
            checkpoint_path = f"checkpoints/{task}/{configkey}/"
            os.makedirs(checkpoint_path, exist_ok=True)
            print(f"\n--- Training {configkey} | Task: {task} | Infinite: {args.infinite} ---")
            
            # Setup Datasets
            n_samples = 100
            n_context = (2, 5) if task == "mapping" else (1, 5)
            
            # Build vocab helper
            dummy_ds = SyntheticICLDataset(task=task, n_samples=5000, n_context=n_context)
            vocab_data = dummy_ds.build_dataset(return_answer=True)
            all_text = ''.join([item['prompt'] + item['answer'] for item in vocab_data])
            vocab = sorted(set(all_text))
            stoi = {ch: i+1 for i, ch in enumerate(vocab)}
            itos = {i+1: ch for i, ch in enumerate(vocab)}
            stoi['<pad>'] = 0
            itos[0] = '<pad>'
            
            def tokenize(s):
                return [stoi.get(ch, 0) for ch in s]

            # Data functions
            if args.infinite:
                train_ds = SyntheticICLDataset(task=task, n_samples=1, n_context=n_context)
                test_ds = SyntheticICLDataset(task=task, n_samples=n_samples, n_context=n_context)
                test_data_list = test_ds.build_dataset(return_answer=True)

                def data_fn(bs):
                    full_sequences, masks = [], []
                    for _ in range(bs):
                        item = train_ds.generate_sequence(return_answer=True)
                        p_tokens, a_tokens = tokenize(item['prompt']), tokenize(item['answer'])
                        full = (p_tokens + a_tokens)[:max_len]
                        full_sequences.append(full)
                        m = [0] * (len(full) - 1)
                        for i in range(max(0, len(p_tokens)-1), len(m)): m[i] = 1
                        masks.append(m)
                    full_sequences = [s + [0]*(max_len - len(s)) for s in full_sequences]
                    masks = [m + [0]*(max_len - 1 - len(m)) for m in masks]
                    return torch.tensor(full_sequences)[:, :-1], torch.tensor(full_sequences)[:, 1:], torch.tensor(masks, dtype=torch.bool)

                def test_data_fn(bs):
                    batch = random.choices(test_data_list, k=bs)
                    full_sequences, masks = [], []
                    for item in batch:
                        p_tokens, a_tokens = tokenize(item['prompt']), tokenize(item['answer'])
                        full = (p_tokens + a_tokens)[:max_len]
                        full_sequences.append(full)
                        m = [0] * (len(full) - 1)
                        for i in range(max(0, len(p_tokens)-1), len(m)): m[i] = 1
                        masks.append(m)
                    full_sequences = [s + [0]*(max_len - len(s)) for s in full_sequences]
                    masks = [m + [0]*(max_len - 1 - len(m)) for m in masks]
                    return torch.tensor(full_sequences)[:, :-1], torch.tensor(full_sequences)[:, 1:], torch.tensor(masks, dtype=torch.bool)
            else:
                train_data = SyntheticICLDataset(task=task, n_samples=100000, n_context=n_context).build_dataset(return_answer=True)
                test_data = SyntheticICLDataset(task=task, n_samples=100, n_context=n_context).build_dataset(return_answer=True)

                def data_fn(bs):
                    items = random.choices(train_data, k=bs)
                    seqs = [tokenize(it['prompt'] + it['answer'])[:max_len] for it in items]
                    seqs = [s + [0]*(max_len - len(s)) for s in seqs]
                    t = torch.tensor(seqs)
                    return t[:, :-1], t[:, 1:]

                def test_data_fn(bs):
                    items = random.choices(test_data, k=bs)
                    seqs = [tokenize(it['prompt'] + it['answer'])[:max_len] for it in items]
                    seqs = [s + [0]*(max_len - len(s)) for s in seqs]
                    t = torch.tensor(seqs)
                    return t[:, :-1], t[:, 1:]

            # Model
            model = TinyTransformer(
                vocab_size=len(stoi), dim=model_cfg['dim'], depth=model_cfg['depth'],
                n_heads=model_cfg['n_heads'], stoi=stoi, itos=itos, configkey=configkey,
                mlp_dim=model_cfg['mlp_dim'], dropout=model_cfg.get('dropout', 0.2),
                max_len=max_len, use_rope=not args.no_rope
            )

            # Checkpoint loading
            latest_step = 0
            checkpoints = glob.glob(f"{checkpoint_path}model-step-*.pt")
            if checkpoints:
                latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r"model-step-(\d+)", x).group(1)))
                latest_step = int(re.search(r"model-step-(\d+)", latest_ckpt).group(1))
                print(f"Resuming from {latest_ckpt}")
                model.load_state_dict(torch.load(latest_ckpt, map_location=device))

            log_file = f"results/logs/{task}/{configkey}/training_log{'_infinite' if args.infinite else ''}.json"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            icl_eval_fn = lambda t=task, s=stoi, i=itos: model.quick_icl_eval(t, s, i, device=device)

            # Find optimal learning rate (only if starting fresh)
            if latest_step == 0:
                print(f"Running LR finder for {configkey} | {task}...")
                _tmp_opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
                finder = LRFinder(model, _tmp_opt, device=device)
                suggested_lr, lr_history = finder.range_test(
                    data_fn=data_fn,
                    start_lr=1e-7,
                    end_lr=1.0,
                    num_iter=100,
                    batch_size=batch_size,
                )
                # Save the LR finder plot
                lr_plot_path = f"results/lr_finder/{task}/{configkey}_lr_finder.png"
                finder.plot(save_path=lr_plot_path, show=False)
                print(f"LR finder complete. Using lr={suggested_lr:.2e} (config default was {lr:.2e})")
                lr = suggested_lr
            else:
                print(f"Resuming from step {latest_step}, skipping LR finder.")

            model.fit(
                data_fn=data_fn, steps=steps, batch_size=batch_size,
                lr=lr, device=device, checkpoint_path=checkpoint_path,
                start_step=latest_step, log_file=log_file, test_data_fn=test_data_fn,
                icl_early_stopping=args.icl_early_stopping, icl_eval_fn=icl_eval_fn
            )

            torch.save(model.state_dict(), checkpoint_path + f"model-step-{steps}.pt")

            if not args.no_eval and HAS_EVAL:
                print("Running evaluation...")
                eval_data = test_data_list if args.infinite else test_data
                evaluate_model(model, eval_data, task, batch_size=batch_size, max_len=max_len, device=device, model_scale=configkey)
                run_suite(configkey, task, device=device)

    if not args.no_viz and HAS_EVAL:
        print("Generating plots...")
        plot_training_progress()
        plot_icl_results()

if __name__ == "__main__":
    main()
