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

def quick_icl_eval(model, task, stoi, itos, n_samples=50, n_shots=3, max_len=256, device='cpu'):
    """
    Lightweight ICL evaluation: generates n_samples few-shot prompts,
    does greedy decoding, and returns exact-match accuracy (0.0 - 1.0).
    """
    model.eval()
    ds = SyntheticICLDataset(task=task, n_samples=n_samples, n_context=n_shots)
    data = ds.build_dataset(return_answer=True)

    correct = 0
    with torch.no_grad():
        for item in data:
            p_tokens = [stoi.get(ch, 0) for ch in item['prompt']]
            curr = p_tokens[:]
            for _ in range(20):
                if len(curr) >= max_len:
                    break
                inp = torch.tensor([curr], dtype=torch.long).to(device)
                logits = model(inp)
                nxt = logits[0, -1, :].argmax().item()
                curr.append(nxt)
                if nxt == 0:
                    break
            gen = ''.join([itos.get(t, '') for t in curr[len(p_tokens):] if t != 0])
            if task in ('addition', 'mapping'):
                m = re.search(r'\d+', gen)
            elif task == 'decoding':
                m = re.search(r'[A-Z]', gen)
            else:
                m = None
            pred = m.group(0) if m else gen.strip()
            if pred == item['answer'].strip():
                correct += 1
    model.train()
    return correct / max(len(data), 1)

def train(model, data_fn, steps, batch_size, lr=3e-4, weight_decay=0.1, device='cpu', 
          checkpoint_path=None, checkpoint_interval=1000, start_step=0, log_file=None, 
          test_data_fn=None, icl_early_stopping=False, icl_eval_fn=None, 
          eval_interval=500, log_interval=250):
    
    model.to(device)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if checkpoint_path:
        os.makedirs(checkpoint_path, exist_ok=True)
        
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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
        # data_fn optionally returns a mask
        batch_data = data_fn(batch_size)
        if len(batch_data) == 3:
            inp, tgt, mask = batch_data
            mask = mask.to(device)
        else:
            inp, tgt = batch_data
            mask = None
            
        inp, tgt = inp.to(device), tgt.to(device)

        logits = model(inp)
        
        if mask is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=0, label_smoothing=0.1)
            # Calculate accuracy on the answer part
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

        if step == start_step:
            progress.set_postfix({"starting_loss": f"{loss.item():.4f}"})
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        monitor_loss = loss.item()
        monitor_acc = accuracy.item()
        
        if test_data_fn and step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                t_data = test_data_fn(batch_size)
                if len(t_data) == 3:
                    t_inp, t_tgt, t_mask = t_data
                    t_mask = t_mask.to(device)
                else:
                    t_inp, t_tgt = t_data
                    t_mask = None
                
                t_inp, t_tgt = t_inp.to(device), t_tgt.to(device)
                t_logits = model(t_inp)
                test_loss = F.cross_entropy(t_logits.reshape(-1, t_logits.size(-1)), t_tgt.reshape(-1), ignore_index=0, label_smoothing=0.1)
                monitor_loss = test_loss.item()
                
                t_pred = t_logits.argmax(dim=-1)
                if t_mask is not None:
                    t_correct = (t_pred == t_tgt) & t_mask
                    t_total = t_mask.sum()
                    monitor_acc = (t_correct.sum().float() / t_total).item() if t_total > 0 else 0.0
                else:
                    monitor_acc = (t_pred == t_tgt).float().mean().item()

            model.train()
             
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
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
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
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path + f"model-step-{step}.pt")

        if step % 50 == 0:    
            progress.set_postfix({
                "loss": f"{loss.item():.3f}", 
                "acc": f"{accuracy.item():.3f}", 
                "icl": f"{current_icl_acc:.3f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

    if early_stop:
        print(f"\nEarly stopping triggered at step {early_stop_step}.")
        if icl_early_stopping:
            print(f"ICL Acc stable at {best_icl_acc:.4f} for {icl_patience_limit * icl_check_interval} steps.")
        else:
            print(f"Val Loss stable for {loss_patience*eval_interval} steps.")

    if log_file:
        with open(log_file, "w") as f:
            json.dump(metrics, f)
            
    return model

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
            
            icl_eval_fn = lambda m=model, t=task, s=stoi, i=itos: quick_icl_eval(m, t, s, i, device=device)

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

            train(
                model=model, data_fn=data_fn, steps=steps, batch_size=batch_size,
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
