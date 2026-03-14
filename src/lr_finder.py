import torch
import torch.nn.functional as F
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

class LRFinder:
    def __init__(self, model, optimizer, criterion=None, device='cpu'):
        """
        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            criterion: Loss function. If None, defaults to cross_entropy with ignore_index=0.
            device: 'cpu' or 'cuda'.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.model_state = None
        self.optimizer_state = None

    def range_test(self, data_fn, start_lr=1e-7, end_lr=10, num_iter=100, batch_size=64, smooth_f=0.05, diverge_th=5):
        """
        Runs the learning rate range test.
        
        Args:
            data_fn: Function that returns (inputs, targets) batch.
            start_lr: The starting learning rate.
            end_lr: The ending learning rate.
            num_iter: Number of iterations to run.
            batch_size: Batch size to request from data_fn.
            smooth_f: Smoothing factor for loss (0.0 to 1.0).
            diverge_th: Threshold for divergence (stop if loss > diverge_th * best_loss).
        """
        # Save states to restore later
        print("Saving model and optimizer state...")
        self.model_state = copy.deepcopy(self.model.state_dict())
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        
        # Reset history
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        
        # Calculate multiplier
        r = end_lr / start_lr
        mult = r ** (1 / num_iter)
        
        current_lr = start_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        self.model.to(self.device)
        self.model.train()
        
        print(f"Starting range test: LR {start_lr:.1e} -> {end_lr:.1e}, {num_iter} steps")
        progress = tqdm(range(num_iter), desc="Finding LR", ncols=100)
        
        for i in progress:
            # Get data
            inputs, targets = data_fn(batch_size)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            # Assuming standard causal LM loss structure: [B, T, V] vs [B, T]
            if self.criterion:
                loss = self.criterion(outputs, targets)
            else:
                # Default behavior matching the project's training loop
                loss = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1), ignore_index=0)
            
            # Backward
            loss.backward()
            
            # Gradient clipping (optional but good for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Track loss
            loss_val = loss.item()
            
            # Smooth loss
            if i == 0:
                avg_loss = loss_val
            else:
                avg_loss = smooth_f * loss_val + (1 - smooth_f) * self.history["loss"][-1]
            
            # Check for divergence
            if self.best_loss is None or avg_loss < self.best_loss:
                self.best_loss = avg_loss
            
            if avg_loss > diverge_th * self.best_loss:
                print(f"\nStopping early, loss diverged at step {i} (Loss: {avg_loss:.4f}, Best: {self.best_loss:.4f})")
                break
                
            self.history["loss"].append(avg_loss)
            self.history["lr"].append(current_lr)
            
            # Update LR
            current_lr *= mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            if i % 10 == 0:
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.1e}"})
            
        progress.close()
        
        print("Restoring model and optimizer state...")
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        print("Restoration complete.")
        
    def plot(self, save_path=None, show=True):
        """
        Plots the loss vs learning rate.
        """
        if not self.history["lr"]:
            print("No history to plot. Run range_test first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["lr"], self.history["loss"])
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        # Highlight the minimum gradient or minimum loss point
        min_loss_idx = np.argmin(self.history["loss"])
        min_loss_lr = self.history["lr"][min_loss_idx]
        plt.plot(min_loss_lr, self.history["loss"][min_loss_idx], 'ro', label=f"Min Loss: {min_loss_lr:.1e}")
        
        plt.legend()
        
        if save_path:
            # Ensure directory exists
            directory = os.path.dirname(save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        if show:
            plt.show()

