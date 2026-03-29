# Project Tiny Transformer: Exploring In-Context Learning

This project investigates the emergence of **In-Context Learning (ICL)** and algorithmic reasoning in small-scale Transformer models (2k to 3M parameters).

## 🚀 Quick Start

### 1. Installation
The project automatically handles virtual environment redirection.
```bash
pip install torch tqdm matplotlib numpy seaborn
```

### 2. Training
Use `experiment.py` to train models on synthetic tasks. Supports infinite data, automatic LR finding, and convergence-based early stopping.
```powershell
# Train until 100% ICL accuracy or convergence
python experiment.py --configs tt-26k tt-141k --tasks mapping arithmetic_symbolic --train_until_convergence --infinite
```

### 3. Evaluation
Run the modular evaluation suite to calculate LCS, Induction Scores, and OOD robustness.
```bash
python src/evaluation/suite.py --configs tt-26k tt-141k --tasks mapping arithmetic_symbolic
```

---

## 🏗️ Architectural Scaling
- **Linear Scaling Law**: Models range from **~2k to ~3.2M parameters** with depths capped at 4 layers for GPU efficiency.
- **RoPE**: Uses Rotary Positional Embeddings for optimal sequence handling.

## 📊 Key Features
- **ICL-Based Early Stopping**: Halts training once 100% ICL accuracy is achieved or patience is met.
- **LR Finder**: Automatically calibrates learning rates before training.
- **Modular Evaluation**: Decoupled engine for Noise Robustness, Label Flipping, and Induction Head probing.

---

## 🏗️ Project Structure
- `src/`
  - `model/`: Core `TinyTransformer` architecture.
  - `dataset/`: Generators for Addition, Arithmetic, Mapping, and Decoding.
  - `evaluation/`: Modular engine for metrics and OOD probes.
  - `lr_finder.py`: Learning rate range tester.
- `scripts/`: Specialized research scripts (Induction probing, Scaling visualization).
- `results/`: JSON reports and task-specific plots (`results/plots/[task]/`).

---
