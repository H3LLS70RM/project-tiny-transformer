# Project Tiny Transformer: ICL Emergence Guidelines

This project explores the emergence of In-Context Learning (ICL) in small-scale Transformer models. It provides a framework for training, evaluating, and visualizing how models of different scales develop the ability to learn from context.

## Architecture

### Core Model: `TinyTransformer`
The project uses a custom, lightweight Transformer architecture designed for research transparency:
- **Type**: Decoder-only (GPT-style) autoregressive Transformer.
- **Components**:
    - **Multi-Head Self-Attention**: With causal masking.
    - **Positional Encoding**: Supports both standard learned Positional Embeddings and **Rotary Positional Embeddings (RoPE)**.
    - **Layer Normalization**: Pre-LN architecture with `GELU` activation in the MLP blocks.
- **Implementation**: Located in `src/model/tiny_transformer.py`.

### Scaling System
Models are scaled using predefined configurations in `src/configs/model_configs.py`, ranging from **1k** to **3000k** parameters. Scaling adjusts:
- Embedding dimension (`dim`)
- Number of layers (`depth`)
- Number of attention heads (`n_heads`)

## Technical Details

### Task Definitions
The model is trained on three distinct synthetic tasks via `SyntheticICLDataset`:
1. **Addition**: Predicting the sum of two numbers (e.g., `12 + 34 = 46`).
2. **Mapping**: Learning a linear function on-the-fly (`x -> ax + b`).
3. **Decoding**: Solving a substitution cipher (e.g., `A -> B`).

### Metrics & Emergence
Beyond standard accuracy, the project uses specialized metrics to measure ICL quality:
- **LCS (Learning-to-Context Slope)**: Quantifies the "emergent" property by correlating learning gain with contextual relevance.
- **Induction Score**: Specifically monitors "Induction Heads" which are responsible for pattern copying.
- **Exact Match Accuracy**: Calculated for 0-shot to N-shot scenarios.

## Tech Stack
- **Languages**: Python 3.10+
- **Deep Learning**: PyTorch (CUDA supported)
- **Numerical Processing**: NumPy
- **Visualization**: Matplotlib
- **Progress Tracking**: `tqdm`
- **Logging**: JSON-based metrics storage.

## Flow Plan

1. **Dataset Generation**: 
   - Uses `SyntheticICLDataset` to create on-the-fly few-shot prompts.
   - Supports fixed-size training or "Infinite Data" (never seeing the same sample twice).

2. **Training Phase**:
   - Orchestrated by `train.py`.
   - Use `--infinite` for on-the-fly data generation.
   - Includes **ICL-based Early Stopping** by default.

3. **Evaluation Phase**:
   - Orchestrated by `src/evaluation/suite.py`.
   - Extracts LCS, Induction scores, and Scaling metrics (0-shot to 5-shot) in one pass.

4. **Visualization**:
   - Orchestrated by `src/plots/visualize.py`.
   - Processes logs and evaluation suites to generate all necessary charts.

## Directory Structure
- `train.py`: Unified training entry point.
- `src/model/`: Transformer architecture definitions.
- `src/dataset/`: Synthetic data generation logic.
- `src/evaluation/suite.py`: Unified evaluation suite.
- `src/plots/visualize.py`: Unified visualization suite (now handles emergence plots).
- `scripts/evaluate_icl_emergence.py`: Measures ICL robustness via Label Flipping and consolidates results.
- `scripts/`: Auxiliary utilities and one-off probing scripts.
- `checkpoints/`: Model state dictionaries.
- `results/`: Logs, plots, and JSON evaluation data.
