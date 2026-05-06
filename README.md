# ccLLM

A minimal GPT-style language model implemented from scratch in a single Python file using PyTorch. Character-level tokenisation, causal self-attention, cosine learning-rate schedule with linear warmup, and weight tying — no external ML libraries beyond PyTorch.

## Requirements

```
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. Runs on CPU or CUDA automatically.

## Quick start

```bash
# 1. Download tiny Shakespeare (~1 MB)
python llm.py download

# 2. Train with defaults (4L × 4H × 128D, 5 000 iters)
python llm.py train --data input.txt

# 3. Generate text
python llm.py generate "HAMLET:"
```

## Commands

### `download`
Downloads the tiny Shakespeare dataset (~1 MB) to `input.txt`.

### `train`
Trains a model on any plain-text file and saves a checkpoint.

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Path to training `.txt` file |
| `--embd` | `128` | Embedding dimension |
| `--heads` | `4` | Number of attention heads |
| `--layers` | `4` | Number of transformer layers |
| `--context` | `256` | Context window (tokens) |
| `--iters` | `5000` | Training iterations |
| `--batch` | `32` | Batch size |
| `--lr` | `3e-4` | Peak learning rate |
| `--checkpoint` | `model.pt` | Output checkpoint path |

Example — larger model:
```bash
python llm.py train --data input.txt --embd 256 --layers 6 --heads 8 --context 512 --iters 10000
```

### `generate`
Generates text from a saved checkpoint.

| Flag | Default | Description |
|------|---------|-------------|
| `prompt` | *(required)* | Seed text |
| `--tokens` | `200` | Number of tokens to generate |
| `--temp` | `0.8` | Sampling temperature |
| `--top-k` | `40` | Top-k sampling (`0` = off) |
| `--checkpoint` | `model.pt` | Checkpoint to load |

### `info`
Prints architecture and parameter count for a saved checkpoint.

```bash
python llm.py info --checkpoint model.pt
```

## Architecture

- Character-level tokeniser (vocabulary built from training corpus)
- Token + positional embeddings with weight tying to output projection
- Stacked transformer blocks: pre-norm → causal self-attention → pre-norm → feed-forward (GELU)
- AdamW optimiser with cosine LR decay and linear warmup
- GPT-2 style scaled initialisation for residual projections
