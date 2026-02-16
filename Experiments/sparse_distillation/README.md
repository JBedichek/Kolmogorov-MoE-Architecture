# Sparse Distillation

Convert a pretrained dense transformer into a sparse Mixture-of-Experts model by training MoE layers to replicate dense FFN behavior.

## Overview

Given a dense model with FFN layers computing `y = FFN(x)`, we train MoE layers to minimize:

```
L = MSE(MoE(x), FFN(x)) + λ·(1 - cos_sim(MoE(x), FFN(x)))
```

This preserves learned representations while adding sparsity.

## Workflow

### Step 1: Load Dense Model

```python
from transformers import AutoModelForCausalLM

dense_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

### Step 2: Configure Distillation

```python
from Experiments.sparse_distillation.distill import DistillationConfig

config = DistillationConfig(
    n_experts=16,              # Experts per layer
    moe_top_k=2,               # Active experts per token
    moe_capacity_factor=1.25,  # Tokens per expert (expert-choice only)
    moe_routing="token_choice",  # Recommended for distillation (see note below)
    d_ff_expert=0.5,           # Expert dim: None=match, 0.5=half, 2048=absolute
    learning_rate=1e-4,
    init_from_dense=True,      # Copy FFN weights to experts (if dims match)
)
# NOTE: Use token_choice for distillation. Expert-choice can leave tokens
# with near-zero output, causing huge mismatch with dense FFN targets.
```

### Step 3: Initialize Trainer

```python
from Experiments.sparse_distillation import SparseDistillationTrainer

trainer = SparseDistillationTrainer(
    dense_model=dense_model,
    config=config,
    device="cuda",
)
```

The trainer:
1. Discovers all FFN/MLP layers via hooks
2. Infers `d_model`, `d_ff`, and `dtype` from the first FFN
3. Creates one MoE layer per FFN layer
4. Optionally initializes expert weights from dense FFN (with noise for symmetry breaking)

### Step 4: Prepare Data

Any tokenized text dataset works. The dataloader should yield batches with `input_ids`:

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# ... tokenize ...
dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)
```

### Step 5: Run Distillation

```python
trainer.distill(dataloader, epochs=3)
```

For each batch:
1. Forward pass through frozen dense model
2. Hooks capture `(input, output)` at each FFN layer
3. Each MoE layer trains on its corresponding FFN's activations
4. Loss = MSE + cosine similarity

No disk caching—activations stream directly to MoE training.

### Step 6: Evaluate Quality

```python
results = trainer.evaluate(eval_dataloader, max_batches=100)
# Returns {layer_idx: {'mse': float, 'cosine_sim': float}}
```

Target: cosine similarity > 0.95 indicates good distillation.

### Step 7: Save Checkpoint

```python
trainer.save("distilled_moe.pt")
```

Saves: MoE weights, config, dimensions, dtype, training history.

### Step 8: Convert to Sparse Model

```python
from Experiments.sparse_distillation.convert import convert_to_sparse

sparse_model = convert_to_sparse(
    dense_model,
    "distilled_moe.pt",
    device="cuda",
)

# Use normally
outputs = sparse_model(input_ids)
```

This replaces each dense FFN with its trained MoE counterpart.

## Training Modes

### Parallel Mode (default)
Trains all MoE layers simultaneously. Fast but high memory.

### Sequential Mode (`--sequential`)
Trains one layer at a time. Much lower memory, suitable for large models.

```python
config = DistillationConfig(
    sequential_layers=True,
    ...
)
```

### Dynamic Expert Reallocation (`--dynamic`)
Tracks per-expert loss and reallocates parameters:
- High-loss experts grow (more d_ff dimensions)
- Low-loss experts shrink (fewer dimensions)

Weight matrices are interpolated to preserve learned functions:
- Growing: bilinear interpolation to expand dimensions
- Shrinking: importance-based selection (keeps highest L2-norm dimensions)

```python
config = DistillationConfig(
    dynamic_experts=True,
    reallocate_every_n_steps=500,  # How often to reallocate
    growth_factor=1.25,            # Grow by 25%
    shrink_factor=0.8,             # Shrink by 20%
    min_d_ff=256,                  # Minimum expert dimension
    max_d_ff=8192,                 # Maximum expert dimension
    ...
)
```

This results in variable-sized experts, with the model adapting its capacity to where it's needed most.

## CLI Usage

```bash
# Standard parallel training
python Experiments/sparse_distillation/example_usage.py \
    --model meta-llama/Llama-3.2-3B \
    --dataset wikitext \
    --n_experts 16 \
    --d_ff_expert 0.5 \
    --epochs 3 \
    --output ./sparse_llama

# Sequential (low memory)
python Experiments/sparse_distillation/example_usage.py \
    --model meta-llama/Llama-3.2-3B \
    --sequential \
    --output ./sparse_llama

# Dynamic expert reallocation
python Experiments/sparse_distillation/example_usage.py \
    --model meta-llama/Llama-3.2-3B \
    --dynamic \
    --reallocate_every 500 \
    --growth_factor 1.25 \
    --shrink_factor 0.8 \
    --output ./sparse_llama
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_experts` | 16 | Experts per MoE layer |
| `moe_top_k` | 2 | Active experts per token (token-choice) |
| `moe_capacity_factor` | 1.25 | Expert capacity multiplier |
| `moe_routing` | `"token_choice"` | `"token_choice"` (recommended) or `"expert_choice"` |
| `d_ff_expert` | `None` | `None`=match dense, `0.5`=half, `int`=absolute |
| `learning_rate` | 1e-4 | MoE training LR |
| `use_cosine_loss` | `True` | Add cosine similarity to loss |
| `cosine_loss_weight` | 0.1 | Weight for cosine term |
| `init_from_dense` | `True` | Initialize experts from FFN weights |
| `sequential_layers` | `False` | Train one layer at a time (low memory) |
| `dynamic_experts` | `False` | Enable dynamic expert sizing |
| `reallocate_every_n_steps` | 500 | Steps between reallocations |
| `growth_factor` | 1.25 | Multiply high-loss expert d_ff by this |
| `shrink_factor` | 0.8 | Multiply low-loss expert d_ff by this |
| `min_d_ff` | 256 | Minimum expert dimension |
| `max_d_ff` | 8192 | Maximum expert dimension |

## Architecture Diagram

```
Dense Model (frozen)                    MoE Layers (training)
┌─────────────────────┐                 ┌─────────────────────┐
│ Layer 0             │                 │ MoE 0               │
│ ┌─────┐   ┌─────┐   │   x,y hooks     │ ┌────────┐          │
│ │Attn │ → │ FFN │───┼────────────────→│ │Router  │          │
│ └─────┘   └─────┘   │                 │ ├────────┤          │
├─────────────────────┤                 │ │E₀ E₁...│          │
│ Layer 1             │                 │ └────────┘          │
│ ┌─────┐   ┌─────┐   │                 ├─────────────────────┤
│ │Attn │ → │ FFN │───┼────────────────→│ MoE 1               │
│ └─────┘   └─────┘   │                 │ ...                 │
└─────────────────────┘                 └─────────────────────┘
                                                 │
                                                 ▼
                                        L = MSE + λ·cos_loss
```

## Memory & Compute

- **Disk**: Zero. Activations stream, no caching.
- **GPU**: Dense model + one MoE layer batch at a time.
- **Speed**: ~same as dense model inference (MoE training is lightweight).

## Parameter Scaling

| Config | Total MoE Params | Active Params (top-2) |
|--------|------------------|----------------------|
| `d_ff_expert=None, n=16` | 16× dense FFN | 2× dense FFN |
| `d_ff_expert=0.5, n=16` | 8× dense FFN | 1× dense FFN |
| `d_ff_expert=0.25, n=16` | 4× dense FFN | 0.5× dense FFN |

## Files

```
sparse_distillation/
├── __init__.py         # Exports
├── hooks.py            # FFN activation extraction
├── distill.py          # Main trainer + config
├── dynamic_experts.py  # Variable-sized experts + reallocation
├── convert.py          # Dense → sparse conversion
├── example_usage.py    # Full CLI example
└── README.md
```
