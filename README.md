# Advanced 5B Parameter MoE Architecture

State-of-the-art Mixture-of-Experts language model with advanced architectural features.

## Features

- **Mixture of Depths (MoD)**: Conditional layer skipping for 25% compute savings
- **SSM-Transformer Hybrid**: Mamba blocks for efficient O(n) sequence modeling
- **Multi-Token Prediction**: Predict 4 tokens ahead simultaneously
- **Muon Optimizer**: Momentum orthogonalized optimization
- **Grouped Query Attention (GQA)**: 4:1 query-to-KV ratio for memory efficiency
- **Flash Attention 2**: Memory-efficient attention implementation
- **RoPE Positional Encodings**: Better length generalization

## Model Specifications

- **Total Parameters**: ~5B
- **Active Parameters**: ~1.5-2B per forward pass (40% sparsity)
- **Layers**: 32 transformer blocks
- **Hidden Size**: 2048
- **Attention Heads**: 16 (4 KV heads for GQA)
- **MoE Experts**: 16 per layer (16 MoE layers total)
- **Context Length**: 2048 tokens

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.model.transformer import MoETransformer

# Create model
config = AdvancedMoEConfig()
model = MoETransformer(config)

# Training
from scripts.train import train
train(model, config)
```

## Training on Dolma

See `scripts/train.py` for full training pipeline.

## Architecture Overview

The model uses a repeating 4-layer pattern (×8 = 32 layers):
1. Attention + Standard FFN (with MoD)
2. Mamba + MoE FFN (with MoD)
3. Attention + MoE FFN (with MoD)
4. Attention + Standard FFN (with MoD)

## Development

```bash
# Run tests
pytest tests/

# Format code
black moe_arch/
isort moe_arch/
```

## License

MIT
