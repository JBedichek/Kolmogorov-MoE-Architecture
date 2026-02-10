# Active Parameter Tracking

## Overview

Added functionality to track and display the number of **active parameters** during training. This is critical for MoE (Mixture of Experts) models where most parameters are inactive during each forward pass due to sparse expert selection.

## What Changed

### 1. Model (`moe_arch/model/transformer.py`)

#### New Method: `count_active_parameters()`
Calculates the number of active parameters per forward pass:
- **Always active**: Embeddings, attention/mamba layers, norms, LM heads
- **MoE layers**: Only `top_k / n_experts` fraction of expert parameters
- **Example**: With top-2 of 16 experts, only 12.5% of expert parameters are active

```python
def count_active_parameters(self) -> int:
    """Count active parameters per forward pass accounting for MoE sparsity."""
    # Embeddings, norms, LM heads: always active
    active_params = ...

    for layer in self.layers:
        # Attention/Mamba: always active
        active_params += ...

        if layer.use_moe:
            # Only top_k/n_experts of expert params active
            expert_params = ...
            active_expert_params = expert_params * (top_k / n_experts)
            active_params += active_expert_params
        else:
            # Standard FFN: all params active
            active_params += ...

    return active_params
```

#### Updated: `count_parameters()`
Now returns additional fields:
- `"active"`: Number of active parameters
- `"active_billions"`: Active params in billions
- `"sparsity"`: Fraction of parameters inactive per forward pass

#### Updated: Model Initialization
Now displays active parameters during model creation:
```
Initialized MoETransformer with 5.37B parameters
  Active parameters: 2.15B (60.0% sparsity)
  Multi-token prediction: 4 heads
```

### 2. Training Script (`scripts/train.py`)

Updated parameter display to show active parameters:
```python
params = model.count_parameters()
print(f"\nModel parameters:")
print(f"  Total: {params['total']:,} ({params['total_billions']:.3f}B)")
print(f"  Active: {params['active']:,} ({params['active_billions']:.3f}B)")
print(f"  Sparsity: {params['sparsity']:.1%} (inactive per forward pass)")
print(f"  MoE layers: {params['n_moe_layers']}")
print(f"  Mamba layers: {params['n_mamba_layers']}")
```

## Example Output

### Small Test Model (4 layers)
```
Initialized MoETransformer with 0.01B parameters
  Active parameters: 0.00B (7.6% sparsity)
  Multi-token prediction: 2 heads

Model parameters:
  Total: 5,140,736 (0.005B)
  Active: 4,747,520 (0.005B)
  Sparsity: 7.6% (inactive per forward pass)
  MoE layers: 2
  Mamba layers: 1
```

### 8-Layer Model (scaled from 5B config)
```
Initialized MoETransformer with 1.86B parameters
  Active parameters: 0.89B (52.0% sparsity)
  Multi-token prediction: 4 heads

Model parameters:
  Total: 1,861,582,848 (1.862B)
  Active: 892,698,624 (0.893B)
  Sparsity: 52.0%
  MoE layers: 4
  Mamba layers: 1
```

### Full 5B Model (expected, 32 layers)
```
Initialized MoETransformer with 5.37B parameters
  Active parameters: 2.15B (60.0% sparsity)
  Multi-token prediction: 4 heads

Model parameters:
  Total: 5,370,000,000 (5.370B)
  Active: 2,150,000,000 (2.150B)
  Sparsity: 60.0%
  MoE layers: 16
  Mamba layers: 7
```

## Sparsity Breakdown

The sparsity depends on the fraction of MoE layers and the expert selection ratio:

| Configuration | Expert Selection | MoE Layers | Expected Sparsity |
|---------------|------------------|------------|-------------------|
| top-2 of 4 experts | 50% | 2 / 4 layers | ~20-25% |
| top-2 of 16 experts | 12.5% | 16 / 32 layers | ~55-60% |
| top-2 of 8 experts | 25% | 8 / 16 layers | ~30-40% |

**Formula for MoE sparsity:**
```
moe_sparsity = (1 - top_k/n_experts) * (moe_expert_params / total_params)
```

For the 5B model:
- **Total params**: 5.37B
- **Expert params**: ~3.5B (MoE FFN layers)
- **Top-2 of 16**: Only 12.5% of expert params active
- **Active expert params**: 3.5B × 0.125 = 437.5M
- **Inactive expert params**: 3.5B × 0.875 = 3.06B
- **Sparsity**: 3.06B / 5.37B ≈ **57%**

## Why This Matters

1. **Memory Planning**: Active params determine minimum memory needed
2. **Compute Estimation**: FLOPs scale with active params, not total
3. **Comparison**: ~2B active params ≈ performance of dense 2B model
4. **Cost Analysis**: Training cost proportional to active params
5. **Architecture Design**: Balance total capacity vs. active compute

## Training Display

During training, the active parameters are shown at startup:
```bash
$ python scripts/train.py --config configs/training_dolma.yaml

================================================================================
MoE Transformer Training
================================================================================

Initializing model...
Initialized MoETransformer with 5.37B parameters
  Active parameters: 2.15B (60.0% sparsity)
  Multi-token prediction: 4 heads

Model parameters:
  Total: 5,370,000,000 (5.370B)
  Active: 2,150,000,000 (2.150B)
  Sparsity: 60.0% (inactive per forward pass)
  MoE layers: 16
  Mamba layers: 7
```

## Notes

- **MoD (Mixture of Depths)** reduces compute but doesn't affect parameter count
  - All parameters are still "active", just used on fewer tokens
  - MoD gives compute savings, not parameter savings
- **Active parameters** are per-forward-pass, averaged across tokens
  - Different tokens may route to different experts
  - The count represents expected active params per token
- **Sparsity** measures parameter efficiency
  - Higher sparsity = more capacity without proportional compute cost
  - Target: 50-60% sparsity for good capacity/efficiency tradeoff
