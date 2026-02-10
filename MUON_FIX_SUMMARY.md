# Muon Optimizer Fix for 3D Conv Parameters

## Problem

When using PyTorch's built-in `torch.optim.Muon` optimizer with the MoE Transformer model, the training script crashed with:

```
ValueError: Muon only supports 2D parameters whereas we found a parameter with size: torch.Size([3584, 1, 4])
```

## Root Cause

The Mamba SSM blocks use `Conv1d` layers for temporal convolutions, which have **3D weight tensors** with shape `[out_channels, in_channels, kernel_size]`. For example: `[512, 1, 4]`.

The parameter classification logic in `get_muon_optimizer()` was sending **all parameters with `dim >= 2`** to Muon:

```python
# OLD (broken) logic:
if param.dim() < 2 or 'embedding' in name or ...:
    adamw_params.append(param)
else:
    muon_params.append(param)  # Sends 2D, 3D, 4D to Muon ❌
```

However, PyTorch's built-in `torch.optim.Muon` **only supports exactly 2D parameters**, not 3D or 4D.

## Solution

Changed the classification logic to only send **exactly 2D parameters** to Muon:

```python
# NEW (fixed) logic:
if param.dim() != 2 or 'embedding' in name or ...:
    adamw_params.append(param)  # 1D, 3D, 4D go to AdamW ✓
else:
    muon_params.append(param)  # Only exactly 2D go to Muon ✓
```

## Changes Made

**File:** `moe_arch/training/muon_optimizer.py`

**Line 260:** Changed from `param.dim() < 2` to `param.dim() != 2`

**Line 268:** Updated comment to clarify "exactly 2D"

**Lines 270-272:** Updated print statements for clarity

## Parameter Distribution

With the fix applied to the test model:

- **Muon params (exactly 2D):** 5,195,776 parameters
  - Linear weight matrices from attention and FFN layers

- **AdamW params (1D/3D+/embed/heads):** 780,544 parameters
  - 4× Conv1d 3D weights (from Mamba blocks): `[512, 1, 4]` each
  - 1D parameters (biases, layer norms)
  - Embeddings
  - LM heads

## Verification

Three test scripts confirm the fix:

1. **`test_muon_conv_fix.py`** - Verifies 3D conv parameters route to AdamW
2. **`test_muon_with_model.py`** - Tests with actual MoE Transformer model
3. **`moe_arch/training/muon_optimizer.py`** - Unit tests pass

All tests pass successfully ✓

## Impact

- Training script now runs without errors
- Mamba Conv1d layers properly optimized with AdamW
- 2D linear layers properly optimized with Muon
- No change to training dynamics (Conv1d was always supposed to use AdamW per Muon paper recommendations)

## Files Modified

- `moe_arch/training/muon_optimizer.py` (lines 260, 268, 270-272)

## Testing

Run the verification tests:

```bash
# Test parameter classification
python test_muon_conv_fix.py

# Test with MoE model
python test_muon_with_model.py

# Run optimizer unit tests
python -m moe_arch.training.muon_optimizer
```

All tests should pass ✓
