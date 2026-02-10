# BFloat16 Precision Bug - Fix Instructions

## The Problem

Your RMSNorm parameters are frozen because:

1. Model parameters are cast to **BFloat16**: `model.to(dtype=torch.bfloat16)`
2. Norm parameters are initialized to **1.0**
3. AdamW produces small updates (~1e-5 to 1e-4)
4. **BFloat16 can't represent changes smaller than 0.0078 at value 1.0**
5. Updates underflow to zero: `1.0 - 0.0001 = 1.0` (in bf16)

## Test Results

```
✓ Gradients ARE flowing (tested with gradient flow script)
✓ Optimizer IS running (AdamW has momentum buffers)
✓ Learning rate is reasonable (3e-4 for AdamW)
✗ Updates are too small for BFloat16 precision
```

## The Fix

### Option 1: Use Proper AMP (RECOMMENDED)

Keep parameters in FP32, use BFloat16 only for compute:

```python
# In scripts/train.py, REPLACE:
model = model.to(dtype=torch.bfloat16)  # ❌ WRONG

# WITH:
model = model.to(args.device)  # Keep in FP32

# Then use autocast for forward pass:
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']

# Backward in FP32 (automatic)
loss.backward()
```

This way:
- Forward pass computes in BFloat16 (memory efficient)
- Parameters stay in FP32 (precision for updates)
- Gradients are in FP32 (precision for optimizer)

### Option 2: Increase Learning Rate for Norms

If you want to keep everything in BFloat16:

```python
# In moe_arch/training/muon_optimizer.py, change line 290:
adamw_opt = torch.optim.AdamW(
    adamw_params,
    lr=lr * 0.3,  # ❌ Too small: 3e-4

# TO:
adamw_opt = torch.optim.AdamW(
    adamw_params,
    lr=lr * 3.0,  # ✓ Larger: 3e-3 (10x higher)
```

This produces updates of ~1e-3, which BFloat16 CAN represent.

**Downside:** May cause instability.

### Option 3: Initialize Norms Differently

Initialize RMSNorm weights to a smaller value where BFloat16 has better precision:

```python
# In moe_arch/model/embeddings.py, line 180:
self.weight = nn.Parameter(torch.ones(dim))  # ❌ Init to 1.0

# TO:
self.weight = nn.Parameter(torch.ones(dim) * 0.1)  # ✓ Init to 0.1
```

At value 0.1, BFloat16 can represent much smaller differences.

**Downside:** Changes model initialization, need to retrain from scratch.

## Recommended Solution

**Use Option 1 (Proper AMP)**. It's the standard approach and gives you:
- Memory efficiency of BFloat16
- Precision of FP32 for parameters
- No need to retune hyperparameters

## Implementation Steps

1. **Modify `scripts/train.py`:**
   - Remove `model = model.to(dtype=torch.bfloat16)` (line 116)
   - Add autocast wrapper in training loop

2. **Find training loop** (likely in `moe_arch/training/trainer.py`)

3. **Wrap forward pass with autocast:**
   ```python
   with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
       outputs = model(batch)
       loss = outputs['loss']
   ```

4. **Keep backward in FP32** (automatic, no changes needed)

5. **Restart training from checkpoint** (optimizer state will adapt)

## Verification

After applying the fix, check that norms are updating:

```python
# Add to training loop
if step % 100 == 0:
    for name, param in model.named_parameters():
        if 'norm' in name:
            print(f"{name}: {param.data.mean().item():.6f}")
            break  # Just check one
```

You should see the value changing from 1.0 after a few steps.

## Why This Wasn't Caught Earlier

- Most parameters initialized ~0.02-0.05 (normal distribution)
- BFloat16 has better precision at these magnitudes
- Only parameters near 1.0 (norms) hit this issue
- Test models are small and train quickly (bug doesn't show up)

## Additional Notes

- This is a common pitfall with BFloat16
- PyTorch's official AMP API handles this automatically
- Casting entire model to BFloat16 is usually wrong
- Always keep optimizer states in FP32

