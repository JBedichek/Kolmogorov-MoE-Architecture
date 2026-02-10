# Training Diagnostic Report

## Executive Summary

**Status:** 🔴 **CRITICAL TRAINING BUG FOUND**

After 754 training steps (1.5M tokens), **all 49 RMSNorm parameters are completely frozen** at their initialization values. This is preventing the model from learning effectively.

## Checkpoint Analysis

- **Location:** `checkpoints/checkpoint_interrupted.pt`
- **Training progress:** Step 754, Epoch 0, 1,544,192 tokens seen
- **Model:** 3.09B parameters (1.31B active, 57.7% sparse)
- **Config:** 1792d, 24L, 14 experts, top-2

## Critical Finding: Frozen RMSNorm Layers

### What's Wrong

**All 49 RMSNorm parameters (6% of total params) are COMPLETELY FROZEN:**

```
layers.0.seq_norm.weight   [all 1.0]  ❌ FROZEN
layers.0.ffn_norm.weight   [all 1.0]  ❌ FROZEN
layers.1.seq_norm.weight   [all 1.0]  ❌ FROZEN
... (all 49 norms)
```

These should be changing during training but haven't moved AT ALL from initialization.

### Impact

-  **Broken normalization:** Model can't learn proper feature scaling
-  **Training instability:** Without adaptive normalization, gradients can explode/vanish
-  **Poor convergence:** Loss won't decrease properly

## What IS Working

✅ **Other 761 parameters (94%) ARE updating:**
- Attention weights
- FFN weights
- MoE expert weights
- Embeddings
- Mamba conv weights (actually changing TOO MUCH - 300-400% relative change!)

✅ **Optimizer is configured correctly:**
- Muon: 746 params at LR=1e-3
- AdamW: 64 params at LR=3e-4 (includes the 49 norms)
- Both have optimizer state (momentum buffers exist)

## Root Cause Analysis

The norms ARE in the optimizer (AdamW), and AdamW HAS state. So why aren't they moving?

**Possible causes:**

### 1. **RMSNorm gradients are zero** (MOST LIKELY)
   - RMSNorm might not be receiving gradients during training
   - Could be a gradient flow issue through the MoD (Mixture of Depths) routers
   - MoD skips 25% of tokens - if routing is broken, norms might not get gradients

### 2. **Learning rate is effectively zero for norms**
   - AdamW LR is 3e-4 (seems reasonable)
   - But if gradients are tiny, updates will be tiny

### 3. **Gradient clipping is too aggressive**
   - If norm gradients are being clipped to near-zero

### 4. **BFloat16 precision issues**
   - Model is in bfloat16
   - Very small updates might underflow to zero

## Diagnostic Evidence

### Test 1: RMSNorm CAN get gradients in isolation
```python
# Standalone test
norm = RMSNorm(128)
x = torch.randn(2, 10, 128)
out = norm(x)
loss = out.sum()
loss.backward()
# ✓ norm.weight.grad is NON-ZERO (mean=-0.18, std=4.6)
```

**Conclusion:** RMSNorm implementation is correct. Problem is in the training loop.

### Test 2: Optimizer configuration
```
AdamW:
  - 64 parameters total
  - 49 are norms
  - LR = 3e-4
  - Has optimizer state (exp_avg, exp_avg_sq)
```

**Conclusion:** Norms ARE in the optimizer with proper LR.

### Test 3: Parameter comparison
```
Fresh model norm: [1.0, 1.0, 1.0, ...]
Checkpoint norm:  [1.0, 1.0, 1.0, ...]  ← EXACT SAME!
```

**Conclusion:** Zero movement after 754 steps.

## Recommended Actions

### IMMEDIATE (Do these first):

1. **Check gradient flow through MoD**
   ```python
   # Add this to training loop
   for name, param in model.named_parameters():
       if 'norm' in name and param.grad is not None:
           print(f"{name}: grad_mean={param.grad.mean():.2e}, grad_std={param.grad.std():.2e}")
   ```

2. **Verify RMSNorm is in computation graph**
   - Check if MoD is bypassing norms
   - Look at `moe_arch/model/layers.py` MoD implementation

3. **Disable MoD temporarily** to test
   - Set `mod_capacity_factor=1.0` (no skipping)
   - If norms start moving → MoD is the problem

### MEDIUM PRIORITY:

4. **Check gradient clipping**
   - Look in `moe_arch/training/trainer.py`
   - If using gradient clipping, try disabling it temporarily

5. **Increase AdamW learning rate**
   - Current: 3e-4 (0.3x of Muon's 1e-3)
   - Try: 1e-3 (same as Muon)
   - This is a workaround, not a fix

6. **Add gradient logging**
   - Log gradient norms for all parameter groups
   - Check if norm gradients exist but are tiny

### DEBUGGING CODE:

Add this to your training script right after `loss.backward()`:

```python
# Check if norms are getting gradients
norm_grads = []
for name, param in model.named_parameters():
    if 'norm' in name.lower() and param.grad is not None:
        grad_norm = param.grad.abs().mean().item()
        norm_grads.append((name, grad_norm))

if len(norm_grads) > 0:
    print("\nNorm gradients:")
    for name, grad in norm_grads[:5]:
        print(f"  {name}: {grad:.2e}")
else:
    print("⚠️  WARNING: NO GRADIENTS FOR NORMS!")
```

## Secondary Issue: Wild Parameter Changes

Some Mamba conv weights are changing by 300-400% relative to their magnitude:

```
layers.21.seq_layer.mamba.conv1d.weight: 399% change
layers.5.seq_layer.mamba.conv1d.weight:  397% change
```

This suggests:
- Learning rate might be too high for some parameters
- Or these parameters are initialized too small

**Recommendation:** Monitor training loss curve. If loss is oscillating/diverging, reduce LR.

## Files to Inspect

1. `moe_arch/model/layers.py` - Check MoD implementation around lines 122-196
2. `moe_arch/training/trainer.py` - Check gradient clipping and optimization loop
3. `moe_arch/model/mod.py` - Check if MoDLayer properly includes norms in computation

## Next Steps

1. Run the debugging code above to check if norms get gradients
2. If no gradients → investigate MoD/layer implementation
3. If tiny gradients → increase AdamW LR or disable gradient clipping
4. If problem persists → disable MoD temporarily to isolate issue

## Summary

- ❌ **49/810 parameters frozen (all RMSNorms)**
- ✅ **761/810 parameters updating normally**
- 🔍 **Root cause: Likely gradient flow issue through MoD**
- 🎯 **Fix: Debug gradient flow, possibly disable/fix MoD**
