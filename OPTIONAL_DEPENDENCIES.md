# Optional Dependencies Guide

## Current Warnings Explained

```
Warning: flash-attn not available, falling back to standard attention
Warning: mamba-ssm not available, using simplified SSM
```

These warnings indicate that optional high-performance libraries are not installed. **Your model will work fine without them**, but installing them provides significant performance improvements.

## Installation

### Option 1: Install Everything (Recommended for Production)

```bash
# Install all optional dependencies
pip install flash-attn mamba-ssm --no-build-isolation

# Or with specific versions
pip install flash-attn>=2.3.0 mamba-ssm>=1.0.0 --no-build-isolation
```

### Option 2: Install Individually

```bash
# Install Flash Attention only (for memory savings)
pip install flash-attn --no-build-isolation

# Install Mamba SSM only (for optimized SSM kernels)
pip install mamba-ssm
```

### Option 3: Skip Installation (Use Fallbacks)

If you're just testing or don't have compatible hardware:
- No installation needed
- Model uses fallback implementations
- Warnings are informational only

## What Each Library Does

### 1. Flash Attention (`flash-attn`)

**What it is:**
- Memory-efficient attention implementation
- Uses CUDA kernels for fused operations
- Reduces memory from O(N²) to O(N)

**Benefits:**
- ✅ ~8x memory reduction for attention
- ✅ ~2-3x speedup for attention operations
- ✅ Allows longer sequences or larger batches

**Fallback without it:**
- Standard PyTorch attention implementation
- Works correctly but uses more memory
- Slightly slower

**Requirements:**
- CUDA GPU (not available on CPU)
- CUDA 11.6+
- PyTorch 2.0+

### 2. Mamba SSM (`mamba-ssm`)

**What it is:**
- Optimized CUDA kernels for Mamba state space models
- Official implementation from the Mamba paper
- Selective scan with data-dependent matrices

**Benefits:**
- ✅ Optimized CUDA kernels (faster)
- ✅ Full Mamba feature set
- ✅ Better numerical stability

**Fallback without it:**
- Simplified SSM implementation in PyTorch
- Works correctly but may be slower
- May have slightly different numerical behavior

**Requirements:**
- CUDA GPU (not available on CPU)
- CUDA 11.6+
- PyTorch 2.0+

## Performance Impact

### With Optional Dependencies (Production)

```
Training Configuration:
  Memory: ~60GB (with Flash Attention)
  Speed: ~35-45k tokens/sec
  Batch size: 8 (effective)

Estimated training time (100B tokens):
  23-30 days on A100 80GB
```

### Without Optional Dependencies (Fallback)

```
Training Configuration:
  Memory: ~70-75GB (standard attention)
  Speed: ~25-35k tokens/sec
  Batch size: 6-8 (may need to reduce)

Estimated training time (100B tokens):
  30-38 days on A100 80GB
```

## Installation Troubleshooting

### Flash Attention Won't Install

```bash
# Make sure you have CUDA toolkit installed
nvcc --version

# Install with pip (not conda)
pip install flash-attn --no-build-isolation

# If it fails, check:
# 1. CUDA version (needs 11.6+)
# 2. PyTorch CUDA version matches system CUDA
# 3. You have a compatible GPU (Ampere/Ada/Hopper)
```

### Mamba SSM Won't Install

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install mamba-ssm
pip install mamba-ssm

# If it fails:
# 1. Make sure PyTorch CUDA version is correct
# 2. Check you have compatible GPU
```

### Neither Works (Use Fallbacks)

If you can't install either library:
- ✅ Model still works with fallbacks
- ✅ All features still available (MoE, MoD, multi-token)
- ⚠️ Will use more memory and be slower
- ⚠️ May need to reduce batch size

## Recommended Configurations

### For Testing/Development (CPU or limited GPU)
```bash
# No optional dependencies needed
# Use fallback implementations
# Config: configs/test.yaml
```

### For Small-Scale Training (Single GPU)
```bash
# Install Flash Attention for memory savings
pip install flash-attn --no-build-isolation

# Mamba SSM is optional
```

### For Production Training (Multi-GPU or Long Training)
```bash
# Install everything
pip install flash-attn mamba-ssm --no-build-isolation

# Config: configs/training_dolma.yaml
```

## Disable Warnings

If you intentionally want to use fallbacks and suppress warnings:

```python
# Add to train.py at the top
import warnings
warnings.filterwarnings('ignore', message='flash-attn not available')
warnings.filterwarnings('ignore', message='mamba-ssm not available')
```

Or run with warnings disabled:
```bash
python -W ignore scripts/train.py --config configs/training_dolma.yaml
```

## Verify Installation

```python
# Check if Flash Attention is available
python -c "
try:
    from flash_attn import flash_attn_func
    print('✅ Flash Attention installed')
except ImportError:
    print('❌ Flash Attention not available')
"

# Check if Mamba SSM is available
python -c "
try:
    from mamba_ssm import Mamba
    print('✅ Mamba SSM installed')
except ImportError:
    print('❌ Mamba SSM not available')
"
```

## Summary

| Library | Required? | Impact | Install Priority |
|---------|-----------|--------|------------------|
| PyTorch | ✅ Required | Core framework | Critical |
| transformers | ✅ Required | Tokenizers | Critical |
| flash-attn | ⚠️ Optional | 8x memory savings | High |
| mamba-ssm | ⚠️ Optional | Faster SSM | Medium |

**Recommendation:**
- **Just testing**: Skip optional deps, use fallbacks
- **Serious training**: Install both for best performance
- **Limited resources**: At minimum install flash-attn

The model works either way - optional dependencies just make it faster and more memory-efficient!
