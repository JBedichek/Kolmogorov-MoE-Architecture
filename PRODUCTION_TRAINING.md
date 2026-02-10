# Production Training Guide

Complete guide for training the MoE model with FineWeb dataset in production.

## Quick Start

```bash
# Start training with default config (10B tokens)
python train_production.py --config configs/production_training.yaml

# Resume from checkpoint
python train_production.py --config configs/production_training.yaml --resume checkpoints_production/checkpoint-5000
```

## Configuration

Edit `configs/production_training.yaml` to customize training:

### Key Parameters

**Total Training Tokens:**
```yaml
training:
  total_tokens: 10_000_000_000  # 10B tokens
```

The script automatically calculates training steps:
- Steps = total_tokens / (batch_size × gradient_accumulation_steps × seq_len)
- Example: 10B / (1 × 8 × 512) = 2,441,406 steps

**Model Size:**
```yaml
model:
  d_model: 768      # Embedding dimension
  n_layers: 12      # Number of layers
  n_experts: 8      # Number of experts per MoE layer
```

Estimated model sizes:
- 500M params: d_model=768, n_layers=12
- 1B params: d_model=1024, n_layers=16
- 1.5B params: d_model=1536, n_layers=20

**Batch Size:**
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 8
```
- Effective batch size = batch_size × gradient_accumulation_steps = 8
- Increase gradient_accumulation_steps to simulate larger batches without more VRAM

**Checkpointing:**
```yaml
training:
  checkpoint_dir: "./checkpoints_production"
  save_interval: 1000  # Save every 1000 steps
```

**Dataset Size:**
```yaml
data:
  max_examples: null  # Use all available data
  # OR
  max_examples: 100000  # Limit to 100K examples for testing
```

**Learning Rate:**
```yaml
training:
  max_lr: 0.001        # Peak learning rate
  min_lr_ratio: 0.1    # Final LR = max_lr × min_lr_ratio
  warmup_steps: 1000   # Warmup for 1000 steps
```

## Features

### ✅ Checkpoint Saving & Resuming

**Automatic Saving:**
- Checkpoints saved every `save_interval` steps
- Only keeps last 3 checkpoints (configurable)
- Saves optimizer state, scheduler state, and model weights

**Resume Training:**
```bash
# Find your latest checkpoint
ls checkpoints_production/

# Resume from specific checkpoint
python train_production.py --config configs/production_training.yaml \
    --resume checkpoints_production/checkpoint-5000
```

### ✅ Progress Tracking

**Console Logging:**
```
{'loss': 59.46, 'grad_norm': 30.34, 'learning_rate': 0.00037, 'epoch': 1.08}
```

**Understanding Loss Values:**
- Logged loss = actual loss × gradient_accumulation_steps
- Example: Logged loss 64.0 ÷ 8 (grad_accum) = 8.0 (true loss per sample)

**Training Stats:**
- Loss decreasing: ✅ Model is learning
- Grad norm stable (~10-50): ✅ Healthy training
- Grad norm exploding (>100): ⚠ May need to adjust LR

### ✅ Weights & Biases Integration

Enable W&B logging:
```yaml
wandb:
  enabled: true
  project: "moe-production"
  run_name: "moe-500m-10b-tokens"
  entity: "your-wandb-username"  # Optional
```

Then login and run:
```bash
wandb login
python train_production.py --config configs/production_training.yaml
```

### ✅ Validation During Training

Enable evaluation:
```yaml
evaluation:
  enabled: true
  eval_steps: 100  # Number of evaluation batches
```

Evaluation runs every `eval_interval` steps.

### ✅ Mixed Precision Training

BF16 is enabled by default for faster training and lower memory:
```yaml
mixed_precision:
  bf16: true
```

Requirements:
- NVIDIA GPU with BF16 support (Ampere or newer: RTX 30xx, A100, etc.)
- Falls back to FP32 if not supported

## Example Configurations

### Small Test Run (1M tokens)
```yaml
training:
  total_tokens: 1_000_000  # 1M tokens
  max_lr: 0.001
  warmup_steps: 100
  save_interval: 100

data:
  max_examples: 1000  # Quick test
```

Expected time: ~5 minutes
Steps: ~244

### Medium Training (1B tokens)
```yaml
training:
  total_tokens: 1_000_000_000  # 1B tokens
  max_lr: 0.001
  warmup_steps: 500
  save_interval: 1000

data:
  max_examples: null  # Use all data
```

Expected time: ~8 hours (depends on GPU)
Steps: ~244,141

### Large Training (100B tokens)
```yaml
training:
  total_tokens: 100_000_000_000  # 100B tokens
  max_lr: 0.001
  warmup_steps: 2000
  save_interval: 5000

model:
  d_model: 1536
  n_layers: 20  # 1.5B parameter model

data:
  max_examples: null
```

Expected time: ~30 days (depends on GPU)
Steps: ~24,414,062

## Memory Requirements

**500M Model (d_model=768, n_layers=12):**
- FP32 weights: ~2GB
- BF16 activations: ~1GB
- Optimizer state: ~4GB
- **Total: ~7-8GB VRAM**
- Recommended: RTX 3080 (10GB) or better

**1B Model (d_model=1024, n_layers=16):**
- **Total: ~12-14GB VRAM**
- Recommended: RTX 3090 (24GB) or A100

**1.5B Model (d_model=1536, n_layers=20):**
- **Total: ~20-25GB VRAM**
- Recommended: A100 (40GB/80GB)

**To reduce memory:**
- Decrease `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Decrease `max_seq_len` (512 → 256)
- Decrease `n_layers`

## Monitoring Training

### Check Progress
```bash
# Watch training log in real-time
tail -f checkpoints_production/logs/events.out.tfevents.*

# Count checkpoints
ls checkpoints_production/ | grep checkpoint | wc -l

# Check latest loss
tail -100 <your-log-file> | grep "'loss'"
```

### Expected Behavior

**Healthy Training:**
- Loss decreases steadily
- Gradient norms stable (10-50 range)
- LR follows schedule (increases during warmup, then decays)
- No NaN or Inf values

**Warning Signs:**
- Loss stuck/not decreasing → Check learning rate, check dataset
- Gradient norms exploding (>1000) → Reduce learning rate
- Loss is NaN → Reduce learning rate or check for bugs
- Very slow progress → Check GPU utilization

### GPU Utilization

Check GPU usage:
```bash
nvidia-smi -l 1
```

Good utilization: 95-100% GPU usage
Low utilization (<80%): May be bottlenecked by data loading

## Datasets

### FineWeb (Default)
```yaml
data:
  dataset_name: "HuggingFaceFW/fineweb"
  dataset_config: "sample-10BT"  # 10B token sample
```

Sizes:
- `sample-10BT`: ~10B tokens (good for testing)
- `sample-100BT`: ~100B tokens
- `sample-350BT`: ~350B tokens (full dataset)

### Other Datasets

**FineWeb-Edu (Educational content):**
```yaml
data:
  dataset_name: "HuggingFaceFW/fineweb-edu"
  dataset_config: "sample-10BT"
```

**OpenWebText:**
```yaml
data:
  dataset_name: "Skylion007/openwebtext"
  dataset_config: null
```

### Tokenization Strategy

**On-the-Fly vs Upfront Tokenization:**

The training script supports two tokenization strategies:

**1. Upfront Tokenization (Default - `tokenize_on_fly: false`):**
```yaml
data:
  tokenize_on_fly: false
```
- Tokenizes the entire dataset before training starts
- Faster training (no tokenization overhead during training)
- Uses more memory (stores all tokenized sequences)
- Best for: Smaller datasets that fit in RAM

**2. On-the-Fly Tokenization (`tokenize_on_fly: true`):**
```yaml
data:
  tokenize_on_fly: true
```
- Tokenizes text during training, batch by batch
- More memory-efficient (only stores raw text)
- Slightly slower per batch (tokenization happens during training)
- Best for: Large datasets, memory-constrained systems

**Memory Comparison:**

For a 10B token dataset with 2048 sequence length:
- Upfront: ~40-50GB RAM for tokenized data
- On-the-fly: ~5-10GB RAM for raw text only

**Performance:**
- Upfront is typically 5-10% faster per step
- On-the-fly allows training on much larger datasets
- Choose based on your memory vs speed needs

**Example Configuration:**
```yaml
# Memory-efficient for large datasets
data:
  dataset_name: "HuggingFaceFW/fineweb"
  dataset_config: "sample-350BT"  # Large 350B token dataset
  tokenize_on_fly: true  # Save memory
  use_streaming: false
  max_examples: null
```

## Interrupting and Resuming

**Safe Interruption:**
- Press `Ctrl+C` once
- Wait for current step to finish
- Last checkpoint will be saved

**Resume:**
```bash
python train_production.py --config configs/production_training.yaml \
    --resume checkpoints_production/checkpoint-<STEP>
```

## Troubleshooting

### "CUDA out of memory"
**Solution:**
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_len`
- Use smaller model

### "Loss not decreasing"
**Check:**
1. Learning rate (should be ~1e-3 for Muon)
2. Dataset loaded correctly (not using dummy data)
3. Gradient norm is not zero
4. BF16 enabled

### "Training very slow"
**Check:**
1. GPU utilization (`nvidia-smi`)
2. Data loading workers (`num_workers: 4` in config)
3. Using streaming vs pre-loading data
4. Tokenization strategy (`tokenize_on_fly: false` is faster)

### "Checkpoints taking too much space"
**Solution:**
```yaml
training:
  save_total_limit: 3  # Only keep last 3 checkpoints
```

## Production Checklist

Before long training runs:

- [ ] Config reviewed and correct
- [ ] Checkpoint directory has enough disk space
- [ ] W&B logging configured (if desired)
- [ ] Test run completed successfully
- [ ] GPU has enough VRAM
- [ ] Resume checkpoint path tested
- [ ] Monitoring set up (W&B or local logging)

## Command Reference

```bash
# Standard training
python train_production.py --config configs/production_training.yaml

# Resume from checkpoint
python train_production.py --config configs/production_training.yaml \
    --resume checkpoints_production/checkpoint-5000

# With W&B logging
wandb login
python train_production.py --config configs/production_training.yaml
```

---

**Ready to train!** The production script handles all the complexity - just configure `production_training.yaml` and run.
