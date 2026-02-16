#!/usr/bin/env python3
"""
Training script for Kolmogorov MoE Architecture.

Trains MoE transformer models on FineWeb dataset with:
- Hybrid Muon+AdamW optimizer (Muon for 2D/3D params, AdamW for 1D)
- Sequence packing for efficient tokenization
- Gradient accumulation and checkpointing
- torch.compile support
- WandB logging (optional)
- Graceful shutdown with checkpoint saving

Usage:
    python train.py --config configs/production_training.yaml
    python train.py --config configs/debug_no_mod.yaml --steps 1000
"""

import torch
import torch.nn.functional as F
import yaml
import argparse
import gc
import time
import os
import signal
import sys
import atexit
import hashlib
import json
import math
from collections import defaultdict
from tqdm import tqdm

# Reduce CUDA memory fragmentation - must be set before any CUDA operations
# expandable_segments: Allocates memory in expandable chunks instead of fixed pools
# max_split_size_mb: Prevents large allocations from fragmenting into small pieces
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
from tools.reallocate_experts import reallocate_experts, analyze_router_entropy
from datasets import load_dataset
from transformers import AutoTokenizer

# Optional WandB
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check_router_collapse_fast(model, device: str = 'cuda', entropy_threshold: float = 0.7):
    """
    Lightweight check for router collapse using router weights only.

    Instead of running forward passes, directly samples router outputs
    on random embeddings. Much faster than full forward passes.

    Args:
        model: The MoE model (handles both compiled and non-compiled)
        device: Device to run on
        entropy_threshold: Normalized entropy below this = collapsed

    Returns:
        (layer_stats, collapsed_layers, summary_str)
    """
    # Handle compiled models
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    was_training = base_model.training
    base_model.eval()

    # Find MoE config
    n_experts = 1
    top_k = 1
    moe_layers_info = []

    for layer_idx, layer in enumerate(base_model.layers):
        if hasattr(layer, 'use_moe') and layer.use_moe and hasattr(layer.ffn, 'n_experts'):
            n_experts = layer.ffn.n_experts
            top_k = layer.ffn.top_k
            moe_layers_info.append((layer_idx, layer.ffn))

    if n_experts == 1 or not moe_layers_info:
        if was_training:
            base_model.train()
        return {}, [], "No MoE layers found"

    d_model = base_model.config.d_model if hasattr(base_model, 'config') else base_model.layers[0].ffn.router.router[0].in_features

    layer_stats = {}
    collapsed_layers = []
    max_entropy = math.log(n_experts)

    # Sample router on random hidden states (just router, not full forward)
    n_samples = 512  # Number of random vectors to sample
    with torch.no_grad():
        # Generate random hidden states
        random_hidden = torch.randn(1, n_samples, d_model, device=device, dtype=torch.bfloat16)

        for layer_idx, moe_layer in moe_layers_info:
            router = moe_layer.router
            router_output = router(random_hidden)

            # Get expert selections
            if len(router_output) == 3:
                # Token-choice
                _, selected_experts, _ = router_output
                # Count expert usage
                expert_counts = torch.bincount(
                    selected_experts.flatten(),
                    minlength=n_experts
                ).float()
            else:
                # Expert-choice: each expert gets equal tokens by design
                expert_counts = torch.ones(n_experts, device=device)

            # Compute usage distribution
            total = expert_counts.sum()
            if total == 0:
                continue
            usage = expert_counts / total

            # Compute entropy
            log_usage = torch.log(usage + 1e-10)
            entropy = -(usage * log_usage).sum().item()
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Find max usage
            max_usage = usage.max().item()
            max_expert = usage.argmax().item()

            # Check for collapse
            is_collapsed = normalized_entropy < entropy_threshold or max_usage > 0.5

            layer_stats[layer_idx] = {
                'entropy': normalized_entropy,
                'max_usage': max_usage,
                'max_expert': max_expert,
                'is_collapsed': is_collapsed,
            }

            if is_collapsed:
                collapsed_layers.append(layer_idx)

    # Build summary string
    if collapsed_layers:
        summary = f"COLLAPSE DETECTED in layers {collapsed_layers}"
    else:
        avg_entropy = sum(s['entropy'] for s in layer_stats.values()) / len(layer_stats) if layer_stats else 0
        summary = f"Routing healthy (avg entropy: {avg_entropy:.3f})"

    if was_training:
        base_model.train()
    return layer_stats, collapsed_layers, summary


# Global training state for signal handler access
_training_state = {
    'model': None,
    'optimizer': None,
    'step': 0,
    'epoch': 0,
    'data_idx': 0,
    'losses': [],
    'total_tokens': 0,
    'config_path': None,
    'checkpoint_dir': None,
    'should_stop': False,
    'checkpoint_saved': False,
    'args': None,
}


def get_model_state_dict(model):
    """Get state dict, handling compiled models."""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    return model.state_dict()


def save_checkpoint(state, path, reason="scheduled"):
    """Save training checkpoint."""
    if state['model'] is None:
        print(f"\n[Checkpoint] No model to save")
        return False

    checkpoint = {
        'step': state['step'],
        'epoch': state['epoch'],
        'data_idx': state['data_idx'],
        'losses': state['losses'][-1000:],  # Keep last 1000 losses
        'total_tokens': state['total_tokens'],
        'config_path': state['config_path'],
        'model_state_dict': get_model_state_dict(state['model']),
        'optimizer_state_dict': state['optimizer'].state_dict() if state['optimizer'] else None,
    }

    torch.save(checkpoint, path)
    loss_str = f"{state['losses'][-1]:.4f}" if state['losses'] else 'N/A'
    print(f"\n[Checkpoint] Saved ({reason}): {path}")
    print(f"  Step: {state['step']}, Epoch: {state['epoch']}, Loss: {loss_str}")
    state['checkpoint_saved'] = True
    return True


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    sig_name = signal.Signals(signum).name
    print(f"\n\n[Signal] Received {sig_name}, saving checkpoint and exiting...")
    _training_state['should_stop'] = True

    # Save checkpoint
    if not _training_state['checkpoint_saved'] and _training_state['model'] is not None:
        ckpt_path = os.path.join(
            _training_state['checkpoint_dir'],
            f"checkpoint-{_training_state['step']}-interrupted.pt"
        )
        save_checkpoint(_training_state, ckpt_path, reason=f"interrupted by {sig_name}")

    sys.exit(0)


def get_cache_path(tokenizer_name: str, seq_len: int, max_examples: int, pack_sequences: bool) -> str:
    """Generate cache file path based on tokenization parameters."""
    cache_dir = "./data/tokenized_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique key from parameters
    tokenizer_short = tokenizer_name.split("/")[-1]
    pack_str = "packed" if pack_sequences else "padded"
    cache_name = f"fineweb_{tokenizer_short}_{max_examples // 1000}k_seq{seq_len}_{pack_str}.pt"
    return os.path.join(cache_dir, cache_name)


def get_dataloader(tokenizer, seq_len, batch_size, max_examples=10000, pack_sequences=True, tokenizer_name="unknown"):
    """Load real text data from FineWeb with caching.

    Args:
        pack_sequences: If True, concatenate documents and chunk into fixed-length
                       sequences (no padding, ~100% efficiency). If False, pad
                       each document individually (wasteful for short docs).
        tokenizer_name: Name of tokenizer for cache key.
    """
    cache_path = get_cache_path(tokenizer_name, seq_len, max_examples, pack_sequences)

    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading cached tokenized data: {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        all_input_ids = cached['input_ids']
        all_attention_masks = cached['attention_masks']
        # Convert to tensors if not already (for efficient batch indexing)
        if not isinstance(all_input_ids, torch.Tensor):
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
        total_tokens = len(all_input_ids) * seq_len
        print(f"  Loaded {len(all_input_ids)} sequences ({total_tokens:,} tokens)")
        return all_input_ids, all_attention_masks

    print("Loading FineWeb dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=False,
    )
    dataset = dataset.select(range(min(max_examples, len(dataset))))

    print(f"Tokenizing {len(dataset)} examples...")

    if pack_sequences:
        # Pack documents together - no padding waste
        # Concatenate all tokens with EOS separator, then chunk
        all_tokens = []
        eos_token = tokenizer.eos_token_id or tokenizer.pad_token_id

        for i in tqdm(range(0, len(dataset), 100), desc="Tokenizing", unit="batch"):
            batch = dataset[i:i+100]
            for text in batch["text"]:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)
                all_tokens.append(eos_token)  # Separate documents

        # Chunk into fixed-length sequences
        all_input_ids = []
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            all_input_ids.append(all_tokens[i:i + seq_len])

        # No padding needed - all sequences are full
        all_attention_masks = [[1] * seq_len for _ in all_input_ids]

        total_tokens = len(all_input_ids) * seq_len
        print(f"Created {len(all_input_ids)} packed sequences (100% token efficiency)")
        print(f"  Total tokens: {total_tokens:,}")
    else:
        # Original approach - pad each document (wasteful)
        all_input_ids = []
        all_attention_masks = []
        for i in tqdm(range(0, len(dataset), 100), desc="Tokenizing", unit="batch"):
            batch = dataset[i:i+100]
            tokens = tokenizer(
                batch["text"],
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            )
            all_input_ids.extend(tokens["input_ids"])
            all_attention_masks.extend(tokens["attention_mask"])

        total_tokens = len(all_input_ids) * seq_len
        real_tokens = sum(sum(mask) for mask in all_attention_masks)
        padding_ratio = 1 - (real_tokens / total_tokens)
        print(f"Created {len(all_input_ids)} padded sequences")
        print(f"  Real tokens: {real_tokens:,} / {total_tokens:,} ({padding_ratio:.1%} padding)")

    # Save to cache
    print(f"Saving tokenized data to cache: {cache_path}")
    torch.save({
        'input_ids': all_input_ids,
        'attention_masks': all_attention_masks,
    }, cache_path)

    # Convert to tensors for efficient batch indexing (avoid Python list -> tensor per batch)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)

    return all_input_ids, all_attention_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/production_training.yaml')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb-project', type=str, default='Kolmogorov-Testing', help='WandB project name')
    parser.add_argument('--wandb-run', type=str, default=None, help='WandB run name')
    parser.add_argument('--grad-accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--grad-checkpoint', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--max-examples', type=int, default=200000, help='Max training examples to load')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Base checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=5000, help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help='Precision mode: bf16 (default), fp16 (weights/grads/optimizer all fp16)')
    parser.add_argument('--no-pack', action='store_true',
                        help='Disable sequence packing (pad each doc individually, less efficient)')
    parser.add_argument('--low-memory', action='store_true',
                        help='Aggressive memory optimization: clear cache frequently, reduce reserved memory')
    parser.add_argument('--lr-scale', type=float, default=1.0,
                        help='Scale learning rates by this factor (useful when resuming with different optimizer)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
    parser.add_argument('--reallocate', action='store_true',
                        help='Reallocate experts based on router entropy before training. '
                             'Use with --resume to fix collapsed routers while preserving attention weights.')
    parser.add_argument('--entropy-threshold', type=float, default=0.85,
                        help='Entropy threshold for healthy layers during reallocation (default: 0.85)')
    parser.add_argument('--balanced-routing', action='store_true',
                        help='Enable balanced routing with capacity constraints (prevents router collapse)')
    parser.add_argument('--collapse-check-interval', type=int, default=50,
                        help='Check for router collapse every N steps (0 to disable, default: 500)')
    parser.add_argument('--collapse-entropy-threshold', type=float, default=0.7,
                        help='Normalized entropy threshold for collapse detection (default: 0.7)')
    args = parser.parse_args()

    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
        print("Warning: CUDA not available, using CPU")

    # Create checkpoint directory with config name as subdirectory
    # e.g., configs/1b_150m_active.yaml -> checkpoints/1b_150m_active/
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    checkpoint_dir = os.path.join(args.checkpoint_dir, config_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup signal handlers for graceful shutdown
    _training_state['checkpoint_dir'] = checkpoint_dir
    _training_state['config_path'] = args.config
    _training_state['args'] = args
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 70)
    print(f"TRAINING: {config_name}")
    print("=" * 70)

    # Load config
    config_dict = load_config(args.config)
    model_cfg = config_dict['model']
    train_cfg = config_dict.get('training', {})

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.grad_accum
    tokens_per_step = effective_batch_size * args.seq_len

    print(f"\nConfig: {args.config}")
    print(f"Device: {device}")
    print(f"MoD enabled: {model_cfg.get('mod_enabled', True)}")
    print(f"MoD capacity: {model_cfg.get('mod_capacity_factor', 0.75)}")
    print(f"n_experts: {model_cfg.get('n_experts')}")
    print(f"n_layers: {model_cfg.get('n_layers')}")
    print(f"d_model: {model_cfg.get('d_model')}")
    print(f"MoE routing: {model_cfg.get('moe_routing', 'token_choice')}")
    if args.balanced_routing or model_cfg.get('moe_balanced_routing', False):
        print(f"  Balanced routing: ENABLED (capacity constraints per expert)")
    if args.collapse_check_interval > 0:
        print(f"  Collapse detection: every {args.collapse_check_interval} steps (threshold: {args.collapse_entropy_threshold})")
    print(f"\nTraining settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Tokens per optimizer step: {tokens_per_step:,}")
    print(f"  torch.compile: {args.compile}")
    print(f"  Gradient checkpointing: {args.grad_checkpoint}")
    print(f"  Precision mode: {args.precision}")
    print(f"  Low memory mode: {args.low_memory}")
    if args.lr_scale != 1.0:
        print(f"  LR scale: {args.lr_scale}x")
    print(f"  Learning rate: {train_cfg.get('max_lr', 0.0003)}")

    # Initialize WandB
    if args.wandb:
        if not HAS_WANDB:
            print("\nWandB not installed. Install with: pip install wandb")
            args.wandb = False
        else:
            run_name = args.wandb_run or f"quick-train-{model_cfg.get('n_layers')}L-{model_cfg.get('n_experts')}E"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": model_cfg,
                    "training": {
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "grad_accum": args.grad_accum,
                        "effective_batch_size": effective_batch_size,
                        "tokens_per_step": tokens_per_step,
                        "compile": args.compile,
                        "grad_checkpoint": args.grad_checkpoint,
                        "precision": args.precision,
                        "lr": train_cfg.get('max_lr', 0.0003),
                    },
                },
            )
            print(f"\nWandB initialized: {args.wandb_project}/{run_name}")

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Performance settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine balanced routing: CLI overrides config
    balanced_routing = args.balanced_routing or model_cfg.get('moe_balanced_routing', False)

    # Create model config
    config = AdvancedMoEConfig(
        vocab_size=model_cfg.get('vocab_size', 50000),
        d_model=model_cfg.get('d_model', 2048),
        n_layers=model_cfg.get('n_layers', 32),
        n_heads=model_cfg.get('n_heads', 16),
        n_kv_heads=model_cfg.get('n_kv_heads', 4),
        head_dim=model_cfg.get('head_dim', 128),
        d_ff=model_cfg.get('d_ff', 5632),
        d_ff_expert=model_cfg.get('d_ff_expert', 2816),
        max_seq_len=args.seq_len,
        n_experts=model_cfg.get('n_experts', 16),
        moe_top_k=model_cfg.get('moe_top_k', 2),
        moe_capacity_factor=model_cfg.get('moe_capacity_factor', 1.25),
        moe_load_balance_loss_weight=model_cfg.get('moe_load_balance_loss_weight', 0.01),
        moe_router_z_loss_weight=model_cfg.get('moe_router_z_loss_weight', 0.001),
        moe_layers=tuple(model_cfg.get('moe_layers', [])),
        moe_implementation=model_cfg.get('moe_implementation', 'batched'),
        moe_routing=model_cfg.get('moe_routing', 'token_choice'),
        moe_balanced_routing=balanced_routing,
        mod_enabled=model_cfg.get('mod_enabled', True),
        mod_capacity_factor=model_cfg.get('mod_capacity_factor', 0.75),
        mod_router_hidden_dim=model_cfg.get('mod_router_hidden_dim', 128),
        mod_load_balance_loss_weight=model_cfg.get('mod_load_balance_loss_weight', 0.001),
        mamba_enabled=model_cfg.get('mamba_enabled', False),
        mamba_layers=tuple(model_cfg.get('mamba_layers', [])),
        n_pred_tokens=model_cfg.get('n_pred_tokens', 1),
        aux_loss_weights=tuple(model_cfg.get('aux_loss_weights', [1.0])),
        use_flash_attention=model_cfg.get('use_flash_attention', True),
        dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )

    # Create model
    print("\nCreating model...")
    model_dtype = torch.float16 if args.precision == 'fp16' else torch.bfloat16
    model = MoETransformer(config).to(device).to(model_dtype)
    model.train()
    print(f"  Model dtype: {model_dtype}")

    # Count parameters
    params = model.count_parameters()
    print(f"  Total params: {params['total_billions']:.3f}B")
    print(f"  Active params: {params['active_billions']:.3f}B ({params['sparsity']:.1%} sparsity)")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # Apply torch.compile
    if args.compile:
        print("\nCompiling model with torch.compile...")
        compile_start = time.time()
        model = torch.compile(model, mode="default")
        print(f"  Compilation setup done in {time.time() - compile_start:.1f}s")
        print("  (Full compilation happens on first forward pass)")

    # Load tokenizer and data
    data_cfg = config_dict.get('data', {})
    tokenizer_name = data_cfg.get('tokenizer_name', 'meta-llama/Llama-2-7b-hf')
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load real data (with caching)
    all_input_ids, all_attention_masks = get_dataloader(
        tokenizer, args.seq_len, args.batch_size,
        max_examples=args.max_examples,
        pack_sequences=not args.no_pack,
        tokenizer_name=tokenizer_name
    )
    num_examples = len(all_input_ids)
    examples_per_epoch = num_examples // args.batch_size

    print(f"\nDataset:")
    print(f"  Total examples: {num_examples:,}")
    print(f"  Examples per epoch: {examples_per_epoch:,}")
    print(f"  Optimizer steps per epoch: {examples_per_epoch // args.grad_accum:,}")

    # Create optimizer based on config
    opt_cfg = config_dict.get('optimizer', {})
    opt_type = opt_cfg.get('type', 'adamw').lower()
    weight_decay = opt_cfg.get('weight_decay', 0.01)

    # IMPORTANT: With torch.compile, we must use the ORIGINAL model's parameters
    # torch.compile wraps the model, and model.named_parameters() after compile
    # may return different Parameter objects than the ones being updated in backprop
    if hasattr(model, '_orig_mod'):
        param_model = model._orig_mod
        print("\n  [Note] Using _orig_mod parameters for optimizer (torch.compile detected)")
    else:
        param_model = model

    if opt_type == 'muon':
        from moe_arch.training.muon_optimizer import Muon

        # Separate params into 3 groups:
        # - 2D/3D: weight matrices -> Muon (with weight decay)
        # - 1D norm weights: -> AdamW with NO weight decay (critical!)
        # - 1D other (biases, embeddings): -> AdamW with weight decay
        #
        # Why no weight decay on norms?
        # - Norm weights are initialized to 1.0
        # - Weight decay pulls toward 0, fighting the learning signal
        # - With small LR (0.0003) + small gradients, they stay stuck at 1.0
        muon_params = []
        adamw_norm_params = []  # No weight decay
        adamw_other_params = []  # With weight decay
        muon_param_count = 0
        norm_param_count = 0
        other_param_count = 0

        for name, param in param_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
                muon_param_count += param.numel()
            elif 'norm' in name.lower():
                # Norm weights: no weight decay
                adamw_norm_params.append(param)
                norm_param_count += param.numel()
            else:
                # Other 1D params (biases, etc): with weight decay
                adamw_other_params.append(param)
                other_param_count += param.numel()

        muon_lr = opt_cfg.get('muon_lr', 0.02) * args.lr_scale
        adamw_lr = opt_cfg.get('adamw_lr', 0.0003) * args.lr_scale
        # Use higher LR for norm weights since they need to move from 1.0
        norm_lr = opt_cfg.get('norm_lr', adamw_lr * 10) * args.lr_scale
        momentum = opt_cfg.get('momentum', 0.95)
        nesterov = opt_cfg.get('nesterov', True)
        betas = tuple(opt_cfg.get('adamw_betas', [0.9, 0.95]))

        # Create optimizers
        muon_opt = Muon(
            muon_params,
            lr=muon_lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )

        # AdamW for 1D params with param groups
        adamw_param_groups = []
        if adamw_norm_params:
            adamw_param_groups.append({
                'params': adamw_norm_params,
                'lr': norm_lr,
                'weight_decay': 0.0,  # Critical: no weight decay on norms!
            })
        if adamw_other_params:
            adamw_param_groups.append({
                'params': adamw_other_params,
                'lr': adamw_lr,
                'weight_decay': weight_decay,
            })

        adamw_opt = torch.optim.AdamW(
            adamw_param_groups,
            betas=betas,
        )

        # Combined optimizer wrapper
        class CombinedOptimizer:
            def __init__(self, optimizers):
                self.optimizers = optimizers

            def zero_grad(self, set_to_none=False):
                for opt in self.optimizers:
                    opt.zero_grad(set_to_none=set_to_none)

            def step(self):
                for opt in self.optimizers:
                    opt.step()

            @property
            def param_groups(self):
                groups = []
                for opt in self.optimizers:
                    groups.extend(opt.param_groups)
                return groups

            def state_dict(self):
                return [opt.state_dict() for opt in self.optimizers]

            def load_state_dict(self, state_dicts):
                # Handle backward compatibility with old checkpoints
                if isinstance(state_dicts, list) and len(state_dicts) == len(self.optimizers):
                    # New format: list of state dicts
                    for opt, state in zip(self.optimizers, state_dicts):
                        # Save current param_group hyperparams (lr, weight_decay, etc.)
                        # These may have been updated in the code but checkpoint has old values
                        saved_hyperparams = []
                        for pg in opt.param_groups:
                            saved_hyperparams.append({
                                k: v for k, v in pg.items() if k != 'params'
                            })

                        # Load optimizer state (momentum buffers, etc.)
                        opt.load_state_dict(state)

                        # Restore our NEW hyperparameters (don't use checkpoint's old values)
                        for pg, saved in zip(opt.param_groups, saved_hyperparams):
                            pg.update(saved)

                    print("  [Note] Loaded optimizer state but preserved current hyperparameters (lr, wd)")
                else:
                    # Old format: single optimizer state dict (incompatible with hybrid)
                    print("  [Warning] Checkpoint has old optimizer format (single optimizer).")
                    print("            Skipping optimizer state load - starting fresh.")
                    print("            This is expected when switching to hybrid Muon+AdamW.")

        optimizer = CombinedOptimizer([muon_opt, adamw_opt])
        lr = muon_lr  # Use Muon lr for logging (handles most params)
        print(f"  Optimizer: Muon+AdamW hybrid")
        print(f"    Muon: {len(muon_params)} params ({muon_param_count:,} weights, lr={muon_lr}, wd={weight_decay})")
        print(f"    AdamW (norms): {len(adamw_norm_params)} params ({norm_param_count:,} weights, lr={norm_lr}, wd=0)")
        print(f"    AdamW (other): {len(adamw_other_params)} params ({other_param_count:,} weights, lr={adamw_lr}, wd={weight_decay})")

        # Verify norm params are in optimizer and will be updated
        if adamw_norm_params:
            test_norm = adamw_norm_params[0]
            print(f"    [Verify] First norm param in optimizer: shape={test_norm.shape}, mean={test_norm.mean().item():.4f}, id={id(test_norm)}")

            # Also verify this matches what's in the model
            actual_norm = param_model.layers[0].seq_norm.weight
            print(f"    [Verify] First norm param in model:     shape={actual_norm.shape}, mean={actual_norm.mean().item():.4f}, id={id(actual_norm)}")
            if id(test_norm) != id(actual_norm):
                print(f"    [ERROR] Parameter ID mismatch! Optimizer won't update model weights!")
            else:
                print(f"    [OK] Parameter IDs match - optimizer should update model correctly")
    else:
        # AdamW with proper param groups (same separation as Muon path)
        # - Norm weights: no weight decay, higher LR
        # - Other params: with weight decay
        lr = opt_cfg.get('adamw_lr', train_cfg.get('max_lr', 0.0003)) * args.lr_scale
        norm_lr = opt_cfg.get('norm_lr', lr * 10) * args.lr_scale  # 10x LR for norms
        betas = tuple(opt_cfg.get('adamw_betas', [0.9, 0.95]))

        # Separate params into groups
        norm_params = []
        other_params = []
        norm_param_count = 0
        other_param_count = 0

        for name, param in param_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 and 'norm' in name.lower():
                norm_params.append(param)
                norm_param_count += param.numel()
            else:
                other_params.append(param)
                other_param_count += param.numel()

        param_groups = [
            {'params': other_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': norm_params, 'lr': norm_lr, 'weight_decay': 0.0},  # No weight decay on norms!
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=betas,
            fused=True,
        )
        print(f"  Optimizer: AdamW")
        print(f"    Main params: {other_param_count:,} weights, lr={lr}, wd={weight_decay}")
        print(f"    Norm params: {norm_param_count:,} weights, lr={norm_lr}, wd=0.0")

    # Training state
    losses = []
    epoch_losses = []
    current_epoch_losses = []
    data_idx = 0
    micro_step = 0
    current_epoch = 0
    start_step = 0
    total_tokens = 0

    # Handle expert reallocation if requested
    if args.reallocate:
        if not args.resume:
            print("\n[Error] --reallocate requires --resume to specify source checkpoint")
            sys.exit(1)
        if not os.path.exists(args.resume):
            print(f"\n[Error] Checkpoint not found: {args.resume}")
            sys.exit(1)

        # Create reallocated checkpoint path
        reallocated_path = os.path.join(
            checkpoint_dir,
            f"reallocated-{os.path.basename(args.resume)}"
        )

        print(f"\n{'='*60}")
        print("EXPERT REALLOCATION MODE")
        print(f"{'='*60}")
        print(f"Source checkpoint: {args.resume}")
        print(f"Output checkpoint: {reallocated_path}")
        print(f"Entropy threshold: {args.entropy_threshold}")

        # Run reallocation
        reallocate_experts(
            checkpoint_path=args.resume,
            config_path=args.config,
            output_path=reallocated_path,
            device=device,
            entropy_threshold=args.entropy_threshold,
        )

        # Use the reallocated checkpoint for training
        args.resume = reallocated_path
        print(f"\nProceeding with reallocated checkpoint...")

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

            # Load model state
            if hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state (skip if this is a reallocated checkpoint)
            is_reallocated = checkpoint.get('reallocated_from') is not None
            if checkpoint.get('optimizer_state_dict') and not is_reallocated:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            elif is_reallocated:
                print("  [Note] Skipping optimizer state (reallocated checkpoint)")

            # Restore training state (reallocated checkpoints start from step 0)
            start_step = checkpoint.get('step', 0)
            current_epoch = checkpoint.get('epoch', 0)
            data_idx = checkpoint.get('data_idx', 0)
            losses = checkpoint.get('losses', [])
            total_tokens = checkpoint.get('total_tokens', 0)

            print(f"  Resumed at step {start_step}, epoch {current_epoch}")
            print(f"  Last loss: {losses[-1]:.4f}" if losses else "  Last loss: N/A")

            if is_reallocated:
                print(f"  [Note] Reallocated from: {checkpoint.get('reallocated_from')}")
                print(f"  [Note] Collapsed layers fixed: {checkpoint.get('reallocated_collapsed_layers')}")
        else:
            print(f"\n[Warning] Checkpoint not found: {args.resume}")

    # Update global training state for signal handler
    _training_state['model'] = model
    _training_state['optimizer'] = optimizer
    _training_state['losses'] = losses
    _training_state['data_idx'] = data_idx
    _training_state['epoch'] = current_epoch
    _training_state['total_tokens'] = total_tokens

    print(f"\nTraining for {args.steps} optimizer steps (starting from {start_step})...")
    if args.save_interval > 0:
        print(f"Checkpoints: every {args.save_interval} steps to {checkpoint_dir}")
    print("-" * 70)

    # Timing
    start_time = time.time()
    step_start_time = time.time()
    log_interval = 10

    # Warmup for torch.compile (first few steps are slow)
    if args.compile:
        print("Running warmup steps for torch.compile...")

    # Progress bar
    pbar = tqdm(
        range(start_step, args.steps),
        initial=start_step,
        total=args.steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )

    for opt_step in pbar:
        # Check for graceful shutdown
        if _training_state['should_stop']:
            tqdm.write("\n[Training] Stopping due to signal...")
            break
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_lm_loss = 0.0
        step_aux_loss = 0.0

        # Gradient accumulation loop
        for accum_step in range(args.grad_accum):
            # Build batch using efficient tensor indexing
            batch_indices = torch.arange(data_idx, data_idx + args.batch_size) % num_examples
            input_ids = all_input_ids[batch_indices].to(device)
            attention_mask = all_attention_masks[batch_indices].to(device)

            # Check for epoch boundaries
            old_epoch = data_idx // num_examples
            data_idx += args.batch_size
            new_epoch = data_idx // num_examples
            if new_epoch > old_epoch and len(current_epoch_losses) > 0:
                epoch_avg = sum(current_epoch_losses) / len(current_epoch_losses)
                epoch_losses.append(epoch_avg)
                tqdm.write(f"\n{'='*70}")
                tqdm.write(f"EPOCH {current_epoch} COMPLETE - Average Loss: {epoch_avg:.4f}")
                if len(epoch_losses) > 1:
                    delta = epoch_avg - epoch_losses[-2]
                    tqdm.write(f"  Change from previous epoch: {delta:+.4f}")
                tqdm.write(f"{'='*70}\n")

                if args.wandb:
                    wandb.log({
                        "epoch": current_epoch,
                        "epoch/avg_loss": epoch_avg,
                    }, step=opt_step)

                current_epoch_losses = []
                current_epoch += 1
            # Forward (pass attention_mask so padding tokens are ignored in loss)
            # return_logits=False saves ~500MB per batch by not returning full vocab logits
            # Note: pass input_ids directly as labels (no clone needed - model doesn't modify in-place)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids, return_logits=False)
            loss = outputs['loss'] / args.grad_accum

            # Extract scalar values BEFORE backward to avoid keeping graph alive
            # Batch all GPU->CPU transfers into single sync point
            lm_loss_tensor = outputs.get('lm_loss', outputs['loss'])
            aux_loss_tensor = outputs.get('aux_loss', None)
            tokens_tensor = attention_mask.sum()

            # Stack scalars and sync once (reduces 4 .item() calls to 1 .tolist())
            if aux_loss_tensor is not None and isinstance(aux_loss_tensor, torch.Tensor):
                scalars = torch.stack([loss * args.grad_accum, lm_loss_tensor, aux_loss_tensor, tokens_tensor.float()])
                loss_val, lm_loss_raw, aux_loss_raw, tokens_in_batch = scalars.tolist()
                lm_loss_val = lm_loss_raw / args.grad_accum
                aux_loss_val = aux_loss_raw / args.grad_accum
            else:
                scalars = torch.stack([loss * args.grad_accum, lm_loss_tensor, tokens_tensor.float()])
                loss_val, lm_loss_raw, tokens_in_batch = scalars.tolist()
                lm_loss_val = lm_loss_raw / args.grad_accum
                aux_loss_val = (aux_loss_tensor / args.grad_accum) if aux_loss_tensor else 0.0

            # Clear outputs dict immediately to free memory
            del outputs, aux_loss_tensor, lm_loss_tensor, tokens_tensor, scalars

            # Backward
            loss.backward()

            # Accumulate metrics
            step_loss += loss_val
            step_lm_loss += lm_loss_val
            step_aux_loss += aux_loss_val
            micro_step += 1
            total_tokens += int(tokens_in_batch)

            # Free all tensors from this micro-step
            del loss, input_ids, attention_mask

        # Debug: find layers with large gradients (before clipping)
        # Run every 100 steps or if we detect explosion
        if (opt_step + 1) % 100 == 0 or opt_step < 5:
            grad_info = []
            total_norm_sq = 0.0
            for name, p in param_model.named_parameters():
                if p.grad is not None:
                    g_norm = p.grad.norm().item()
                    total_norm_sq += g_norm ** 2
                    if g_norm > 50:  # Flag large gradients
                        grad_info.append((name, g_norm, p.shape))

            total_norm = total_norm_sq ** 0.5
            if total_norm > 100 or opt_step < 5:
                tqdm.write(f"\n[Grad Debug] Total norm: {total_norm:.1f}")
                if grad_info:
                    grad_info.sort(key=lambda x: -x[1])  # Sort by magnitude
                    tqdm.write(f"  Top layers with large gradients (>50):")
                    for name, g_norm, shape in grad_info[:10]:  # Top 10
                        tqdm.write(f"    {name}: {g_norm:.1f} (shape={shape})")
                else:
                    tqdm.write(f"  No individual layer >50, but total is high - many small contributions")

        # Gradient clipping - use param_model to match optimizer's parameters
        grad_norm = torch.nn.utils.clip_grad_norm_(param_model.parameters(), 1.0)
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        # Debug: check if clipping is destroying norm gradients
        if opt_step < 3 or (opt_step + 1) % 500 == 0:
            clip_factor = 1.0 / grad_norm_val if grad_norm_val > 1.0 else 1.0
            norm_weight = param_model.layers[0].seq_norm.weight
            if norm_weight.grad is not None:
                post_clip_grad = norm_weight.grad.norm().item()
                # Calculate expected update magnitude
                # For AdamW: update ≈ lr * grad / sqrt(exp_avg_sq + eps)
                # Early in training, update ≈ lr * grad
                opt_to_check = optimizer.optimizers[1] if opt_type == 'muon' else optimizer
                norm_lr = None
                for pg in opt_to_check.param_groups:
                    if pg.get('weight_decay', -1) == 0.0:  # Norm group has wd=0
                        norm_lr = pg['lr']
                        break
                if norm_lr is None:
                    norm_lr = opt_to_check.param_groups[-1]['lr']  # Fallback to last group

                expected_update = post_clip_grad * norm_lr
                bf16_min_step = 2**-7  # ~0.0078, minimum distinguishable step at scale 1.0

                tqdm.write(f"\n[Clip Debug] Step {opt_step}: gnorm={grad_norm_val:.1f}, clip_factor={clip_factor:.2e}")
                tqdm.write(f"  Norm grad after clip: {post_clip_grad:.2e}")
                tqdm.write(f"  Norm LR: {norm_lr:.2e}")
                tqdm.write(f"  Expected update: {expected_update:.2e}")
                tqdm.write(f"  BF16 min step at 1.0: {bf16_min_step:.2e}")
                if expected_update < bf16_min_step:
                    tqdm.write(f"  [PROBLEM] Update smaller than BF16 precision - will round to 0!")

        del grad_norm

        # Optimizer step
        optimizer.step()

        # After first step: verify optimizer is actually updating parameters
        if opt_step == start_step:
            base_model = param_model  # Use same model as optimizer
            norm_weight = base_model.layers[0].seq_norm.weight
            norm_mean = norm_weight.mean().item()
            norm_std = norm_weight.std().item()
            if abs(norm_mean - 1.0) < 1e-6:
                tqdm.write(f"\n[WARNING] After step 1, norm weights still at 1.0!")
                tqdm.write(f"          Optimizer may not be updating 1D params correctly.")
                tqdm.write(f"          norm_weight.grad exists: {norm_weight.grad is not None}")
                if norm_weight.grad is not None:
                    tqdm.write(f"          grad_norm: {norm_weight.grad.norm().item():.2e}")
                    tqdm.write(f"          grad_mean: {norm_weight.grad.mean().item():.2e}")
                    tqdm.write(f"          grad_min/max: {norm_weight.grad.min().item():.2e} / {norm_weight.grad.max().item():.2e}")

                # Deep debug for both optimizer types
                opt_to_check = optimizer.optimizers[1] if opt_type == 'muon' else optimizer
                opt_name = "AdamW (from hybrid)" if opt_type == 'muon' else "AdamW"

                tqdm.write(f"          Checking {opt_name} optimizer...")
                tqdm.write(f"          param_groups: {len(opt_to_check.param_groups)}")
                for i, pg in enumerate(opt_to_check.param_groups):
                    tqdm.write(f"            Group {i}: {len(pg['params'])} params, lr={pg['lr']:.2e}, wd={pg.get('weight_decay', 'N/A')}")

                # Check if norm_weight is in any param group
                found_in_optimizer = False
                for pg in opt_to_check.param_groups:
                    for p in pg['params']:
                        if p is norm_weight:
                            found_in_optimizer = True
                            tqdm.write(f"          norm_weight IS in optimizer (same object)")
                            # Check optimizer state for this param
                            if p in opt_to_check.state:
                                state = opt_to_check.state[p]
                                tqdm.write(f"          Optimizer state keys: {list(state.keys())}")
                                if 'step' in state:
                                    tqdm.write(f"          step={state['step']}")
                                if 'exp_avg' in state:
                                    tqdm.write(f"          exp_avg norm: {state['exp_avg'].norm().item():.2e}")
                                if 'exp_avg_sq' in state:
                                    tqdm.write(f"          exp_avg_sq norm: {state['exp_avg_sq'].norm().item():.2e}")
                            else:
                                tqdm.write(f"          [BUG] No optimizer state for this param!")
                            break
                        elif p.shape == norm_weight.shape and p.data_ptr() == norm_weight.data_ptr():
                            found_in_optimizer = True
                            tqdm.write(f"          norm_weight found via data_ptr (same storage, different object)")
                            break

                if not found_in_optimizer:
                    tqdm.write(f"          [BUG] norm_weight NOT in optimizer!")
                    tqdm.write(f"          norm_weight id: {id(norm_weight)}, data_ptr: {norm_weight.data_ptr()}")
                    tqdm.write(f"          Searching for similar params in optimizer...")
                    for pg in opt_to_check.param_groups:
                        for p in pg['params']:
                            if p.shape == norm_weight.shape:
                                tqdm.write(f"            Found param with same shape: id={id(p)}, data_ptr={p.data_ptr()}")

                # Also check: did backward actually populate gradients?
                tqdm.write(f"\n          Checking all norm params for gradients:")
                norm_with_grad = 0
                norm_without_grad = 0
                for name, p in param_model.named_parameters():
                    if 'norm' in name.lower() and p.ndim == 1:
                        if p.grad is not None:
                            norm_with_grad += 1
                        else:
                            norm_without_grad += 1
                            tqdm.write(f"            [NO GRAD] {name}")
                tqdm.write(f"          Norm params with grad: {norm_with_grad}, without: {norm_without_grad}")

            else:
                tqdm.write(f"\n[OK] Norm weights updating: mean={norm_mean:.6f}, std={norm_std:.6f} (was 1.0)")

        # Memory cleanup periodically (every 100 steps) - doing this every step is expensive
        if (opt_step + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        losses.append(step_loss)
        current_epoch_losses.append(step_loss)

        # Update global training state
        _training_state['step'] = opt_step + 1
        _training_state['epoch'] = current_epoch
        _training_state['data_idx'] = data_idx
        _training_state['losses'] = losses
        _training_state['total_tokens'] = total_tokens
        _training_state['checkpoint_saved'] = False  # Reset for next interval

        # Save checkpoint at intervals
        if args.save_interval > 0 and (opt_step + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{opt_step + 1}.pt")
            save_checkpoint(_training_state, ckpt_path, reason="interval")

        # Calculate metrics
        step_time = time.time() - step_start_time
        tokens_per_sec = tokens_per_step / step_time if step_time > 0 else 0
        elapsed = time.time() - start_time

        # Log to WandB
        if args.wandb:
            wandb.log({
                "step": opt_step,
                "loss": step_loss,
                "lm_loss": step_lm_loss,
                "aux_loss": step_aux_loss,
                "grad_norm": grad_norm_val,
                "tokens_per_sec": tokens_per_sec,
                "total_tokens": total_tokens,
                "lr": lr,
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            }, step=opt_step)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{step_loss:.3f}',
            'lr': f'{lr:.1e}',
            'tok/s': f'{tokens_per_sec:,.0f}',
        })

        # Print detailed progress every log_interval steps
        if (opt_step + 1) % log_interval == 0 or opt_step == 0:
            avg_loss = sum(losses[-log_interval:]) / len(losses[-log_interval:])
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9

            tqdm.write(f"Step {opt_step+1:5d} | "
                       f"loss={step_loss:.4f} avg{log_interval}={avg_loss:.4f} | "
                       f"lm={step_lm_loss:.4f} aux={step_aux_loss:.4f} | "
                       f"gnorm={grad_norm_val:.2f} | "
                       f"lr={lr:.2e} | "
                       f"{tokens_per_sec:,.0f} tok/s | "
                       f"mem={mem_alloc:.1f}/{mem_reserved:.1f}GB | "
                       f"epoch={current_epoch}")

        # Periodic sanity check: verify norm weights are being updated (every 500 steps)
        if (opt_step + 1) % 500 == 0:
            base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            norm_weight = base_model.layers[0].seq_norm.weight
            norm_mean = norm_weight.mean().item()
            norm_std = norm_weight.std().item()
            # Also check gradient if available
            grad_info = ""
            if norm_weight.grad is not None:
                grad_norm = norm_weight.grad.norm().item()
                grad_info = f", grad_norm={grad_norm:.2e}"
            elif hasattr(norm_weight, '_grad') and norm_weight._grad is not None:
                grad_norm = norm_weight._grad.norm().item()
                grad_info = f", grad_norm={grad_norm:.2e}"

            # Use looser tolerance for bf16 (3-4 significant digits)
            if abs(norm_mean - 1.0) < 1e-3 and norm_std < 1e-3:
                tqdm.write(f"  [WARNING] RMSNorm weights ~1.0 (mean={norm_mean:.6f}, std={norm_std:.6f}{grad_info})")
                tqdm.write(f"            dtype={norm_weight.dtype}, requires_grad={norm_weight.requires_grad}")
            else:
                tqdm.write(f"  [Norm check] layer0.seq_norm: mean={norm_mean:.4f}, std={norm_std:.4f}{grad_info}")

        # Periodic router collapse check (lightweight - just router, no full forward)
        if args.collapse_check_interval > 0 and (opt_step + 1) % args.collapse_check_interval == 0:
            layer_stats, collapsed_layers, summary = check_router_collapse_fast(
                model,
                device=device,
                entropy_threshold=args.collapse_entropy_threshold,
            )

            if collapsed_layers:
                tqdm.write(f"\n  [ROUTER COLLAPSE WARNING] {summary}")
                for layer_idx in collapsed_layers:
                    stats = layer_stats[layer_idx]
                    tqdm.write(f"    Layer {layer_idx}: entropy={stats['entropy']:.3f}, "
                              f"max_usage={stats['max_usage']:.1%} (expert {stats['max_expert']})")
                tqdm.write(f"  Consider: --balanced-routing, or switch to expert_choice routing\n")
            else:
                # Log average entropy
                if layer_stats:
                    avg_entropy = sum(s['entropy'] for s in layer_stats.values()) / len(layer_stats)
                    max_usage = max(s['max_usage'] for s in layer_stats.values())
                    tqdm.write(f"  [Router check] {summary} (max_usage: {max_usage:.1%})")

            # Log to WandB
            if args.wandb and layer_stats:
                avg_entropy = sum(s['entropy'] for s in layer_stats.values()) / len(layer_stats)
                max_usage = max(s['max_usage'] for s in layer_stats.values())
                wandb.log({
                    "router/avg_entropy": avg_entropy,
                    "router/max_usage": max_usage,
                    "router/collapsed_layers": len(collapsed_layers),
                }, step=opt_step)

        step_start_time = time.time()

    # Close progress bar
    pbar.close()

    # Training complete
    total_time = time.time() - start_time
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    # Save final checkpoint
    final_step = opt_step + 1 if 'opt_step' in dir() else start_step
    final_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{final_step}-final.pt")
    _training_state['step'] = final_step
    save_checkpoint(_training_state, final_ckpt_path, reason="training complete")

    print("-" * 70)
    print("\nTRAINING COMPLETE")
    print("=" * 70)

    # Final epoch stats
    if len(current_epoch_losses) > 0:
        final_epoch_avg = sum(current_epoch_losses) / len(current_epoch_losses)
        epoch_losses.append(final_epoch_avg)
        print(f"\nFinal partial epoch {current_epoch} - Average Loss: {final_epoch_avg:.4f}")

    # Summary statistics
    print(f"\nResults:")
    if losses:
        print(f"  Initial loss (step 1):  {losses[0]:.4f}")
        print(f"  Final loss (step {final_step}): {losses[-1]:.4f}")
        print(f"  Change: {losses[-1] - losses[0]:+.4f}")
    else:
        print("  No training steps completed")

    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average throughput: {avg_tokens_per_sec:,.0f} tokens/sec")
    print(f"  Steps completed: {args.steps}")
    print(f"  Epochs completed: {current_epoch} + partial")

    # Epoch loss progression
    if len(epoch_losses) > 0:
        print(f"\nEpoch Loss Progression:")
        for i, eloss in enumerate(epoch_losses):
            delta = ""
            if i > 0:
                d = eloss - epoch_losses[i-1]
                delta = f" ({d:+.4f})"
            status = "improving" if i > 0 and eloss < epoch_losses[i-1] else ""
            print(f"  Epoch {i}: {eloss:.4f}{delta} {status}")

    # Learning assessment
    if len(losses) >= 2:
        if losses[-1] < losses[0] - 0.5:
            print("\n[OK] Loss is DECREASING - model is learning!")
        elif losses[-1] < losses[0] - 0.1:
            print("\n[WARN] Loss decreased slightly - learning slowly")
        else:
            print("\n[ERROR] Loss NOT decreasing - something is wrong!")
            print("   Possible issues:")
            print("   - Learning rate too low or too high")
            print("   - MoD blocking gradient flow")
            print("   - Router collapse")
            print("   - Architecture issue")

        # Show loss curve
        print("\nLoss curve (every 100 steps):")
        for i in range(0, len(losses), 100):
            bar_len = int((12 - losses[i]) * 3)  # Scale for display
            bar = "#" * max(0, bar_len)
            print(f"  {i+1:5d}: {losses[i]:.2f} {bar}")

    # Print checkpoint info
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"  Resume with: python train.py --resume {final_ckpt_path} --config {args.config}")

    # Close WandB
    if args.wandb:
        wandb.log({
            "final/initial_loss": losses[0] if losses else 0,
            "final/final_loss": losses[-1] if losses else 0,
            "final/loss_change": (losses[-1] - losses[0]) if len(losses) > 1 else 0,
            "final/total_time_s": total_time,
            "final/avg_tokens_per_sec": avg_tokens_per_sec,
            "final/epochs_completed": current_epoch,
        })
        wandb.finish()
        print(f"\nWandB run completed: {args.wandb_project}")


if __name__ == '__main__':
    main()
