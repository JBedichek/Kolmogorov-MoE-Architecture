#!/usr/bin/env python3
"""
Distributed Training Script for MoE with Expert Parallelism.

Supports:
1. Expert Parallelism: Each GPU holds a subset of experts
2. FSDP: Fully Sharded Data Parallel for memory efficiency
3. Gradient checkpointing: Trade compute for memory

Usage:
    # Single node, multi-GPU with expert parallelism
    torchrun --nproc_per_node=8 train_distributed.py --config configs/production_training.yaml

    # Multi-node training
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_distributed.py --config configs/production_training.yaml
"""

import os
import sys
import yaml
import argparse
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import defaultdict
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.utils.data import DataLoader, IterableDataset

# Local imports
from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.layers import TransformerBlock


# =============================================================================
# Data Loading
# =============================================================================

class PackedSequenceDataset(IterableDataset):
    """
    Dataset that packs multiple sequences into fixed-length chunks.
    Maximizes GPU utilization by eliminating padding waste.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        tokenizer,
        max_seq_len: int,
        max_examples: Optional[int] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_examples = max_examples
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.eos_token_id = tokenizer.eos_token_id

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield packed sequences."""
        from datasets import load_dataset

        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split='train',
            streaming=True,
        )

        if self.world_size > 1:
            dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)

        token_buffer = []
        examples_seen = 0

        for example in dataset:
            if self.max_examples and examples_seen >= self.max_examples:
                break

            text = example.get('text', '')
            if not text.strip():
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(self.eos_token_id)
            token_buffer.extend(tokens)
            examples_seen += 1

            while len(token_buffer) >= self.max_seq_len:
                packed_tokens = token_buffer[:self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len:]

                input_ids = torch.tensor(packed_tokens, dtype=torch.long)
                labels = input_ids.clone()

                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                }

        # Yield remaining tokens
        if len(token_buffer) > 0:
            pad_len = self.max_seq_len - len(token_buffer)
            pad_id = self.tokenizer.pad_token_id or self.eos_token_id
            packed_tokens = token_buffer + [pad_id] * pad_len

            input_ids = torch.tensor(packed_tokens, dtype=torch.long)
            labels = input_ids.clone()
            labels[-pad_len:] = -100

            yield {
                'input_ids': input_ids,
                'labels': labels,
            }


def create_dataloader(
    config: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create dataloader with packed sequences."""
    data_config = config.get('data', {})

    dataset = PackedSequenceDataset(
        dataset_name=data_config.get('dataset_name', 'HuggingFaceFW/fineweb'),
        dataset_config=data_config.get('dataset_config', 'sample-10BT'),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_examples=data_config.get('max_examples'),
        rank=rank,
        world_size=world_size,
    )

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x['input_ids'] for x in batch])
        labels = torch.stack([x['labels'] for x in batch])
        return {'input_ids': input_ids, 'labels': labels}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def load_tokenizer(config: Dict[str, Any]):
    """Load tokenizer from config."""
    from transformers import AutoTokenizer

    data_config = config.get('data', {})
    tokenizer_name = data_config.get('tokenizer_name', 'meta-llama/Meta-Llama-3-8B')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: Dict[str, Any], rank: int, world_size: int) -> nn.Module:
    """Create model with expert parallelism configuration."""
    model_config = config.get('model', {})

    # Build AdvancedMoEConfig from YAML
    moe_config = AdvancedMoEConfig(
        vocab_size=model_config.get('vocab_size', 128256),
        d_model=model_config.get('d_model', 2048),
        n_layers=model_config.get('n_layers', 12),
        n_heads=model_config.get('n_heads', 16),
        n_kv_heads=model_config.get('n_kv_heads', 4),
        head_dim=model_config.get('head_dim', 128),
        d_ff=model_config.get('d_ff', 5504),
        max_seq_len=model_config.get('max_seq_len', 2048),
        n_experts=model_config.get('n_experts', 64),
        moe_top_k=model_config.get('moe_top_k', 2),
        moe_capacity_factor=model_config.get('moe_capacity_factor', 1.25),
        d_ff_expert=model_config.get('d_ff_expert', 1524),
        moe_load_balance_loss_weight=model_config.get('moe_load_balance_loss_weight', 0.01),
        moe_router_z_loss_weight=model_config.get('moe_router_z_loss_weight', 0.001),
        moe_layers=tuple(model_config.get('moe_layers', list(range(12)))),
        moe_implementation=model_config.get('moe_implementation', 'sparse'),
        mod_enabled=model_config.get('mod_enabled', False),
        mod_capacity_factor=model_config.get('mod_capacity_factor', 0.5),
        mamba_enabled=model_config.get('mamba_enabled', False),
        mamba_layers=tuple(model_config.get('mamba_layers', [])),
        n_pred_tokens=model_config.get('n_pred_tokens', 1),
        aux_loss_weights=tuple(model_config.get('aux_loss_weights', [1.0])),
        use_flash_attention=model_config.get('use_flash_attention', True),
        rope_theta=model_config.get('rope_theta', 10000.0),
        norm_type=model_config.get('norm_type', 'rmsnorm'),
        ffn_activation=model_config.get('ffn_activation', 'swiglu'),
        dropout=model_config.get('dropout', 0.0),
        attention_dropout=model_config.get('attention_dropout', 0.0),
        residual_dropout=model_config.get('residual_dropout', 0.0),
    )

    # For expert parallelism, we need n_experts divisible by world_size
    if moe_config.moe_implementation == 'expert_parallel':
        n_experts = moe_config.n_experts
        if n_experts % world_size != 0:
            # Adjust to nearest divisible count
            adjusted = (n_experts // world_size) * world_size
            if rank == 0:
                print(f"Adjusting n_experts from {n_experts} to {adjusted} for expert parallelism")
            moe_config.n_experts = adjusted

    # Create model
    model = MoETransformer(moe_config)

    return model, moe_config


def wrap_model_with_fsdp(
    model: nn.Module,
    rank: int,
    use_bf16: bool = True,
    cpu_offload: bool = False,
    activation_checkpointing: bool = True,
) -> FSDP:
    """
    Wrap model with FSDP for memory-efficient distributed training.

    Full sharding includes:
    - Parameters: Sharded across GPUs, gathered only when needed
    - Gradients: Sharded, reduced in chunks
    - Optimizer states: Sharded (each GPU only stores states for its shard)
    - Activations: Checkpointed (recomputed during backward to save memory)
    """

    # Mixed precision policy - use BF16 for everything
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        reduce_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        buffer_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )

    # Auto-wrap policy: wrap each TransformerBlock for memory efficiency
    auto_wrap_policy = ModuleWrapPolicy({TransformerBlock})

    # Optional CPU offload for extreme memory savings
    cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None

    # Wrap with FSDP - default to FULL_SHARD but can use NO_SHARD for debugging
    # NO_SHARD is essentially DDP wrapped in FSDP API
    sharding = ShardingStrategy.FULL_SHARD
    model = FSDP(
        model,
        sharding_strategy=sharding,  # Shard everything
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch for speed
        forward_prefetch=True,  # Prefetch next layer params during forward
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,  # Limit memory for all-gathers
        cpu_offload=cpu_offload_policy,
        use_orig_params=True,  # Better compatibility with optimizer
    )

    # Apply activation checkpointing to save activation memory
    if activation_checkpointing:
        # Checkpoint each TransformerBlock - recompute activations during backward
        check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
        if rank == 0:
            print("Activation checkpointing enabled for TransformerBlocks")

    return model


def create_optimizer(model: nn.Module, config: Dict[str, Any], use_8bit: bool = False) -> torch.optim.Optimizer:
    """Create optimizer from config.

    Args:
        model: The model to optimize
        config: Configuration dict
        use_8bit: If True, use 8-bit AdamW (4x memory savings for optimizer states)
    """
    opt_config = config.get('optimizer', {})
    training_config = config.get('training', {})

    opt_type = opt_config.get('type', 'adamw')
    lr = training_config.get('max_lr', 3e-3)
    weight_decay = opt_config.get('weight_decay', 0.01)

    # Separate weight decay for different param groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
            print("Using 8-bit AdamW (4x memory savings for optimizer states)")
            return optimizer
        except ImportError:
            print("Warning: bitsandbytes not available, falling back to standard AdamW")

    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        # Default to AdamW if Muon not available
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.95),
        )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    training_config = config.get('training', {})
    warmup_steps = training_config.get('warmup_steps', 500)
    min_lr_ratio = training_config.get('min_lr_ratio', 0.1)

    def lr_lambda(step):
        # Warmup
        if step < warmup_steps:
            return step / warmup_steps
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Debugging utilities for training diagnostics
# =============================================================================

def check_parameter_gradients(model: nn.Module, step: int, rank: int = 0) -> Dict[str, Any]:
    """
    Check that all parameters have gradients and are being updated.
    Returns diagnostics about gradient flow.
    """
    stats = {
        'total_params': 0,
        'params_with_grad': 0,
        'params_no_requires_grad': 0,  # requires_grad=False
        'params_none_grad': 0,          # requires_grad=True but grad=None
        'params_zero_grad': 0,
        'params_nan_grad': 0,
        'params_inf_grad': 0,
        'grad_norms': defaultdict(float),
        'param_norms': defaultdict(float),
        'module_grad_counts': defaultdict(lambda: {'total': 0, 'with_grad': 0}),
        'zero_grad_names': [],
        'no_grad_names': [],
        'none_grad_names': [],
    }

    for name, param in model.named_parameters():
        stats['total_params'] += 1

        # Determine module type - must match startup check patterns
        name_lower = name.lower()
        if 'mod_router' in name_lower or 'ffn_mod_router' in name_lower:
            module_type = 'mod_router'
        elif 'lm_head' in name_lower:
            module_type = 'lm_head'
        elif '.ffn.router' in name_lower or ('moe' in name_lower and 'router' in name_lower):
            module_type = 'moe_router'
        elif 'grouped_experts' in name_lower or 'local_experts' in name_lower:
            module_type = 'expert_weights'
        elif '.ffn.' in name_lower and any(f'.{w}' in name_lower for w in ['w1', 'w2', 'w3']):
            module_type = 'expert_weights'
        elif 'embed' in name_lower or 'tok_embed' in name_lower:
            module_type = 'embeddings'
        elif 'norm' in name_lower:
            module_type = 'norms'
        elif any(proj in name_lower for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj']):
            module_type = 'attention'
        elif 'seq_layer' in name_lower and 'ffn' not in name_lower:
            module_type = 'attention'
        else:
            module_type = 'other'

        stats['module_grad_counts'][module_type]['total'] += 1

        if not param.requires_grad:
            stats['params_no_requires_grad'] += 1
            stats['no_grad_names'].append(f"{name} (requires_grad=False)")
            continue

        if param.grad is None:
            stats['params_none_grad'] += 1
            stats['none_grad_names'].append(f"{name} (grad=None)")
            continue

        stats['params_with_grad'] += 1
        stats['module_grad_counts'][module_type]['with_grad'] += 1

        grad_norm = param.grad.float().norm().item()
        param_norm = param.float().norm().item()

        # Check for issues
        if grad_norm == 0:
            stats['params_zero_grad'] += 1
            stats['zero_grad_names'].append(name)
        if torch.isnan(param.grad).any():
            stats['params_nan_grad'] += 1
        if torch.isinf(param.grad).any():
            stats['params_inf_grad'] += 1

        stats['grad_norms'][module_type] += grad_norm ** 2
        stats['param_norms'][module_type] += param_norm ** 2

    # Combine no_grad lists
    stats['no_grad_names'] = stats['no_grad_names'] + stats['none_grad_names']

    # Convert to RMS
    for key in stats['grad_norms']:
        stats['grad_norms'][key] = math.sqrt(stats['grad_norms'][key])
        stats['param_norms'][key] = math.sqrt(stats['param_norms'][key])

    return stats


def collect_router_stats(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Collect MoE router statistics to diagnose routing issues.
    """
    stats = {
        'router_entropy': [],
        'expert_load': [],
        'top_expert_prob': [],
        'expert_usage_pct': [],
        'router_confidence': [],
    }

    # Get the underlying model from FSDP wrapper
    if hasattr(model, 'module'):
        base_model = model.module
    elif hasattr(model, '_fsdp_wrapped_module'):
        base_model = model._fsdp_wrapped_module
    else:
        base_model = model

    # Find MoE layers and collect stats
    input_ids = batch['input_ids'][:, :-1]

    with torch.no_grad():
        # Get embeddings
        if hasattr(base_model, 'embedding'):
            x = base_model.embedding(input_ids)
        elif hasattr(base_model, 'tok_embeddings'):
            x = base_model.tok_embeddings(input_ids)
        else:
            # Can't trace through model
            return stats

        # Go through layers and collect router stats
        for layer_idx, layer in enumerate(base_model.layers if hasattr(base_model, 'layers') else []):
            if hasattr(layer, 'ffn') and hasattr(layer.ffn, 'router') and layer.ffn.router is not None:
                # This is an MoE layer
                router = layer.ffn.router

                # Get hidden states at this layer (approximate - just use input)
                # In practice we'd need to run attention first, but this gives us router behavior

                # Compute router logits
                if hasattr(router, 'router'):
                    router_logits = router.router(x)  # (batch, seq, n_experts)
                else:
                    router_logits = router(x)
                    if isinstance(router_logits, tuple):
                        router_logits = router_logits[2]  # (weights, indices, logits)

                router_probs = F.softmax(router_logits, dim=-1)

                # Entropy: higher = more uniform distribution
                entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean().item()

                # Expert load: how many tokens each expert gets
                n_experts = router_probs.shape[-1]
                top_k = 2  # Assuming top_k=2
                _, top_indices = torch.topk(router_probs, top_k, dim=-1)
                expert_counts = torch.bincount(top_indices.reshape(-1), minlength=n_experts).float()
                expert_load = expert_counts / expert_counts.sum()

                # How many experts are actually used (>1% of tokens)
                experts_used = (expert_load > 0.01).sum().item()

                # Top expert probability (router confidence)
                top_prob = router_probs.max(dim=-1).values.mean().item()

                stats['router_entropy'].append(entropy)
                stats['expert_load'].append(expert_load.cpu().tolist())
                stats['top_expert_prob'].append(top_prob)
                stats['expert_usage_pct'].append(100 * experts_used / n_experts)
                stats['router_confidence'].append(top_prob)

                # Only check first MoE layer for efficiency
                break

    return stats


def print_debug_stats(param_stats: Dict, router_stats: Dict, step: int, rank: int = 0):
    """Print comprehensive debug statistics."""
    if rank != 0:
        return

    print(f"\n{'='*60}")
    print(f"DEBUG STATS - Step {step}")
    print(f"{'='*60}")

    # Parameter gradient stats
    print(f"\n--- Parameter Gradients ---")
    print(f"Total params: {param_stats['total_params']}")
    print(f"Params with grad: {param_stats['params_with_grad']}")
    print(f"Params requires_grad=False: {param_stats['params_no_requires_grad']}")
    print(f"Params grad=None: {param_stats['params_none_grad']}")
    print(f"Params zero grad: {param_stats['params_zero_grad']}")
    print(f"Params NaN grad: {param_stats['params_nan_grad']}")
    print(f"Params Inf grad: {param_stats['params_inf_grad']}")

    # Show gradient status by module
    print(f"\n--- Gradients by Module Type ---")
    for module_type, counts in sorted(param_stats['module_grad_counts'].items()):
        status = "✓" if counts['with_grad'] == counts['total'] else f"⚠️ {counts['with_grad']}/{counts['total']}"
        print(f"  {module_type:20s}: {counts['with_grad']:3d}/{counts['total']:3d} have gradients {status}")

    # Total params without gradients (either requires_grad=False or grad=None)
    params_no_grad = param_stats['params_no_requires_grad'] + param_stats['params_none_grad']
    if params_no_grad > 0:
        print(f"\n⚠️ WARNING: {params_no_grad} params have NO gradients (requires_grad=False or grad=None)!")
        if len(param_stats['no_grad_names']) <= 20:
            for name in param_stats['no_grad_names']:
                print(f"  - {name}")
        else:
            print(f"  (First 20 of {len(param_stats['no_grad_names'])})")
            for name in param_stats['no_grad_names'][:20]:
                print(f"  - {name}")

    if param_stats['params_zero_grad'] > 0:
        print(f"\n⚠️ WARNING: {param_stats['params_zero_grad']} params have ZERO gradients (grad exists but is 0)!")
        if len(param_stats['zero_grad_names']) <= 20:
            for name in param_stats['zero_grad_names']:
                print(f"  - {name}")
        else:
            print(f"  (First 20 of {len(param_stats['zero_grad_names'])})")
            for name in param_stats['zero_grad_names'][:20]:
                print(f"  - {name}")

    print(f"\n--- Gradient Norms by Module ---")
    for module_type, norm in sorted(param_stats['grad_norms'].items()):
        param_norm = param_stats['param_norms'].get(module_type, 0)
        ratio = norm / (param_norm + 1e-10)
        status = "OK" if 1e-6 < norm < 1e3 else "⚠️ CHECK"
        print(f"  {module_type:20s}: grad_norm={norm:.6f}, param_norm={param_norm:.4f}, ratio={ratio:.6f} {status}")

    # Router stats
    if router_stats.get('router_entropy'):
        print(f"\n--- MoE Router Stats ---")
        entropy = router_stats['router_entropy'][0]
        top_prob = router_stats['top_expert_prob'][0]
        usage_pct = router_stats['expert_usage_pct'][0]
        expert_load = router_stats['expert_load'][0]

        # Compute load balance metrics
        n_experts = len(expert_load)
        load_std = torch.tensor(expert_load).std().item()
        load_max = max(expert_load)
        load_min = min([x for x in expert_load if x > 0] or [0])
        ideal_load = 1.0 / n_experts

        print(f"  Router entropy: {entropy:.4f} (higher=more uniform, max={math.log(n_experts):.2f})")
        print(f"  Top expert prob: {top_prob:.4f} (lower=more distributed)")
        print(f"  Experts used (>1%): {usage_pct:.1f}% ({int(usage_pct * n_experts / 100)}/{n_experts})")
        print(f"  Load balance: std={load_std:.4f}, max={load_max:.4f}, min={load_min:.4f}, ideal={ideal_load:.4f}")

        # Diagnose issues
        if entropy < 1.0:
            print(f"  ⚠️ LOW ENTROPY: Router is collapsing! Consider increasing router temperature or entropy regularization.")
        if usage_pct < 50:
            print(f"  ⚠️ LOW EXPERT USAGE: Only {usage_pct:.0f}% of experts used. Router may be collapsed.")
        if top_prob > 0.8:
            print(f"  ⚠️ HIGH CONFIDENCE: Router is too confident ({top_prob:.2f}). May need more exploration.")
        if load_std > 0.1:
            print(f"  ⚠️ IMBALANCED LOAD: Expert load is imbalanced (std={load_std:.4f}). Check load balancing loss.")

        # Show expert distribution
        print(f"\n  Expert load distribution (sorted):")
        sorted_load = sorted(enumerate(expert_load), key=lambda x: -x[1])
        for i, (expert_id, load) in enumerate(sorted_load[:5]):
            bar = '█' * int(load * 50)
            print(f"    Expert {expert_id:2d}: {load:.4f} {bar}")
        if n_experts > 5:
            print(f"    ... ({n_experts - 5} more experts)")
            # Show least used
            print(f"    Least used:")
            for expert_id, load in sorted_load[-3:]:
                bar = '█' * int(load * 50)
                print(f"    Expert {expert_id:2d}: {load:.4f} {bar}")

    print(f"{'='*60}\n")


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    grad_clip: float,
    accumulation_steps: int,
    step: int,
    debug: bool = False,
    rank: int = 0,
) -> Dict[str, float]:
    """Single training step with gradient accumulation."""

    input_ids = batch['input_ids']
    labels = batch['labels'] if 'labels' in batch else input_ids.clone()

    # For causal LM: input is tokens[:-1], labels is tokens[1:]
    # The model predicts next token at each position
    inputs = input_ids[:, :-1]
    targets = labels[:, 1:]

    # Forward pass
    outputs = model(input_ids=inputs, labels=targets)
    loss = outputs['loss'] / accumulation_steps

    # Backward pass
    loss.backward()

    metrics = {
        'loss': outputs['loss'].item(),
        'lm_loss': outputs.get('lm_loss', outputs['loss']).item(),
        'aux_loss': outputs.get('aux_loss', 0.0) if isinstance(outputs.get('aux_loss'), float) else outputs.get('aux_loss', torch.tensor(0.0)).item(),
    }

    # Debug: check gradients and router stats
    if debug and (step + 1) % accumulation_steps == 0:
        param_stats = check_parameter_gradients(model, step, rank)
        router_stats = collect_router_stats(model, batch)
        print_debug_stats(param_stats, router_stats, step, rank)

    # Gradient accumulation
    if (step + 1) % accumulation_steps == 0:
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Distributed MoE Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--cpu-offload', action='store_true', help='Offload params to CPU (slower but uses less GPU memory)')
    parser.add_argument('--no-activation-checkpointing', action='store_true', help='Disable activation checkpointing')
    parser.add_argument('--8bit-optimizer', action='store_true', dest='use_8bit_optimizer', help='Use 8-bit AdamW (4x memory savings)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with gradient/router diagnostics')
    parser.add_argument('--debug-interval', type=int, default=100, help='How often to print debug stats (in steps)')
    parser.add_argument('--use-ddp', action='store_true', help='Use DDP instead of FSDP (for gradient debugging)')
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"Starting distributed training with {world_size} GPUs")
        print(f"Config: {args.config}")

    # Load config
    config = load_config(args.config)

    # Create model
    model, moe_config = create_model(config, rank, world_size)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created: {total_params/1e9:.2f}B parameters")
        print(f"MoE implementation: {moe_config.moe_implementation}")
        print(f"Experts: {moe_config.n_experts}, top_k: {moe_config.moe_top_k}")
        print(f"Multi-token prediction: {moe_config.n_pred_tokens} heads, weights: {moe_config.aux_loss_weights}")
        print(f"MoD enabled: {moe_config.mod_enabled}, capacity: {moe_config.mod_capacity_factor}")

    # Wrap with FSDP or DDP
    use_bf16 = config.get('mixed_precision', {}).get('bf16', True)

    if args.use_ddp:
        # Use simple DDP for gradient debugging
        model = model.cuda()
        if use_bf16:
            model = model.to(torch.bfloat16)
        model = DDP(model, device_ids=[local_rank])

        if rank == 0:
            print("Model wrapped with DDP (no sharding - for debugging)")
            print(f"  - Parameters: replicated across {world_size} GPUs")
            print(f"  - Gradients: all-reduced")
    else:
        # Use FSDP for memory efficiency
        model = wrap_model_with_fsdp(
            model,
            rank,
            use_bf16=use_bf16,
            cpu_offload=args.cpu_offload,
            activation_checkpointing=not args.no_activation_checkpointing,
        )

        if rank == 0:
            print("Model wrapped with FSDP (full sharding)")
            print(f"  - Parameters: sharded across {world_size} GPUs")
            print(f"  - Gradients: sharded and reduced in chunks")
            print(f"  - Optimizer states: sharded")
            print(f"  - Activations: {'checkpointed' if not args.no_activation_checkpointing else 'NOT checkpointed'}")
            print(f"  - CPU offload: {'enabled' if args.cpu_offload else 'disabled'}")

    # Startup check: verify all parameters require gradients
    if rank == 0:
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        param_groups = defaultdict(lambda: {'total': 0, 'trainable': 0, 'samples': []})

        for name, param in model.named_parameters():
            total_params += param.numel()
            name_lower = name.lower()

            # Determine param group - order matters (more specific first)
            # Note: With FSDP, parameter names may have prefixes like '_fsdp_wrapped_module.'
            # We use substring matching which works regardless of prefix
            if 'mod_router' in name_lower or 'ffn_mod_router' in name_lower:
                group = 'mod_router'
            elif 'lm_head' in name_lower:
                # Check lm_head BEFORE router to avoid issues with ordering
                group = 'lm_head'
            elif '.ffn.router' in name_lower or ('moe' in name_lower and 'router' in name_lower):
                # MoE router: inside .ffn.router or has 'moe' and 'router'
                group = 'moe_router'
            elif 'grouped_experts' in name_lower or 'local_experts' in name_lower:
                # Expert weights in grouped_experts or local_experts (expert parallel)
                group = 'expert_weights'
            elif '.ffn.' in name_lower and any(f'.{w}' in name_lower for w in ['w1', 'w2', 'w3']):
                # Expert weight matrices (w1, w2, w3) inside FFN
                group = 'expert_weights'
            elif 'embed' in name_lower or 'tok_embed' in name_lower:
                group = 'embeddings'
            elif 'norm' in name_lower:
                group = 'norms'
            elif any(proj in name_lower for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj']):
                group = 'attention'
            elif 'seq_layer' in name_lower and 'ffn' not in name_lower:
                # Attention weights inside seq_layer (not FFN)
                group = 'attention'
            else:
                group = 'other'

            param_groups[group]['total'] += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                param_groups[group]['trainable'] += param.numel()
            else:
                frozen_params += param.numel()

            # Keep a few sample names for debugging
            if len(param_groups[group]['samples']) < 3:
                param_groups[group]['samples'].append(name)

        print(f"\n--- Parameter Trainability Check ---")
        print(f"Note: With FSDP on {world_size} GPUs, each rank sees 1/{world_size} of total params")
        print(f"Total parameters (this shard): {total_params:,}")
        print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"\nBy module type:")
        for group, counts in sorted(param_groups.items()):
            if counts['total'] == 0:
                # No parameters found for this category - skip empty categories
                continue
            pct = 100 * counts['trainable'] / counts['total'] if counts['total'] > 0 else 0
            status = "✓" if pct == 100 else "⚠️ FROZEN" if pct == 0 else f"⚠️ {pct:.0f}%"
            print(f"  {group:20s}: {counts['trainable']:>12,} / {counts['total']:>12,} trainable {status}")

        # Show 'other' category details if it has significant params
        if param_groups['other']['total'] > 0:
            print(f"\n  'other' category samples: {param_groups['other']['samples'][:5]}")

        # Debug: show sample parameter names for verification
        print(f"\n  Sample parameter names by category:")
        for group in ['attention', 'embeddings', 'expert_weights', 'lm_head', 'mod_router', 'moe_router', 'norms']:
            samples = param_groups[group].get('samples', [])
            if samples:
                print(f"    {group}: {samples[0][:80]}...")

        if frozen_params > 0:
            print(f"\n⚠️ WARNING: {frozen_params:,} parameters are frozen and won't be trained!")
        print()

    # Create optimizer and scheduler
    training_config = config.get('training', {})
    optimizer = create_optimizer(model, config, use_8bit=args.use_8bit_optimizer)

    total_tokens = training_config.get('total_tokens', 10_000_000_000)
    batch_size = training_config.get('batch_size', 1)
    seq_len = moe_config.max_seq_len
    accum_steps = training_config.get('gradient_accumulation_steps', 4)
    tokens_per_step = batch_size * seq_len * world_size * accum_steps
    total_steps = total_tokens // tokens_per_step

    scheduler = create_scheduler(optimizer, config, total_steps)

    if rank == 0:
        print(f"Training for {total_steps} steps ({total_tokens/1e9:.1f}B tokens)")
        print(f"Tokens per step: {tokens_per_step:,}")

    # Load tokenizer and create real dataloader
    if rank == 0:
        print("Loading tokenizer...")
    tokenizer = load_tokenizer(config)
    if rank == 0:
        print(f"Tokenizer loaded: {tokenizer.__class__.__name__}, vocab_size={len(tokenizer)}")

    if rank == 0:
        print("Creating dataloader...")
    dataloader = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        max_seq_len=seq_len,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        print("Dataloader ready")

    # Training loop
    grad_clip = training_config.get('gradient_clip_norm', 1.0)
    log_interval = training_config.get('log_interval', 10)
    debug_interval = args.debug_interval if args.debug else 0

    model.train()
    start_time = time.time()
    tokens_processed = 0
    data_iter = iter(dataloader)

    if args.debug:
        if rank == 0:
            print(f"\n🔍 DEBUG MODE ENABLED - printing diagnostics every {debug_interval} steps")
            print("\n--- Initial Gradient Flow Test ---")
            print("Running one forward-backward pass to verify gradient flow...")
            print("Note: With FSDP, gradient stats are aggregated across all ranks")

        # Get a test batch (all ranks need to participate)
        test_batch = next(iter(dataloader))
        test_batch = {k: v.cuda() for k, v in test_batch.items()}

        # Forward-backward on all ranks
        model.train()
        inputs = test_batch['input_ids'][:, :-1]
        targets = test_batch['labels'][:, 1:]
        outputs = model(input_ids=inputs, labels=targets)
        outputs['loss'].backward()

        # Synchronize all ranks
        dist.barrier()

        # Check gradients on ALL ranks and aggregate
        # Each rank checks its own shard and we reduce the stats
        total_params = 0
        params_with_grad = 0
        params_none_grad = 0
        grad_by_module = defaultdict(lambda: {'total': 0, 'with_grad': 0, 'grad_norm': 0.0})

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            total_params += 1
            name_lower = name.lower()

            # Categorize - must match startup check patterns
            if 'mod_router' in name_lower or 'ffn_mod_router' in name_lower:
                mod = 'mod_router'
            elif 'lm_head' in name_lower:
                mod = 'lm_head'
            elif '.ffn.router' in name_lower or ('moe' in name_lower and 'router' in name_lower):
                mod = 'moe_router'
            elif 'grouped_experts' in name_lower or 'local_experts' in name_lower:
                mod = 'expert_weights'
            elif '.ffn.' in name_lower and any(f'.{w}' in name_lower for w in ['w1', 'w2', 'w3']):
                mod = 'expert_weights'
            elif 'embed' in name_lower or 'tok_embed' in name_lower:
                mod = 'embeddings'
            elif 'norm' in name_lower:
                mod = 'norms'
            elif any(p in name_lower for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj']):
                mod = 'attention'
            elif 'seq_layer' in name_lower and 'ffn' not in name_lower:
                mod = 'attention'
            else:
                mod = 'other'

            grad_by_module[mod]['total'] += 1

            if param.grad is not None:
                params_with_grad += 1
                grad_by_module[mod]['with_grad'] += 1
                grad_by_module[mod]['grad_norm'] += param.grad.float().norm().item() ** 2
            else:
                params_none_grad += 1
                # Don't print per-param warnings - we'll aggregate across ranks

        # Aggregate gradient stats across all ranks using all_reduce
        # Create tensors for reduction
        module_names = sorted(grad_by_module.keys())
        local_stats = torch.zeros(len(module_names) * 3, device='cuda')  # total, with_grad, grad_norm per module
        for i, mod in enumerate(module_names):
            local_stats[i * 3 + 0] = grad_by_module[mod]['total']
            local_stats[i * 3 + 1] = grad_by_module[mod]['with_grad']
            local_stats[i * 3 + 2] = grad_by_module[mod]['grad_norm']

        # Sum across all ranks
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        # Also reduce total counters
        counters = torch.tensor([total_params, params_with_grad, params_none_grad], device='cuda', dtype=torch.float32)
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        total_params_all = int(counters[0].item())
        params_with_grad_all = int(counters[1].item())
        params_none_grad_all = int(counters[2].item())

        # Only rank 0 prints the aggregated results
        if rank == 0:
            print(f"\nGradient flow summary (aggregated across {world_size} ranks):")
            print(f"  Total trainable params: {total_params_all}")
            print(f"  Params with gradients: {params_with_grad_all}")
            print(f"  Params without gradients: {params_none_grad_all}")

            print(f"\nBy module:")
            for i, mod in enumerate(module_names):
                total = int(local_stats[i * 3 + 0].item())
                with_grad = int(local_stats[i * 3 + 1].item())
                grad_norm_sq = local_stats[i * 3 + 2].item()
                grad_norm = math.sqrt(grad_norm_sq) if grad_norm_sq > 0 else 0
                status = "✓" if with_grad == total else "⚠️ MISSING"
                print(f"  {mod:20s}: {with_grad:3d}/{total:3d} grads, norm={grad_norm:.6f} {status}")

            if params_none_grad_all > 0:
                print(f"\n⚠️ {params_none_grad_all} parameters did not receive gradients!")
                print("This could cause training to plateau.")
            else:
                print(f"\n✓ All parameters received gradients!")

        # Clear gradients for actual training
        optimizer.zero_grad()
        if rank == 0:
            print()

    for step in range(total_steps):
        # Get next batch from real data
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart dataloader if exhausted
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to GPU
        batch = {k: v.cuda() for k, v in batch.items()}

        # Enable debug at specific intervals
        do_debug = args.debug and (step + 1) % debug_interval == 0

        metrics = train_step(
            model, batch, optimizer, scheduler,
            grad_clip, accum_steps, step,
            debug=do_debug, rank=rank
        )

        tokens_processed += batch_size * seq_len * world_size

        if rank == 0 and (step + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = tokens_processed / elapsed
            lr = scheduler.get_last_lr()[0]

            # Show aux_loss if significant
            aux_loss_str = ""
            if metrics.get('aux_loss', 0) > 0.001:
                aux_loss_str = f" | Aux: {metrics['aux_loss']:.4f}"

            print(f"Step {step+1}/{total_steps} | "
                  f"Loss: {metrics['loss']:.4f}{aux_loss_str} | "
                  f"LR: {lr:.2e} | "
                  f"Tok/s: {tok_per_sec:.0f}")

    if rank == 0:
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Average throughput: {tokens_processed/total_time:.0f} tokens/sec")

    cleanup_distributed()


if __name__ == '__main__':
    main()
