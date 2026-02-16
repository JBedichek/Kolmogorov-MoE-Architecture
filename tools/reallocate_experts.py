#!/usr/bin/env python3
"""
Expert Reallocation Script for MoE Models

Analyzes router entropy across layers and reallocates experts from
collapsed layers (low entropy) to healthy layers (high entropy).

This preserves:
- Trained attention weights
- Well-utilized expert weights

And resets:
- All router weights (fresh initialization)
- Underutilized expert weights (copied from healthy experts)

Usage:
    python tools/reallocate_experts.py --checkpoint path/to/checkpoint.pt --config config.yaml --output new_checkpoint.pt
"""

import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import os
import copy

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig


def create_model_config(config_dict: dict) -> AdvancedMoEConfig:
    """Create AdvancedMoEConfig from YAML config dictionary."""
    model_cfg = config_dict['model']

    moe_layers = model_cfg.get('moe_layers')
    if moe_layers is not None:
        moe_layers = tuple(moe_layers)

    mamba_layers = model_cfg.get('mamba_layers', [])
    if mamba_layers is not None:
        mamba_layers = tuple(mamba_layers)

    aux_loss_weights = model_cfg.get('aux_loss_weights', [1.0])
    if aux_loss_weights is not None:
        aux_loss_weights = tuple(aux_loss_weights)

    return AdvancedMoEConfig(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        n_layers=model_cfg['n_layers'],
        n_heads=model_cfg['n_heads'],
        n_kv_heads=model_cfg.get('n_kv_heads', model_cfg['n_heads']),
        head_dim=model_cfg.get('head_dim', model_cfg['d_model'] // model_cfg['n_heads']),
        d_ff=model_cfg.get('d_ff', model_cfg['d_model'] * 4),
        max_seq_len=model_cfg.get('max_seq_len', 2048),
        n_experts=model_cfg.get('n_experts', 1),
        moe_top_k=model_cfg.get('moe_top_k', 2),
        moe_capacity_factor=model_cfg.get('moe_capacity_factor', 1.25),
        d_ff_expert=model_cfg.get('d_ff_expert', model_cfg['d_model'] * 2),
        moe_load_balance_loss_weight=model_cfg.get('moe_load_balance_loss_weight', 0.01),
        moe_router_z_loss_weight=model_cfg.get('moe_router_z_loss_weight', 0.001),
        moe_layers=moe_layers,
        moe_implementation=model_cfg.get('moe_implementation', 'batched'),
        mod_enabled=model_cfg.get('mod_enabled', False),
        mod_capacity_factor=model_cfg.get('mod_capacity_factor', 0.5),
        mod_router_hidden_dim=model_cfg.get('mod_router_hidden_dim', 64),
        mod_load_balance_loss_weight=model_cfg.get('mod_load_balance_loss_weight', 0.01),
        mamba_enabled=model_cfg.get('mamba_enabled', False),
        mamba_layers=mamba_layers,
        n_pred_tokens=model_cfg.get('n_pred_tokens', 1),
        aux_loss_weights=aux_loss_weights,
        use_flash_attention=model_cfg.get('use_flash_attention', True),
        rope_theta=model_cfg.get('rope_theta', 10000.0),
        dropout=model_cfg.get('dropout', 0.0),
        attention_dropout=model_cfg.get('attention_dropout', 0.0),
        residual_dropout=model_cfg.get('residual_dropout', 0.0),
    )


def analyze_router_entropy(model, n_samples: int = 50, seq_len: int = 256, device: str = 'cuda'):
    """
    Analyze router entropy for each MoE layer.

    Returns dict mapping layer_idx -> {entropy, expert_usage, is_collapsed}
    """
    model.eval()
    vocab_size = model.token_embedding.embedding.weight.shape[0]

    expert_counts = defaultdict(lambda: defaultdict(int))
    expert_probs = defaultdict(list)

    # Find MoE layer info
    n_experts = 1
    top_k = 1
    for layer in model.layers:
        if hasattr(layer, 'use_moe') and layer.use_moe and hasattr(layer.ffn, 'n_experts'):
            n_experts = layer.ffn.n_experts
            top_k = layer.ffn.top_k
            break

    with torch.no_grad():
        for i in range(n_samples):
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
            hidden = model.token_embedding(input_ids)

            for layer_idx, layer in enumerate(model.layers):
                is_moe = hasattr(layer, 'use_moe') and layer.use_moe

                if is_moe and hasattr(layer.ffn, 'router') and layer.ffn.router is not None:
                    moe_layer = layer.ffn
                    normed_hidden = layer.ffn_norm(hidden)

                    routing_weights, selected_experts, router_logits = moe_layer.router(normed_hidden)
                    router_probs = F.softmax(router_logits, dim=-1)

                    for expert_idx in selected_experts.flatten().tolist():
                        expert_counts[layer_idx][expert_idx] += 1

                    expert_probs[layer_idx].append(router_probs.mean(dim=(0, 1)).cpu().numpy())

                output = layer(hidden)
                hidden = output if isinstance(output, torch.Tensor) else output[0]

    # Compute entropy and collapse status for each layer
    results = {}
    total_selections = n_samples * seq_len * top_k

    for layer_idx in sorted(expert_counts.keys()):
        counts = expert_counts[layer_idx]
        probs = np.mean(expert_probs[layer_idx], axis=0)

        # Compute normalized entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(n_experts)
        normalized_entropy = entropy / max_entropy

        # Compute usage statistics
        usage_array = np.array([counts.get(i, 0) for i in range(n_experts)])
        usage_fractions = usage_array / total_selections

        # Compute Gini coefficient
        sorted_usage = np.sort(usage_array)
        n = len(sorted_usage)
        cumsum = np.cumsum(sorted_usage)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_usage)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)

        # Determine collapse status
        max_usage = max(usage_fractions)
        is_collapsed = max_usage > 3 / n_experts or gini > 0.5 or normalized_entropy < 0.7

        # Find most and least used experts
        sorted_experts = sorted(range(n_experts), key=lambda x: counts.get(x, 0), reverse=True)

        results[layer_idx] = {
            'entropy': normalized_entropy,
            'gini': gini,
            'is_collapsed': is_collapsed,
            'expert_usage': usage_fractions,
            'most_used': sorted_experts[:3],
            'least_used': sorted_experts[-3:],
            'probs': probs,
        }

    return results, n_experts, top_k


def reallocate_experts(checkpoint_path: str, config_path: str, output_path: str,
                       device: str = 'cuda', entropy_threshold: float = 0.85):
    """
    Reallocate experts based on router entropy analysis.

    Strategy:
    1. Identify healthy layers (high entropy) and collapsed layers (low entropy)
    2. For collapsed layers: copy expert weights from the most-used experts in healthy layers
    3. Reset ALL router weights to fresh initialization
    4. Keep attention weights intact
    """
    print("="*60)
    print("EXPERT REALLOCATION")
    print("="*60)

    # Load config
    print(f"\nLoading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = create_model_config(config)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model and load weights
    model = MoETransformer(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Analyze router entropy
    print("\nAnalyzing router entropy...")
    entropy_results, n_experts, top_k = analyze_router_entropy(model, device=device)

    # Categorize layers
    healthy_layers = []
    collapsed_layers = []

    print(f"\nLayer Analysis (threshold={entropy_threshold}):")
    for layer_idx, info in sorted(entropy_results.items()):
        status = "COLLAPSED" if info['is_collapsed'] else "healthy"
        print(f"  Layer {layer_idx:2d}: entropy={info['entropy']:.3f}, gini={info['gini']:.3f} [{status}]")

        if info['entropy'] >= entropy_threshold and not info['is_collapsed']:
            healthy_layers.append(layer_idx)
        else:
            collapsed_layers.append(layer_idx)

    print(f"\nHealthy layers ({len(healthy_layers)}): {healthy_layers}")
    print(f"Collapsed layers ({len(collapsed_layers)}): {collapsed_layers}")

    if not healthy_layers:
        print("\nWARNING: No healthy layers found! Using layers with highest entropy as donors.")
        # Sort by entropy and use top half as donors
        sorted_layers = sorted(entropy_results.items(), key=lambda x: x[1]['entropy'], reverse=True)
        mid = len(sorted_layers) // 2
        healthy_layers = [l[0] for l in sorted_layers[:mid]]
        collapsed_layers = [l[0] for l in sorted_layers[mid:]]
        print(f"Adjusted healthy layers: {healthy_layers}")
        print(f"Adjusted collapsed layers: {collapsed_layers}")

    # Get state dict for modification
    state_dict = model.state_dict()
    new_state_dict = copy.deepcopy(state_dict)

    # Find the best experts from healthy layers
    print("\nIdentifying best experts from healthy layers...")
    best_experts = []
    for layer_idx in healthy_layers:
        info = entropy_results[layer_idx]
        # Get the most-used experts from this layer
        for expert_idx in info['most_used'][:2]:  # Top 2 most used
            best_experts.append((layer_idx, expert_idx, info['expert_usage'][expert_idx]))

    # Sort by usage
    best_experts.sort(key=lambda x: x[2], reverse=True)
    print(f"Best experts (layer, expert, usage): {best_experts[:6]}")

    # Reallocate experts in collapsed layers
    print("\nReallocating experts in collapsed layers...")

    for collapsed_layer_idx in collapsed_layers:
        collapsed_info = entropy_results[collapsed_layer_idx]

        # Get the least-used experts in this collapsed layer
        underused_experts = collapsed_info['least_used']

        print(f"\n  Layer {collapsed_layer_idx}: replacing experts {underused_experts}")

        for i, target_expert_idx in enumerate(underused_experts):
            if i >= len(best_experts):
                break

            source_layer_idx, source_expert_idx, _ = best_experts[i % len(best_experts)]

            # Copy expert weights from source to target
            # Expert weights are in: layers.{layer}.ffn.grouped_experts.{gate_proj, up_proj, down_proj}
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                source_key = f'layers.{source_layer_idx}.ffn.grouped_experts.{proj}'
                target_key = f'layers.{collapsed_layer_idx}.ffn.grouped_experts.{proj}'

                if source_key in state_dict and target_key in new_state_dict:
                    # Expert weights are [n_experts, ...], so index by expert
                    source_weights = state_dict[source_key][source_expert_idx].clone()
                    new_state_dict[target_key][target_expert_idx] = source_weights

            print(f"    Expert {target_expert_idx} <- Layer {source_layer_idx} Expert {source_expert_idx}")

    # Reset ALL router weights
    print("\nResetting all router weights...")

    # Create a fresh model to get initialized router weights
    fresh_model = MoETransformer(model_config).to(device)
    fresh_state_dict = fresh_model.state_dict()

    router_keys_reset = 0
    for key in new_state_dict.keys():
        if 'router' in key:
            if key in fresh_state_dict:
                new_state_dict[key] = fresh_state_dict[key].clone()
                router_keys_reset += 1

    print(f"  Reset {router_keys_reset} router weight tensors")

    # Verify attention weights are preserved
    attention_keys = [k for k in new_state_dict.keys() if 'seq_layer' in k or 'q_proj' in k or 'k_proj' in k or 'v_proj' in k or 'o_proj' in k]
    print(f"  Preserved {len(attention_keys)} attention weight tensors")

    # Create new checkpoint
    new_checkpoint = {
        'step': 0,  # Reset step since we're effectively starting fresh
        'epoch': 0,
        'data_idx': 0,
        'losses': [],
        'total_tokens': 0,
        'config_path': config_path,
        'model_state_dict': new_state_dict,
        # Don't include optimizer state - it's stale now
        'reallocated_from': checkpoint_path,
        'reallocated_collapsed_layers': collapsed_layers,
        'reallocated_healthy_layers': healthy_layers,
    }

    # Save
    print(f"\nSaving reallocated checkpoint to {output_path}")
    torch.save(new_checkpoint, output_path)

    # Verify
    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"  Saved {file_size:.2f} GB")

    print("\n" + "="*60)
    print("REALLOCATION COMPLETE")
    print("="*60)
    print(f"\nTo resume training with reallocated weights:")
    print(f"  python train.py --config {config_path} --resume {output_path}")

    return new_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Reallocate MoE experts based on router entropy")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output', type=str, required=True, help='Path to save reallocated checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--entropy-threshold', type=float, default=0.85,
                        help='Entropy threshold for healthy layers (default: 0.85)')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    reallocate_experts(
        args.checkpoint,
        args.config,
        args.output,
        device=device,
        entropy_threshold=args.entropy_threshold,
    )


if __name__ == "__main__":
    main()
