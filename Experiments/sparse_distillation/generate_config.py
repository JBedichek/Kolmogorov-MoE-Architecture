#!/usr/bin/env python3
"""
Generate per-layer MoE config from cluster analysis.

Takes cluster analysis JSON and generates a config where:
- Each layer has its own n_experts (based on best cluster balance)
- d_ff_expert is adjusted per layer to maintain consistent expansion ratio
- top_k is adjusted per layer to maintain consistent active parameter ratio

Usage:
    python Experiments/sparse_distillation/generate_config.py \
        ./sparse_output/cluster_analysis.json \
        --expansion_ratio 3.0 \
        --active_ratio 0.25 \
        --min_experts 4 \
        --output ./sparse_output/layer_config.json
"""

import argparse
import json
import os


def compute_layer_config(
    analysis: dict,
    target_expansion_ratio: float = 3.0,
    target_active_ratio: float = 0.25,
    min_experts: int = 4,
    min_top_k: int = 1,
    max_top_k: int = 8,
) -> dict:
    """
    Compute per-layer n_experts, d_ff_expert, and top_k to maintain consistent ratios.

    The expansion ratio is: (n_experts * d_ff_expert) / d_ff_dense
    The active ratio is: (top_k * d_ff_expert) / d_ff_dense

    Args:
        analysis: Cluster analysis JSON data
        target_expansion_ratio: Target total MoE params / dense FFN params
        target_active_ratio: Target active params / dense FFN params (0-1 typical)
        min_experts: Minimum number of experts per layer
        min_top_k: Minimum top_k value
        max_top_k: Maximum top_k value

    Returns:
        Per-layer config dict
    """
    d_model = analysis['d_model']
    d_ff_dense = analysis['d_ff']
    layers = analysis['layers']

    # Target: n_experts * d_ff_expert = target_expansion_ratio * d_ff_dense
    target_total = target_expansion_ratio * d_ff_dense

    # Target: top_k * d_ff_expert = target_active_ratio * d_ff_dense
    target_active = target_active_ratio * d_ff_dense

    layer_configs = {}

    print(f"\nComputing per-layer config:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff_dense: {d_ff_dense}")
    print(f"  target_expansion_ratio: {target_expansion_ratio}")
    print(f"  target_active_ratio: {target_active_ratio}")
    print(f"  min_experts: {min_experts}")
    print()

    print(f"{'Layer':>6} | {'n_experts':>10} | {'d_ff_expert':>12} | {'top_k':>6} | {'Active d_ff':>12} | {'Balance':>8}")
    print("-" * 75)

    for layer_idx, layer_data in sorted(layers.items(), key=lambda x: int(x[0])):
        n_experts = layer_data['n_experts']
        balance = layer_data['balance']

        # Enforce minimum experts
        if n_experts < min_experts:
            # Find next best k >= min_experts from all_results if available
            if 'all_results' in layer_data:
                # Convert string keys to int (JSON stores keys as strings)
                all_results = {int(k): v for k, v in layer_data['all_results'].items()}
                valid_results = {k: v for k, v in all_results.items()
                               if k >= min_experts}
                if valid_results:
                    n_experts = max(valid_results.keys(), key=lambda k: valid_results[k]['balance'])
                    balance = valid_results[n_experts]['balance']
                else:
                    n_experts = min_experts
            else:
                n_experts = min_experts

        # Compute d_ff_expert to hit target expansion
        d_ff_expert = int(target_total / n_experts)

        # Round to multiple of 64 for efficiency
        d_ff_expert = max(64, (d_ff_expert // 64) * 64)

        # Compute top_k to hit target active ratio
        # top_k * d_ff_expert = target_active
        # top_k = target_active / d_ff_expert
        top_k = target_active / d_ff_expert
        top_k = int(round(top_k))
        top_k = max(min_top_k, min(max_top_k, top_k))

        # Ensure top_k <= n_experts
        top_k = min(top_k, n_experts)

        # Actual active d_ff
        active_d_ff = top_k * d_ff_expert

        layer_configs[layer_idx] = {
            'n_experts': n_experts,
            'd_ff_expert': d_ff_expert,
            'top_k': top_k,
            'balance': balance,
            'silhouette': layer_data.get('silhouette', 0),
        }

        print(f"{layer_idx:>6} | {n_experts:>10} | {d_ff_expert:>12} | {top_k:>6} | {active_d_ff:>12} | {balance:>8.4f}")

    # Summary stats
    n_experts_list = [cfg['n_experts'] for cfg in layer_configs.values()]
    d_ff_list = [cfg['d_ff_expert'] for cfg in layer_configs.values()]
    top_k_list = [cfg['top_k'] for cfg in layer_configs.values()]

    print()
    print(f"Summary:")
    print(f"  n_experts range: {min(n_experts_list)} - {max(n_experts_list)}")
    print(f"  d_ff_expert range: {min(d_ff_list)} - {max(d_ff_list)}")
    print(f"  top_k range: {min(top_k_list)} - {max(top_k_list)}")

    # Compute total params
    # SwiGLU: 3 * d_model * d_ff per expert (gate, up, down projections)
    total_moe_params = sum(
        cfg['n_experts'] * 3 * d_model * cfg['d_ff_expert']
        for cfg in layer_configs.values()
    )
    total_dense_params = len(layer_configs) * 3 * d_model * d_ff_dense

    print(f"  Total MoE FFN params: {total_moe_params / 1e9:.3f}B")
    print(f"  Equivalent dense FFN params: {total_dense_params / 1e9:.3f}B")
    print(f"  Actual expansion ratio: {total_moe_params / total_dense_params:.3f}x")

    # Active params (per-layer top_k)
    total_active_params = sum(
        cfg['top_k'] * 3 * d_model * cfg['d_ff_expert']
        for cfg in layer_configs.values()
    )
    print(f"  Active params: {total_active_params / 1e9:.3f}B")
    print(f"  Actual active ratio: {total_active_params / total_dense_params:.3f}x")

    return {
        'd_model': d_model,
        'd_ff_dense': d_ff_dense,
        'target_expansion_ratio': target_expansion_ratio,
        'target_active_ratio': target_active_ratio,
        'min_experts': min_experts,
        'layers': layer_configs,
        'stats': {
            'total_moe_params': total_moe_params,
            'total_dense_params': total_dense_params,
            'actual_expansion_ratio': total_moe_params / total_dense_params,
            'total_active_params': total_active_params,
            'actual_active_ratio': total_active_params / total_dense_params,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate per-layer MoE config from cluster analysis")
    parser.add_argument('analysis_path', type=str, help='Path to cluster_analysis.json')
    parser.add_argument('--expansion_ratio', type=float, default=3.0,
                        help='Target (n_experts * d_ff_expert) / d_ff_dense (default: 3.0)')
    parser.add_argument('--active_ratio', type=float, default=0.25,
                        help='Target (top_k * d_ff_expert) / d_ff_dense (default: 0.25)')
    parser.add_argument('--min_experts', type=int, default=4,
                        help='Minimum number of experts per layer (default: 4)')
    parser.add_argument('--min_top_k', type=int, default=1,
                        help='Minimum top_k value (default: 1)')
    parser.add_argument('--max_top_k', type=int, default=8,
                        help='Maximum top_k value (default: 8)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for layer config JSON')
    args = parser.parse_args()

    # Load analysis
    with open(args.analysis_path, 'r') as f:
        analysis = json.load(f)

    print(f"Loaded analysis from: {args.analysis_path}")
    print(f"  {len(analysis['layers'])} layers")

    # Compute config
    config = compute_layer_config(
        analysis,
        target_expansion_ratio=args.expansion_ratio,
        target_active_ratio=args.active_ratio,
        min_experts=args.min_experts,
        min_top_k=args.min_top_k,
        max_top_k=args.max_top_k,
    )

    # Save
    if args.output is None:
        args.output = args.analysis_path.replace('cluster_analysis.json', 'layer_config.json')

    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved config to: {args.output}")


if __name__ == "__main__":
    main()
