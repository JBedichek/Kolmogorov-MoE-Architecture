#!/usr/bin/env python3
"""
Parameter calculator for MoE transformer models.

Calculates total and active parameter counts based on model configuration.
Useful for designing models with specific parameter budgets.

Usage:
    python tools/param_calculator.py --config configs/production_training.yaml
    python tools/param_calculator.py --d-model 1024 --n-layers 12 --n-experts 16 --top-k 2
"""

import argparse
import yaml
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelParams:
    """Model configuration for parameter calculation."""
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 18
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 128
    d_ff: int = 4096  # Dense FFN hidden dim (for non-MoE layers)
    d_ff_expert: int = 2048  # Expert FFN hidden dim
    n_experts: int = 16
    moe_top_k: int = 2
    moe_layers: Optional[List[int]] = None  # Which layers are MoE (None = all)
    tie_embeddings: bool = False  # Whether to tie input/output embeddings


def calculate_params(cfg: ModelParams) -> dict:
    """
    Calculate total and active parameters for a MoE transformer.

    Returns dict with detailed breakdown.
    """
    # Determine which layers are MoE vs dense
    if cfg.moe_layers is None:
        moe_layer_indices = set(range(cfg.n_layers))
    else:
        moe_layer_indices = set(cfg.moe_layers)

    n_moe_layers = len(moe_layer_indices)
    n_dense_layers = cfg.n_layers - n_moe_layers

    # === EMBEDDING PARAMETERS ===
    embedding_params = cfg.vocab_size * cfg.d_model
    lm_head_params = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model

    # === ATTENTION PARAMETERS (per layer) ===
    # Q projection: d_model -> n_heads * head_dim
    q_proj = cfg.d_model * (cfg.n_heads * cfg.head_dim)
    # K projection: d_model -> n_kv_heads * head_dim
    k_proj = cfg.d_model * (cfg.n_kv_heads * cfg.head_dim)
    # V projection: d_model -> n_kv_heads * head_dim
    v_proj = cfg.d_model * (cfg.n_kv_heads * cfg.head_dim)
    # O projection: n_heads * head_dim -> d_model
    o_proj = (cfg.n_heads * cfg.head_dim) * cfg.d_model

    attention_params_per_layer = q_proj + k_proj + v_proj + o_proj
    total_attention_params = attention_params_per_layer * cfg.n_layers

    # === LAYER NORM PARAMETERS (per layer) ===
    # Pre-attention norm + pre-FFN norm (RMSNorm has only scale, no bias)
    norm_params_per_layer = 2 * cfg.d_model
    total_norm_params = norm_params_per_layer * cfg.n_layers
    # Final norm before LM head
    final_norm_params = cfg.d_model

    # === DENSE FFN PARAMETERS (for non-MoE layers) ===
    # SwiGLU: gate_proj + up_proj + down_proj
    # gate_proj: d_model -> d_ff
    # up_proj: d_model -> d_ff
    # down_proj: d_ff -> d_model
    dense_ffn_per_layer = 3 * cfg.d_model * cfg.d_ff
    total_dense_ffn_params = dense_ffn_per_layer * n_dense_layers

    # === MOE PARAMETERS (for MoE layers) ===
    # Router: d_model -> n_experts (per MoE layer)
    router_params_per_layer = cfg.d_model * cfg.n_experts
    total_router_params = router_params_per_layer * n_moe_layers

    # Expert FFN (SwiGLU per expert)
    expert_ffn_params = 3 * cfg.d_model * cfg.d_ff_expert
    experts_per_layer = cfg.n_experts * expert_ffn_params
    total_expert_params = experts_per_layer * n_moe_layers

    # === TOTALS ===
    total_params = (
        embedding_params +
        lm_head_params +
        total_attention_params +
        total_norm_params +
        final_norm_params +
        total_dense_ffn_params +
        total_router_params +
        total_expert_params
    )

    # === ACTIVE PARAMETERS ===
    # Dense params (always active)
    dense_params = (
        embedding_params +
        lm_head_params +
        total_attention_params +
        total_norm_params +
        final_norm_params +
        total_dense_ffn_params +
        total_router_params  # Router is always computed
    )

    # Active expert params (only top_k out of n_experts)
    active_expert_ratio = cfg.moe_top_k / cfg.n_experts
    active_expert_params = total_expert_params * active_expert_ratio

    active_params = dense_params + active_expert_params

    # Sparsity
    sparsity = 1.0 - (active_params / total_params) if total_params > 0 else 0.0

    return {
        # Summary
        "total_params": total_params,
        "active_params": active_params,
        "sparsity": sparsity,

        # Breakdown
        "embedding_params": embedding_params,
        "lm_head_params": lm_head_params,
        "attention_params": total_attention_params,
        "norm_params": total_norm_params + final_norm_params,
        "dense_ffn_params": total_dense_ffn_params,
        "router_params": total_router_params,
        "expert_params": total_expert_params,

        # Per-layer info
        "attention_per_layer": attention_params_per_layer,
        "expert_ffn_per_expert": expert_ffn_params,
        "experts_per_layer": experts_per_layer,

        # Config summary
        "n_moe_layers": n_moe_layers,
        "n_dense_layers": n_dense_layers,
        "moe_top_k": cfg.moe_top_k,
        "n_experts": cfg.n_experts,
    }


def format_params(n: int) -> str:
    """Format parameter count in human-readable form."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)


def print_report(params: dict, cfg: ModelParams):
    """Print a detailed parameter report."""
    print("=" * 60)
    print("MODEL PARAMETER REPORT")
    print("=" * 60)

    print(f"\n{'Configuration':}")
    print(f"  vocab_size:    {cfg.vocab_size:,}")
    print(f"  d_model:       {cfg.d_model}")
    print(f"  n_layers:      {cfg.n_layers} ({params['n_moe_layers']} MoE, {params['n_dense_layers']} dense)")
    print(f"  n_heads:       {cfg.n_heads} (n_kv_heads={cfg.n_kv_heads})")
    print(f"  head_dim:      {cfg.head_dim}")
    print(f"  d_ff:          {cfg.d_ff} (dense layers)")
    print(f"  d_ff_expert:   {cfg.d_ff_expert}")
    print(f"  n_experts:     {cfg.n_experts}")
    print(f"  moe_top_k:     {cfg.moe_top_k}")

    print(f"\n{'Parameter Breakdown':}")
    print(f"  Embedding:     {format_params(params['embedding_params']):>10}")
    print(f"  LM Head:       {format_params(params['lm_head_params']):>10}")
    print(f"  Attention:     {format_params(params['attention_params']):>10}")
    print(f"  Norms:         {format_params(params['norm_params']):>10}")
    print(f"  Dense FFN:     {format_params(params['dense_ffn_params']):>10}")
    print(f"  Router:        {format_params(params['router_params']):>10}")
    print(f"  Experts:       {format_params(params['expert_params']):>10}")

    print(f"\n{'Summary':}")
    print(f"  Total params:  {format_params(params['total_params']):>10} ({params['total_params']:,})")
    print(f"  Active params: {format_params(params['active_params']):>10} ({params['active_params']:,})")
    print(f"  Sparsity:      {params['sparsity']*100:>9.1f}%")

    print(f"\n{'Per-Layer Details':}")
    print(f"  Attention/layer:    {format_params(params['attention_per_layer'])}")
    print(f"  Expert FFN/expert:  {format_params(params['expert_ffn_per_expert'])}")
    print(f"  All experts/layer:  {format_params(params['experts_per_layer'])}")

    # Memory estimates (bf16)
    print(f"\n{'Memory Estimates (bf16)':}")
    model_memory = params['total_params'] * 2 / 1e9
    active_memory = params['active_params'] * 2 / 1e9
    grad_memory = params['total_params'] * 2 / 1e9  # Gradients same dtype
    optimizer_memory = params['total_params'] * 4 / 1e9  # Adam/Muon momentum in fp32
    print(f"  Model weights:      {model_memory:>6.2f} GB")
    print(f"  Active forward:     {active_memory:>6.2f} GB")
    print(f"  Gradients:          {grad_memory:>6.2f} GB")
    print(f"  Optimizer state:    {optimizer_memory:>6.2f} GB")
    print(f"  Total training:     {model_memory + grad_memory + optimizer_memory:>6.2f} GB (+ activations)")

    print("=" * 60)


def load_config(path: str) -> ModelParams:
    """Load model config from YAML file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    model = cfg.get('model', {})
    return ModelParams(
        vocab_size=model.get('vocab_size', 32000),
        d_model=model.get('d_model', 1024),
        n_layers=model.get('n_layers', 18),
        n_heads=model.get('n_heads', 8),
        n_kv_heads=model.get('n_kv_heads', 4),
        head_dim=model.get('head_dim', 128),
        d_ff=model.get('d_ff', 4096),
        d_ff_expert=model.get('d_ff_expert', 2048),
        n_experts=model.get('n_experts', 16),
        moe_top_k=model.get('moe_top_k', 2),
        moe_layers=model.get('moe_layers'),
        tie_embeddings=model.get('tie_embeddings', False),
    )


def design_model(target_total: float, target_active: float, vocab_size: int = 32000) -> ModelParams:
    """
    Design a model configuration to hit target parameter counts.

    Args:
        target_total: Target total params in billions
        target_active: Target active params in billions
        vocab_size: Vocabulary size

    Returns:
        ModelParams configuration
    """
    target_total_params = target_total * 1e9
    target_active_params = target_active * 1e9

    # Calculate target sparsity
    target_sparsity = 1.0 - (target_active_params / target_total_params)

    print(f"\nDesigning model:")
    print(f"  Target total:  {target_total}B")
    print(f"  Target active: {target_active}B")
    print(f"  Target sparsity: {target_sparsity*100:.1f}%")

    # Start with reasonable defaults and iterate
    best_cfg = None
    best_diff = float('inf')

    # Search over configurations
    for n_layers in [8, 10, 12, 14, 16, 18]:
        for d_model in [512, 768, 1024, 1280, 1536]:
            for n_experts in [8, 16, 32, 64]:
                for top_k in [1, 2]:
                    for d_ff_expert in [1024, 1536, 2048, 2560, 3072]:
                        # Quick filter: sparsity from MoE
                        moe_sparsity = 1.0 - (top_k / n_experts)
                        if moe_sparsity < target_sparsity * 0.5:
                            continue

                        cfg = ModelParams(
                            vocab_size=vocab_size,
                            d_model=d_model,
                            n_layers=n_layers,
                            n_heads=d_model // 128,  # head_dim = 128
                            n_kv_heads=max(1, d_model // 256),  # GQA 2:1
                            head_dim=128,
                            d_ff=int(d_model * 4),  # Not used if all MoE
                            d_ff_expert=d_ff_expert,
                            n_experts=n_experts,
                            moe_top_k=top_k,
                            moe_layers=list(range(n_layers)),  # All MoE
                        )

                        params = calculate_params(cfg)

                        # Score: weighted distance from targets
                        total_diff = abs(params['total_params'] - target_total_params) / target_total_params
                        active_diff = abs(params['active_params'] - target_active_params) / target_active_params
                        diff = total_diff + active_diff * 2  # Weight active more

                        if diff < best_diff:
                            best_diff = diff
                            best_cfg = cfg
                            best_params = params

    if best_cfg:
        print(f"\nBest configuration found:")
        print_report(best_params, best_cfg)
        return best_cfg
    else:
        print("No suitable configuration found")
        return None


def main():
    parser = argparse.ArgumentParser(description="Calculate MoE transformer parameters")
    parser.add_argument('--config', type=str, help='Path to YAML config file')

    # Manual config options
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--d-model', type=int, default=1024)
    parser.add_argument('--n-layers', type=int, default=18)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-kv-heads', type=int, default=4)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--d-ff', type=int, default=4096)
    parser.add_argument('--d-ff-expert', type=int, default=2048)
    parser.add_argument('--n-experts', type=int, default=16)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--all-moe', action='store_true', help='All layers are MoE')

    # Design mode
    parser.add_argument('--design', action='store_true', help='Design model for targets')
    parser.add_argument('--target-total', type=float, default=1.0, help='Target total params (B)')
    parser.add_argument('--target-active', type=float, default=0.15, help='Target active params (B)')

    args = parser.parse_args()

    if args.design:
        design_model(args.target_total, args.target_active, args.vocab_size)
        return

    if args.config:
        cfg = load_config(args.config)
    else:
        moe_layers = list(range(args.n_layers)) if args.all_moe else None
        cfg = ModelParams(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            d_ff=args.d_ff,
            d_ff_expert=args.d_ff_expert,
            n_experts=args.n_experts,
            moe_top_k=args.top_k,
            moe_layers=moe_layers,
        )

    params = calculate_params(cfg)
    print_report(params, cfg)


if __name__ == "__main__":
    main()
