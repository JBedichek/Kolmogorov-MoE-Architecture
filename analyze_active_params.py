"""
Analyze active parameter distribution and propose changes to reduce arithmetic intensity.
"""

from moe_arch.model.config import get_3b_config, AdvancedMoEConfig
from dataclasses import replace

def analyze_config(config: AdvancedMoEConfig, name: str):
    """Analyze parameter distribution for a config."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {name}")
    print(f"{'='*70}")

    params = config.count_parameters()

    print(f"\n1. Total Parameters: {params['total_billions']:.2f}B")
    print(f"   - Token embeddings: {params['token_embeddings']/1e6:.1f}M")
    print(f"   - Attention layers: {params['attention']/1e6:.1f}M")
    print(f"   - Mamba layers: {params['mamba']/1e6:.1f}M")
    print(f"   - Standard FFN: {params['standard_ffn']/1e6:.1f}M")
    print(f"   - MoE FFN: {params['moe_ffn']/1e6:.1f}M")
    print(f"   - MoD routers: {params['mod_routers']/1e6:.1f}M")
    print(f"   - Multi-token heads: {params['multi_token_heads']/1e6:.1f}M")

    print(f"\n2. Layer Distribution:")
    print(f"   - Standard Attention layers: {params['n_standard_attn_layers']}")
    print(f"   - Attention+MoE layers: {params['n_moe_attn_layers']}")
    print(f"   - Standard Mamba layers: {params['n_mamba_standard_layers']}")
    print(f"   - RoutingMamba layers: {params['n_mamba_moe_layers']}")

    # Calculate ACTIVE parameters (what's actually used per forward pass)
    # Embeddings: always active
    active_embed = params['token_embeddings']

    # Attention: always active (all attention layers use all params)
    active_attn = params['attention']

    # Mamba: active based on routing
    # Standard mamba: fully active
    # RoutingMamba: top-k of n_experts active
    active_mamba = params['mamba']  # Rough estimate, Mamba is complex
    if params['n_mamba_moe_layers'] > 0:
        # RoutingMamba uses top-k of experts
        mamba_sparsity = config.moe_top_k / config.n_experts
        # Approximate: some fraction is active
        active_mamba = active_mamba * 0.5  # Rough estimate

    # FFN: depends on MoE vs standard
    # Standard FFN: fully active
    active_standard_ffn = params['standard_ffn']

    # MoE FFN: top-k of n_experts active
    moe_sparsity = config.moe_top_k / config.n_experts
    active_moe_ffn = params['moe_ffn'] * moe_sparsity

    # MoD routers: always active (tiny)
    active_mod = params['mod_routers']

    # Multi-token heads: always active
    active_heads = params['multi_token_heads']

    # LM head: always active
    active_lm_head = params['lm_head']

    # MoD effect: reduces computation by skipping tokens
    mod_multiplier = config.mod_capacity_factor if config.mod_enabled else 1.0

    # Total active (before MoD)
    active_before_mod = (
        active_embed + active_attn + active_mamba +
        active_standard_ffn + active_moe_ffn +
        active_mod + active_heads + active_lm_head
    )

    # After MoD (attention and FFN are affected)
    active_after_mod = (
        active_embed +
        (active_attn + active_mamba + active_standard_ffn + active_moe_ffn) * mod_multiplier +
        active_mod + active_heads + active_lm_head
    )

    print(f"\n3. Active Parameters (per forward pass):")
    print(f"   Before MoD: {active_before_mod/1e9:.2f}B")
    print(f"   After MoD ({config.mod_capacity_factor:.0%} tokens): {active_after_mod/1e9:.2f}B")
    print(f"   MoE sparsity: {config.moe_top_k}/{config.n_experts} = {moe_sparsity:.1%}")
    print(f"   MoD reduction: {(1-config.mod_capacity_factor)*100:.0f}% tokens skip")

    sparsity_ratio = 1 - (active_after_mod / params['total'])
    print(f"\n4. Overall Sparsity: {sparsity_ratio*100:.1f}%")
    print(f"   (using {active_after_mod/params['total']*100:.1f}% of total params per forward pass)")

    return {
        'total': params['total'],
        'active': active_after_mod,
        'sparsity': sparsity_ratio,
    }

def main():
    print("="*70)
    print("ACTIVE PARAMETER ANALYSIS")
    print("="*70)

    # Current 3B config
    current_config = get_3b_config()
    current_stats = analyze_config(current_config, "Current 3B Config")

    print(f"\n\n{'='*70}")
    print("STRATEGIES TO REDUCE ACTIVE PARAMETERS")
    print(f"{'='*70}")
    print(f"\nGoal: Reduce from {current_stats['active']/1e9:.2f}B to ~0.65B active")
    print(f"Required reduction: {(1 - 0.65/(current_stats['active']/1e9))*100:.0f}%")

    print("\n" + "="*70)
    print("OPTION 1: More Aggressive MoD (50% token selection)")
    print("="*70)
    opt1 = replace(current_config, mod_capacity_factor=0.5)
    opt1_stats = analyze_config(opt1, "Option 1: MoD 50%")
    reduction1 = (current_stats['active'] - opt1_stats['active']) / current_stats['active'] * 100
    print(f"\nReduction: {reduction1:.1f}%")

    print("\n" + "="*70)
    print("OPTION 2: Top-1 Expert Selection (instead of Top-2)")
    print("="*70)
    opt2 = replace(current_config, moe_top_k=1)
    opt2_stats = analyze_config(opt2, "Option 2: Top-1")
    reduction2 = (current_stats['active'] - opt2_stats['active']) / current_stats['active'] * 100
    print(f"\nReduction: {reduction2:.1f}%")

    print("\n" + "="*70)
    print("OPTION 3: More Experts (32 instead of 16, keep Top-2)")
    print("="*70)
    opt3 = replace(current_config, n_experts=32)
    opt3_stats = analyze_config(opt3, "Option 3: 32 Experts")
    reduction3 = (current_stats['active'] - opt3_stats['active']) / current_stats['active'] * 100
    print(f"\nReduction: {reduction3:.1f}%")
    print("Note: This increases total params significantly")

    print("\n" + "="*70)
    print("OPTION 4: COMBINED - Top-1 + MoD 50%")
    print("="*70)
    opt4 = replace(current_config, moe_top_k=1, mod_capacity_factor=0.5)
    opt4_stats = analyze_config(opt4, "Option 4: Top-1 + MoD 50%")
    reduction4 = (current_stats['active'] - opt4_stats['active']) / current_stats['active'] * 100
    print(f"\nReduction: {reduction4:.1f}%")

    print("\n" + "="*70)
    print("OPTION 5: MoE in Attention + Top-1 + MoD 50%")
    print("="*70)
    print("This would:")
    print("  - Add MoE to attention layers (sparse attention experts)")
    print("  - Top-1 expert everywhere")
    print("  - 50% token selection in MoD")
    print("  - Maximum sparsity while keeping total params similar")
    print("\nEstimated active: ~0.6-0.7B (need to implement MoE attention)")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nFor ~50% reduction without quality loss:")
    print("  1. Increase experts to 32 (or 24)")
    print("  2. Keep top-2 (maintains quality)")
    print("  3. MoD at 60% (moderate reduction)")
    print("\nFor maximum sparsity:")
    print("  1. Implement MoE in Attention")
    print("  2. Top-1 experts everywhere")
    print("  3. MoD at 50%")
    print("  4. This gets you to ~0.6B active")

    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Configuration':<30s} {'Total':<12s} {'Active':<12s} {'Reduction':<12s}")
    print("-"*70)
    print(f"{'Current':<30s} {current_stats['total']/1e9:>10.2f}B {current_stats['active']/1e9:>10.2f}B {'--':>10s}")
    print(f"{'MoD 50%':<30s} {opt1_stats['total']/1e9:>10.2f}B {opt1_stats['active']/1e9:>10.2f}B {reduction1:>9.1f}%")
    print(f"{'Top-1':<30s} {opt2_stats['total']/1e9:>10.2f}B {opt2_stats['active']/1e9:>10.2f}B {reduction2:>9.1f}%")
    print(f"{'32 Experts':<30s} {opt3_stats['total']/1e9:>10.2f}B {opt3_stats['active']/1e9:>10.2f}B {reduction3:>9.1f}%")
    print(f"{'Top-1 + MoD 50%':<30s} {opt4_stats['total']/1e9:>10.2f}B {opt4_stats['active']/1e9:>10.2f}B {reduction4:>9.1f}%")

if __name__ == "__main__":
    main()
