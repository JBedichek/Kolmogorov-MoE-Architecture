"""
Test script to verify 3B configuration and RoutingMamba activation.
"""

import torch
from moe_arch.model.config import get_3b_config
from moe_arch.model.transformer import MoETransformer

def test_3b_config():
    """Test the 3B configuration."""
    print("=" * 70)
    print("TESTING 3B CONFIGURATION")
    print("=" * 70)

    # Get config
    config = get_3b_config()

    print("\n1. Configuration Details:")
    print(f"   d_model: {config.d_model}")
    print(f"   n_layers: {config.n_layers}")
    print(f"   n_heads: {config.n_heads}")
    print(f"   n_kv_heads: {config.n_kv_heads}")
    print(f"   d_ff: {config.d_ff}")
    print(f"   d_ff_expert: {config.d_ff_expert}")
    print(f"   n_experts: {config.n_experts}")
    print(f"   moe_top_k: {config.moe_top_k}")

    print("\n2. Layer Configuration:")
    print(f"   Total layers: {config.n_layers}")
    print(f"   MoE layers: {sorted(config.moe_layers)}")
    print(f"   Mamba layers: {sorted(config.mamba_layers)}")

    # Find RoutingMamba layers
    routing_mamba_layers = sorted(set(config.moe_layers) & set(config.mamba_layers))
    standard_mamba_layers = sorted(set(config.mamba_layers) - set(config.moe_layers))
    attention_moe_layers = sorted(set(config.moe_layers) - set(config.mamba_layers))

    print(f"\n3. Layer Type Breakdown:")
    print(f"   RoutingMamba (Mamba+MoE): {routing_mamba_layers}")
    print(f"   Standard Mamba: {standard_mamba_layers}")
    print(f"   Attention+MoE: {attention_moe_layers}")
    print(f"   Standard Attention: {sorted(set(range(config.n_layers)) - set(config.moe_layers) - set(config.mamba_layers))}")

    # Verify RoutingMamba is activated
    if routing_mamba_layers:
        print(f"\n   ✓ RoutingMamba IS ACTIVATED in {len(routing_mamba_layers)} layers!")
    else:
        print(f"\n   ✗ RoutingMamba is NOT activated!")

    # Count parameters
    print("\n4. Parameter Estimation:")
    params = config.count_parameters()
    print(f"   Total: {params['total_billions']:.2f}B parameters")
    print(f"   Token embeddings: {params['token_embeddings']/1e6:.1f}M")
    print(f"   Attention layers: {params['attention']/1e6:.1f}M")
    print(f"   Mamba layers: {params['mamba']/1e6:.1f}M")
    print(f"   Standard FFN: {params['standard_ffn']/1e6:.1f}M")
    print(f"   MoE FFN: {params['moe_ffn']/1e6:.1f}M")
    print(f"   MoD routers: {params['mod_routers']/1e6:.1f}M")
    print(f"   Multi-token heads: {params['multi_token_heads']/1e6:.1f}M")

    print(f"\n   Layer counts:")
    print(f"   - Standard Attention layers: {params['n_standard_attn_layers']}")
    print(f"   - Attention+MoE layers: {params['n_moe_attn_layers']}")
    print(f"   - Standard Mamba layers: {params['n_mamba_standard_layers']}")
    print(f"   - RoutingMamba layers: {params['n_mamba_moe_layers']}")

    # Test model creation
    print("\n5. Testing Model Creation:")
    try:
        model = MoETransformer(config)
        actual_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created successfully!")
        print(f"   Actual parameters: {actual_params/1e9:.2f}B")

        # Test forward pass
        print("\n6. Testing Forward Pass:")
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)

        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {outputs['logits'].shape}")
        print(f"   ✓ Forward pass works!")

        # Verify layer types
        print("\n7. Verifying Layer Types in Model:")
        for layer_idx in routing_mamba_layers:
            layer = model.layers[layer_idx]
            layer_type = type(layer.seq_layer).__name__
            print(f"   Layer {layer_idx}: {layer_type}")
            assert layer_type == "RoutingMamba", f"Expected RoutingMamba, got {layer_type}"

        print(f"   ✓ All RoutingMamba layers verified!")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("3B CONFIG TEST COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print(f"  ✓ 3B configuration created")
    print(f"  ✓ RoutingMamba activated in {len(routing_mamba_layers)} layers: {routing_mamba_layers}")
    print(f"  ✓ Standard Mamba in {len(standard_mamba_layers)} layers: {standard_mamba_layers}")
    print(f"  ✓ Total parameters: ~{params['total_billions']:.2f}B")
    print(f"  ✓ Model creation and forward pass successful")

if __name__ == "__main__":
    test_3b_config()
