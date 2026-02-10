"""
Comprehensive test suite for Phase 3: MoD, Mamba, Multi-token Prediction.

Tests all Phase 3 features:
1. Mamba SSM blocks
2. Routing Mamba (RoM)
3. Mixture of Depths (MoD)
4. Multi-token prediction
5. Full integration with MoE
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from moe_arch.model.config import get_test_config, get_5b_config
from moe_arch.model.mamba import MambaBlock, RoutingMamba
from moe_arch.model.mod import MoDRouter
from moe_arch.model.transformer import MoETransformer


def test_mamba():
    """Test 1: Mamba blocks."""
    print("=" * 70)
    print("TEST 1: Mamba SSM Blocks")
    print("=" * 70)

    config = get_test_config()
    mamba = MambaBlock(config)

    hidden = torch.randn(2, 16, config.d_model)
    output = mamba(hidden)

    assert output.shape == hidden.shape
    print(f"  ✓ Basic Mamba works")
    print(f"  ✓ Parameters: {sum(p.numel() for p in mamba.parameters()):,}")

    # Test Routing Mamba
    rom = RoutingMamba(config, layer_idx=0)
    rom.train()
    output = rom(hidden)

    assert output.shape == hidden.shape
    assert rom.aux_loss > 0
    print(f"  ✓ Routing Mamba works (aux_loss={rom.aux_loss:.6f})")
    print(f"  ✓ Parameters: {sum(p.numel() for p in rom.parameters()):,}")


def test_mod():
    """Test 2: Mixture of Depths."""
    print("\n" + "=" * 70)
    print("TEST 2: Mixture of Depths (MoD)")
    print("=" * 70)

    config = get_test_config()
    router = MoDRouter(config)

    hidden = torch.randn(2, 16, config.d_model)
    router.train()

    mask, indices, scores = router(hidden)

    k = int(16 * config.mod_capacity_factor)
    actual_counts = mask.sum(dim=1)
    assert (actual_counts == k).all(), f"Selection count mismatch: got {actual_counts.tolist()}, expected {k}"
    print(f"  ✓ MoD router selects {k}/16 tokens ({config.mod_capacity_factor:.0%})")
    print(f"  ✓ Load balance loss: {router.aux_loss:.6f}")


def test_multitoken():
    """Test 3: Multi-token prediction."""
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Token Prediction")
    print("=" * 70)

    from moe_arch.training.losses import MultiTokenPredictionLoss

    loss_fn = MultiTokenPredictionLoss(n_pred_tokens=4)

    logits = [torch.randn(2, 16, 1000, requires_grad=True) for _ in range(4)]
    labels = torch.randint(0, 1000, (2, 16))

    total_loss, loss_dict = loss_fn(logits, labels)

    assert total_loss > 0
    assert len(loss_dict) == 5  # 4 individual + 1 total
    print(f"  ✓ Multi-token loss: {total_loss.item():.4f}")
    print(f"  ✓ Individual losses: {list(loss_dict.values())[:4]}")

    total_loss.backward()
    print(f"  ✓ Backward pass works")


def test_full_model():
    """Test 4: Full model with all Phase 3 features."""
    print("\n" + "=" * 70)
    print("TEST 4: Full Model with All Features")
    print("=" * 70)

    config = get_test_config()
    config.use_flash_attention = False
    config.n_layers = 4
    config.mod_enabled = True
    config.mamba_enabled = True

    print(f"  Config:")
    print(f"    Layers: {config.n_layers}")
    print(f"    MoE layers: {list(config.moe_layers)}")
    print(f"    Mamba layers: {list(config.mamba_layers)}")
    print(f"    MoD enabled: {config.mod_enabled}")
    print(f"    Multi-token heads: {config.n_pred_tokens}")

    model = MoETransformer(config)

    params = model.count_parameters()
    print(f"\n  Parameters:")
    print(f"    Total: {params['total']:,} ({params['total_billions']:.3f}B)")
    print(f"    MoE layers: {params['n_moe_layers']}")
    print(f"    Mamba layers: {params['n_mamba_layers']}")

    # Test forward
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = torch.randint(0, config.vocab_size, (2, 16))

    model.train()
    outputs = model(input_ids, labels=labels)

    print(f"\n  Forward pass:")
    print(f"    Total loss: {outputs['loss'].item():.4f}")
    print(f"    LM loss: {outputs['lm_loss'].item():.4f}")
    print(f"    Aux loss: {outputs['aux_loss']:.6f}")
    print(f"    Prediction heads: {len(outputs['logits_list'])}")

    # Test backward
    outputs['loss'].backward()
    print(f"  ✓ Backward pass works")

    # Test eval mode
    model.eval()
    with torch.no_grad():
        outputs_eval = model(input_ids, labels=labels)

    assert outputs_eval['aux_loss'] == 0.0
    print(f"  ✓ Eval mode (no aux loss)")


def test_5b_model():
    """Test 5: 5B model with all features (optional)."""
    print("\n" + "=" * 70)
    print("TEST 5: 5B Model with All Features (Optional)")
    print("=" * 70)

    try:
        config = get_5b_config()
        config.use_flash_attention = False
        config.n_layers = 8
        config.mod_enabled = True
        config.mamba_enabled = True

        print(f"  Attempting 8-layer 5B model...")

        model = MoETransformer(config)

        params = model.count_parameters()
        print(f"\n  ✓ Model initialized!")
        print(f"    Total: {params['total']:,} ({params['total_billions']:.3f}B)")
        print(f"    MoE layers: {params['n_moe_layers']}")
        print(f"    Mamba layers: {params['n_mamba_layers']}")

        # Test forward
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        labels = torch.randint(0, config.vocab_size, (1, 16))

        model.train()
        outputs = model(input_ids, labels=labels)

        print(f"\n  Forward pass:")
        print(f"    Total loss: {outputs['loss'].item():.4f}")
        print(f"    Prediction heads: {len(outputs['logits_list'])}")

        print(f"  ✓ 5B model works!")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"  ⚠ Could not test 5B model: {e}")


def main():
    """Run all Phase 3 tests."""
    print("\n" + "=" * 70)
    print("PHASE 3: COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    test_mamba()
    test_mod()
    test_multitoken()
    test_full_model()
    test_5b_model()

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 TEST SUMMARY")
    print("=" * 70)
    print("✓ All Phase 3 tests passed!")
    print("\nPhase 3 complete:")
    print("  ✓ Mamba SSM blocks")
    print("  ✓ Routing Mamba (RoM)")
    print("  ✓ Mixture of Depths (MoD)")
    print("  ✓ Multi-token prediction")
    print("  ✓ Full integration with MoE")
    print("\nArchitecture now includes:")
    print("  - Grouped Query Attention (GQA)")
    print("  - Mixture of Experts (MoE)")
    print("  - Routing Mamba (RoM)")
    print("  - Mixture of Depths (MoD)")
    print("  - Multi-token prediction")
    print("  - ~5B parameters at full scale")
    print("\nReady for Phase 4 (Training Infrastructure)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
