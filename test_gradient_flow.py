"""
Test gradient flow to RMSNorm parameters.
Diagnose why norms aren't updating during training.
"""

import torch
import torch.nn as nn
from moe_arch.model.config import get_test_config
from moe_arch.model.transformer import MoETransformer

def test_gradient_flow_with_mod():
    """Test if RMSNorm gets gradients with MoD enabled."""
    print("="*70)
    print("GRADIENT FLOW TEST: WITH MoD (75% capacity)")
    print("="*70)

    config = get_test_config()
    config.mod_enabled = True
    config.mod_capacity_factor = 0.75
    config.use_flash_attention = False  # Disable flash attention for CPU testing
    
    print(f"\nConfig:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  MoD enabled: {config.mod_enabled}")
    print(f"  MoD capacity: {config.mod_capacity_factor}")
    
    # Create model
    model = MoETransformer(config)
    model.train()
    
    # Create input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward pass
    print("\n1. Running forward pass...")
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    print("\n2. Running backward pass...")
    loss.backward()
    print("   ✓ Backward complete")
    
    # Check norm gradients
    print("\n3. Checking RMSNorm gradients...")
    norm_grads = []
    for name, param in model.named_parameters():
        if 'norm' in name.lower() and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            grad_std = param.grad.std().item()
            norm_grads.append((name, grad_mean, grad_max, grad_std))
    
    if len(norm_grads) == 0:
        print("   ✗ CRITICAL: NO GRADIENTS FOR ANY NORMS!")
        return False
    
    print(f"   Found gradients for {len(norm_grads)} norm parameters")
    
    # Check if gradients are non-zero
    zero_grads = [name for name, mean, _, _ in norm_grads if mean < 1e-8]
    tiny_grads = [name for name, mean, _, _ in norm_grads if 1e-8 <= mean < 1e-6]
    good_grads = [name for name, mean, _, _ in norm_grads if mean >= 1e-6]
    
    print(f"\n   Gradient statistics:")
    print(f"   - Zero gradients (<1e-8): {len(zero_grads)}")
    print(f"   - Tiny gradients (<1e-6): {len(tiny_grads)}")
    print(f"   - Normal gradients (≥1e-6): {len(good_grads)}")
    
    if zero_grads:
        print(f"\n   ✗ Parameters with ZERO gradients:")
        for name in zero_grads[:5]:
            print(f"      {name}")
    
    if tiny_grads:
        print(f"\n   ⚠ Parameters with TINY gradients:")
        for name in tiny_grads[:5]:
            print(f"      {name}")
    
    if good_grads:
        print(f"\n   ✓ Parameters with normal gradients:")
        for name, mean, max_g, std in norm_grads[:5]:
            if mean >= 1e-6:
                print(f"      {name:40s} mean={mean:.2e}, max={max_g:.2e}, std={std:.2e}")
    
    # Overall assessment
    if len(zero_grads) > len(norm_grads) // 2:
        print(f"\n   ✗ PROBLEM: {len(zero_grads)}/{len(norm_grads)} norms have zero gradients!")
        return False
    elif len(good_grads) == len(norm_grads):
        print(f"\n   ✓ All norms have normal gradients")
        return True
    else:
        print(f"\n   ⚠ Mixed: Some norms have gradients, some don't")
        return False

def test_gradient_flow_without_mod():
    """Test if RMSNorm gets gradients with MoD disabled."""
    print("\n" + "="*70)
    print("GRADIENT FLOW TEST: WITHOUT MoD (100% capacity)")
    print("="*70)

    config = get_test_config()
    config.mod_enabled = False  # DISABLE MoD
    config.use_flash_attention = False  # Disable flash attention for CPU testing
    
    print(f"\nConfig:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  MoD enabled: {config.mod_enabled}")
    
    # Create model
    model = MoETransformer(config)
    model.train()
    
    # Create input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward + backward
    print("\n1. Running forward + backward...")
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Backward complete")
    
    # Check norm gradients
    print("\n2. Checking RMSNorm gradients...")
    norm_grads = []
    for name, param in model.named_parameters():
        if 'norm' in name.lower() and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            norm_grads.append((name, grad_mean, grad_max))
    
    if len(norm_grads) == 0:
        print("   ✗ CRITICAL: NO GRADIENTS FOR ANY NORMS!")
        return False
    
    print(f"   Found gradients for {len(norm_grads)} norm parameters")
    
    # Check gradient magnitudes
    good_grads = [name for name, mean, _ in norm_grads if mean >= 1e-6]
    
    print(f"\n   Normal gradients (≥1e-6): {len(good_grads)}/{len(norm_grads)}")
    
    if good_grads:
        print(f"\n   ✓ Sample gradients:")
        for name, mean, max_g in norm_grads[:5]:
            if mean >= 1e-6:
                print(f"      {name:40s} mean={mean:.2e}, max={max_g:.2e}")
    
    if len(good_grads) == len(norm_grads):
        print(f"\n   ✓ All norms have normal gradients without MoD")
        return True
    else:
        print(f"\n   ✗ Some norms still have zero/tiny gradients")
        return False

def test_mod_router_gradients():
    """Test if MoD router itself gets gradients."""
    print("\n" + "="*70)
    print("GRADIENT FLOW TEST: MoD Router")
    print("="*70)

    config = get_test_config()
    config.mod_enabled = True
    config.mod_capacity_factor = 0.75
    config.use_flash_attention = False  # Disable flash attention for CPU testing

    # Create model
    model = MoETransformer(config)
    model.train()

    # Forward + backward
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()

    # Check MoD router gradients
    print("\nChecking MoD router gradients...")
    router_grads = []
    for name, param in model.named_parameters():
        if 'mod_router' in name.lower() and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            router_grads.append((name, grad_mean))

    if len(router_grads) == 0:
        print("   ✗ MoD routers have NO gradients!")
        return False

    print(f"   Found gradients for {len(router_grads)} MoD router parameters")
    for name, mean in router_grads[:5]:
        print(f"      {name:50s} mean={mean:.2e}")

    # Use 1e-8 threshold - aux_router may have smaller gradients but that's OK
    # The main router.weight should have larger gradients (typically 1e-4 to 1e-3)
    good = [name for name, mean in router_grads if mean >= 1e-8]
    main_router_good = any('router.weight' in name and mean >= 1e-5 for name, mean in router_grads)

    if len(good) == len(router_grads) and main_router_good:
        print(f"\n   ✓ All MoD routers have gradients")
        return True
    else:
        print(f"\n   ⚠ Some MoD routers have zero/tiny gradients ({len(good)}/{len(router_grads)} >= 1e-8)")
        return False


def test_moe_router_gradients():
    """Test if MoE router gets gradients."""
    print("\n" + "="*70)
    print("GRADIENT FLOW TEST: MoE Router")
    print("="*70)

    config = get_test_config()
    config.mod_enabled = False  # Disable MoD to isolate MoE testing
    config.use_flash_attention = False  # Disable flash attention for CPU testing

    # Create model
    model = MoETransformer(config)
    model.train()

    # Forward + backward
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()

    # Check MoE router gradients (but not MoD router)
    print("\nChecking MoE router gradients...")
    router_grads = []
    for name, param in model.named_parameters():
        # MoE routers are at ffn.router, but not ffn_mod_router
        if 'router' in name.lower() and 'mod_router' not in name.lower() and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            router_grads.append((name, grad_mean))

    if len(router_grads) == 0:
        print("   ✗ MoE routers have NO gradients!")
        return False

    print(f"   Found gradients for {len(router_grads)} MoE router parameters")
    for name, mean in router_grads[:5]:
        print(f"      {name:50s} mean={mean:.2e}")

    good = [name for name, mean in router_grads if mean >= 1e-8]
    if len(good) == len(router_grads):
        print(f"\n   ✓ All MoE routers have gradients")
        return True
    else:
        print(f"\n   ⚠ Some MoE routers have zero/tiny gradients")
        return False


def test_lm_head_gradients():
    """Test if LM head gets gradients."""
    print("\n" + "="*70)
    print("GRADIENT FLOW TEST: LM Head")
    print("="*70)

    config = get_test_config()
    config.mod_enabled = False
    config.use_flash_attention = False  # Disable flash attention for CPU testing

    # Create model
    model = MoETransformer(config)
    model.train()

    # Forward + backward
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()

    # Check LM head gradients
    print("\nChecking LM head gradients...")
    lm_grads = []
    for name, param in model.named_parameters():
        if 'lm_head' in name.lower() and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            lm_grads.append((name, grad_mean))

    if len(lm_grads) == 0:
        print("   ✗ LM heads have NO gradients!")
        return False

    print(f"   Found gradients for {len(lm_grads)} LM head parameters")
    for name, mean in lm_grads:
        status = "✓" if mean >= 1e-8 else "⚠️"
        print(f"      {name:50s} mean={mean:.2e} {status}")

    good = [name for name, mean in lm_grads if mean >= 1e-8]
    if len(good) == len(lm_grads):
        print(f"\n   ✓ All LM heads have gradients")
        return True
    else:
        print(f"\n   ⚠ Some LM heads have zero/tiny gradients")
        return False


def test_full_gradient_flow():
    """Comprehensive test of all module gradients."""
    print("\n" + "="*70)
    print("COMPREHENSIVE GRADIENT FLOW TEST")
    print("="*70)

    config = get_test_config()
    config.mod_enabled = True
    config.mod_capacity_factor = 0.5
    config.use_flash_attention = False  # Disable flash attention for CPU testing

    print(f"\nConfig:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_experts: {config.n_experts}")
    print(f"  MoD enabled: {config.mod_enabled}")
    print(f"  MoD capacity: {config.mod_capacity_factor}")

    # Create model
    model = MoETransformer(config)
    model.train()

    # Forward + backward
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    print(f"\nLoss: {loss.item():.4f}")
    loss.backward()

    # Check all gradients by category
    print("\n" + "-"*60)
    print("Gradient status by module type:")
    print("-"*60)

    categories = {
        'mod_router': [],
        'moe_router': [],
        'expert_weights': [],
        'attention': [],
        'lm_head': [],
        'embeddings': [],
        'norms': [],
        'other': [],
    }

    for name, param in model.named_parameters():
        name_lower = name.lower()

        # Categorize
        if 'ffn_mod_router' in name_lower or 'mod_router' in name_lower:
            cat = 'mod_router'
        elif 'router' in name_lower:
            cat = 'moe_router'
        elif 'grouped_experts' in name_lower:
            cat = 'expert_weights'
        elif any(x in name_lower for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            cat = 'attention'
        elif 'lm_head' in name_lower:
            cat = 'lm_head'
        elif 'embed' in name_lower:
            cat = 'embeddings'
        elif 'norm' in name_lower:
            cat = 'norms'
        else:
            cat = 'other'

        # Get gradient info
        has_grad = param.grad is not None
        grad_mean = param.grad.abs().mean().item() if has_grad else 0.0
        categories[cat].append((name, has_grad, grad_mean))

    all_ok = True
    for cat, params in categories.items():
        if not params:
            continue

        total = len(params)
        with_grad = sum(1 for _, has, _ in params if has)
        nonzero = sum(1 for _, has, mean in params if has and mean >= 1e-10)

        if with_grad == total and nonzero == total:
            status = "✓ OK"
        else:
            status = f"⚠️ {with_grad}/{total} have grad, {nonzero} nonzero"
            all_ok = False

        avg_mean = sum(m for _, _, m in params if m > 0) / max(nonzero, 1)
        print(f"  {cat:20s}: {nonzero:3d}/{total:3d} have gradients, avg={avg_mean:.2e} {status}")

    print("-"*60)

    if all_ok:
        print("\n✓ ALL PARAMETERS HAVE NON-ZERO GRADIENTS!")
    else:
        print("\n⚠️ SOME PARAMETERS HAVE MISSING OR ZERO GRADIENTS")

        # Show problematic params
        print("\nProblematic parameters:")
        for cat, params in categories.items():
            for name, has_grad, mean in params:
                if not has_grad:
                    print(f"  - {name}: NO GRADIENT")
                elif mean < 1e-10:
                    print(f"  - {name}: gradient={mean:.2e} (too small)")

    return all_ok

def main():
    print("="*70)
    print("COMPREHENSIVE GRADIENT FLOW DIAGNOSIS")
    print("="*70)
    print("\nThis will test if all parameters receive gradients")
    print("during training, and identify where gradient flow breaks.\n")

    # Test 1: With MoD
    test1_passed = test_gradient_flow_with_mod()

    # Test 2: Without MoD
    test2_passed = test_gradient_flow_without_mod()

    # Test 3: MoD Router
    test3_passed = test_mod_router_gradients()

    # Test 4: MoE Router
    test4_passed = test_moe_router_gradients()

    # Test 5: LM Head
    test5_passed = test_lm_head_gradients()

    # Test 6: Full model
    test6_passed = test_full_gradient_flow()

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)

    print(f"\nTest 1 - Norms with MoD enabled:    {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Test 2 - Norms without MoD:         {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"Test 3 - MoD router gradients:      {'✓ PASS' if test3_passed else '✗ FAIL'}")
    print(f"Test 4 - MoE router gradients:      {'✓ PASS' if test4_passed else '✗ FAIL'}")
    print(f"Test 5 - LM head gradients:         {'✓ PASS' if test5_passed else '✗ FAIL'}")
    print(f"Test 6 - Full gradient flow:        {'✓ PASS' if test6_passed else '✗ FAIL'}")

    all_passed = test1_passed and test2_passed and test3_passed and test4_passed and test5_passed and test6_passed

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if all_passed:
        print("\n✓ ALL GRADIENTS ARE FLOWING CORRECTLY")
        print("  If you still see training issues, check:")
        print("    - FSDP configuration (might affect gradient visibility)")
        print("    - Activation checkpointing interactions")
        print("    - Learning rate and optimizer settings")
    else:
        print("\n⚠️ SOME GRADIENT FLOW ISSUES DETECTED")

        if not test3_passed:
            print("\n  MoD Router issue:")
            print("    - Check the straight-through estimator in layers.py")
            print("    - Verify aux_loss uses differentiable scores")

        if not test4_passed:
            print("\n  MoE Router issue:")
            print("    - Check routing_weights multiplication in GroupedExperts")
            print("    - Verify gradient flows through index_add_")

        if not test5_passed:
            print("\n  LM Head issue:")
            print("    - Check loss computation in MultiTokenPredictionLoss")
            print("    - Verify logits are connected to loss")

    print("\n" + "="*70)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
