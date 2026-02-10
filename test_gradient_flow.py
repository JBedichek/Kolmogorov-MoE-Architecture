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
    
    good = [name for name, mean in router_grads if mean >= 1e-6]
    if len(good) == len(router_grads):
        print(f"\n   ✓ All MoD routers have gradients")
        return True
    else:
        print(f"\n   ⚠ Some MoD routers have zero/tiny gradients")
        return False

def main():
    print("="*70)
    print("COMPREHENSIVE GRADIENT FLOW DIAGNOSIS")
    print("="*70)
    print("\nThis will test if RMSNorm parameters receive gradients")
    print("during training, and identify where gradient flow breaks.\n")
    
    # Test 1: With MoD
    test1_passed = test_gradient_flow_with_mod()
    
    # Test 2: Without MoD
    test2_passed = test_gradient_flow_without_mod()
    
    # Test 3: MoD Router
    test3_passed = test_mod_router_gradients()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print(f"\nTest 1 - Norms with MoD enabled:    {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Test 2 - Norms without MoD:         {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"Test 3 - MoD router gradients:      {'✓ PASS' if test3_passed else '✗ FAIL'}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if test1_passed and test2_passed:
        print("\n✓ GRADIENTS ARE FLOWING CORRECTLY")
        print("  The norm freezing is likely due to:")
        print("    - BFloat16 precision issues")
        print("    - Optimizer state not loading correctly")
        print("    - Learning rate scheduler zeroing LR")
    elif not test1_passed and test2_passed:
        print("\n✗ MoD IS BLOCKING GRADIENTS")
        print("  Norms get gradients WITHOUT MoD but not WITH MoD")
        print("  Problem is in the MoD implementation (layers.py)")
        print("\n  RECOMMENDED FIX:")
        print("    1. Disable MoD temporarily: set mod_enabled=False")
        print("    2. Or set mod_capacity_factor=1.0 (no token skipping)")
        print("    3. Fix the MoD scatter/gather logic in layers.py")
    elif not test1_passed and not test2_passed:
        print("\n✗ FUNDAMENTAL GRADIENT FLOW PROBLEM")
        print("  Norms don't get gradients even without MoD")
        print("  Problem is deeper - likely in:")
        print("    - RMSNorm implementation")
        print("    - Pre-norm architecture")
        print("    - Residual connections")
    else:
        print("\n⚠ UNEXPECTED RESULT")
        print("  Gradients work with MoD but not without?")
        print("  This shouldn't happen - check test configuration")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
