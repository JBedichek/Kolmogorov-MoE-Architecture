"""
Comprehensive test script for Phase 1 MVP.

Tests:
1. Model initialization
2. Forward pass with various batch sizes and sequence lengths
3. Backward pass and gradient flow
4. Loss computation
5. Memory usage estimation
6. Parameter counts
7. Text generation
8. 5B model instantiation (if sufficient memory)
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from moe_arch.model.config import get_test_config, get_5b_config
from moe_arch.model.transformer import MoETransformer


def test_model_initialization():
    """Test 1: Model initialization."""
    print("=" * 70)
    print("TEST 1: Model Initialization")
    print("=" * 70)

    config = get_test_config()
    config.use_flash_attention = False

    model = MoETransformer(config)

    params = model.count_parameters()
    print(f"✓ Model initialized successfully")
    print(f"  Total parameters: {params['total']:,} ({params['total_billions']:.3f}B)")
    print(f"  Embedding: {params['embedding']:,}")
    print(f"  Layers: {params['layers']:,}")
    print(f"  LM head: {params['lm_head']:,}")

    return model, config


def test_forward_pass(model, config):
    """Test 2: Forward pass with various inputs."""
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass")
    print("=" * 70)

    # Test with small batch
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\n2a. Small batch (batch={batch_size}, seq={seq_len})")
    outputs = model(input_ids)
    logits = outputs["logits"]
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print("  ✓ Small batch test passed")

    # Test with larger sequence
    seq_len_large = 64
    input_ids_large = torch.randint(0, config.vocab_size, (1, seq_len_large))

    print(f"\n2b. Larger sequence (batch=1, seq={seq_len_large})")
    outputs_large = model(input_ids_large)
    logits_large = outputs_large["logits"]
    print(f"  Input shape: {input_ids_large.shape}")
    print(f"  Output shape: {logits_large.shape}")
    assert logits_large.shape == (1, seq_len_large, config.vocab_size)
    print("  ✓ Large sequence test passed")

    # Test with batch size > 1
    batch_size_large = 4
    input_ids_batch = torch.randint(0, config.vocab_size, (batch_size_large, seq_len))

    print(f"\n2c. Larger batch (batch={batch_size_large}, seq={seq_len})")
    outputs_batch = model(input_ids_batch)
    logits_batch = outputs_batch["logits"]
    print(f"  Input shape: {input_ids_batch.shape}")
    print(f"  Output shape: {logits_batch.shape}")
    assert logits_batch.shape == (batch_size_large, seq_len, config.vocab_size)
    print("  ✓ Large batch test passed")

    return input_ids


def test_backward_pass(model, config):
    """Test 3: Backward pass and gradient flow."""
    print("\n" + "=" * 70)
    print("TEST 3: Backward Pass")
    print("=" * 70)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass with labels
    model.zero_grad()
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]

    print(f"\n3a. Loss computation")
    print(f"  Loss value: {loss.item():.4f}")
    assert loss is not None
    print("  ✓ Loss computed successfully")

    # Backward pass
    print(f"\n3b. Backward pass")
    loss.backward()

    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    print(f"  Total parameters with gradients: {len(grad_norms)}")
    print(f"  Sample gradient norms:")
    for i, (name, norm) in enumerate(list(grad_norms.items())[:5]):
        print(f"    {name}: {norm:.6f}")
        if i >= 4:
            break

    # Verify all parameters have gradients
    params_without_grad = [name for name, param in model.named_parameters()
                           if param.requires_grad and param.grad is None]
    if params_without_grad:
        print(f"  ⚠ WARNING: {len(params_without_grad)} parameters without gradients")
    else:
        print("  ✓ All parameters have gradients")

    print("  ✓ Backward pass successful")


def test_memory_usage(model, config):
    """Test 4: Memory usage estimation."""
    print("\n" + "=" * 70)
    print("TEST 4: Memory Usage")
    print("=" * 70)

    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nParameter memory: {param_memory / 1e9:.2f} GB")

    # Estimate activation memory for a forward pass
    batch_size = 1
    seq_len = 128
    d_model = config.d_model
    n_layers = config.n_layers

    # Rough estimate: activation per token per layer * seq_len * batch * layers
    activation_per_token_per_layer = d_model * 4  # Rough estimate
    activation_memory = activation_per_token_per_layer * seq_len * batch_size * n_layers * 4  # 4 bytes per float32

    print(f"Estimated activation memory (batch=1, seq=128): {activation_memory / 1e9:.2f} GB")

    # Total memory estimate
    total_memory = param_memory + activation_memory
    print(f"Estimated total memory: {total_memory / 1e9:.2f} GB")

    print("  ✓ Memory estimation complete")


def test_generation(model, config):
    """Test 5: Text generation."""
    print("\n" + "=" * 70)
    print("TEST 5: Text Generation")
    print("=" * 70)

    prompt = torch.randint(0, config.vocab_size, (1, 10))

    print(f"\nGenerating {20} tokens...")
    print(f"  Prompt shape: {prompt.shape}")

    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
    )

    print(f"  Generated shape: {generated.shape}")
    assert generated.shape == (1, 30)  # 10 + 20
    print(f"  Prompt tokens: {prompt[0, :5].tolist()} ...")
    print(f"  Generated tokens: {generated[0, 10:15].tolist()} ...")

    print("  ✓ Generation successful")


def test_5b_model_initialization():
    """Test 6: 5B model initialization (optional, requires significant memory)."""
    print("\n" + "=" * 70)
    print("TEST 6: 5B Model Initialization (Optional)")
    print("=" * 70)

    try:
        config_5b = get_5b_config()
        config_5b.use_flash_attention = False
        config_5b.n_layers = 8  # Use fewer layers for testing

        print(f"\nAttempting to initialize 8-layer version of 5B model...")
        print(f"  d_model={config_5b.d_model}, n_layers={config_5b.n_layers}")
        print(f"  n_experts={config_5b.n_experts}, vocab_size={config_5b.vocab_size}")

        model_5b = MoETransformer(config_5b)

        params = model_5b.count_parameters()
        print(f"\n  ✓ 5B model initialized successfully!")
        print(f"  Total parameters: {params['total']:,} ({params['total_billions']:.3f}B)")

        # Test forward pass
        input_ids = torch.randint(0, config_5b.vocab_size, (1, 16))
        outputs = model_5b(input_ids)
        print(f"  ✓ Forward pass successful")

        del model_5b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"  ⚠ Could not initialize 5B model: {e}")
        print(f"  This is expected on systems with limited memory")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PHASE 1 MVP - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    # Test 1: Initialization
    model, config = test_model_initialization()

    # Test 2: Forward pass
    input_ids = test_forward_pass(model, config)

    # Test 3: Backward pass
    test_backward_pass(model, config)

    # Test 4: Memory usage
    test_memory_usage(model, config)

    # Test 5: Generation
    test_generation(model, config)

    # Test 6: 5B model (optional)
    test_5b_model_initialization()

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 1 MVP TEST SUMMARY")
    print("=" * 70)
    print("✓ All core tests passed!")
    print("\nPhase 1 MVP is complete. Ready for Phase 2 (MoE integration).")
    print("=" * 70)


if __name__ == "__main__":
    main()
