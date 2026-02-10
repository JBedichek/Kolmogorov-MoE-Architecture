"""
Test Muon optimizer with the actual MoE Transformer model.
"""

import torch
from moe_arch.model.config import get_test_config
from moe_arch.model.transformer import MoETransformer
from moe_arch.training.muon_optimizer import get_muon_optimizer

def test_muon_with_moe_model():
    """Test Muon optimizer with MoE Transformer model."""
    print("=" * 70)
    print("TESTING MUON OPTIMIZER WITH MOE TRANSFORMER")
    print("=" * 70)

    # Get test config (smaller model)
    config = get_test_config()
    print(f"\n1. Creating test model...")
    print(f"   d_model: {config.d_model}")
    print(f"   n_layers: {config.n_layers}")
    print(f"   Mamba layers: {config.mamba_layers}")
    print(f"   MoE layers: {config.moe_layers}")

    # Create model
    model = MoETransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")

    # Check for 3D parameters (conv1d from Mamba)
    print(f"\n2. Checking for 3D parameters (Mamba conv1d):")
    conv3d_params = []
    linear2d_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.dim() == 3:
            conv3d_params.append((name, param.shape))
        elif param.dim() == 2:
            linear2d_params.append((name, param.shape))
        else:
            other_params.append((name, param.shape))

    if conv3d_params:
        print(f"   Found {len(conv3d_params)} 3D parameters:")
        for name, shape in conv3d_params[:5]:  # Show first 5
            print(f"     {name}: {list(shape)}")
        if len(conv3d_params) > 5:
            print(f"     ... and {len(conv3d_params) - 5} more")

    print(f"   Found {len(linear2d_params)} 2D parameters")
    print(f"   Found {len(other_params)} other parameters")

    # Create optimizer
    print(f"\n3. Creating Muon optimizer:")
    try:
        optimizer = get_muon_optimizer(model, lr=1e-3)
        print(f"   ✓ Optimizer created successfully!")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward + backward + optimizer step
    print(f"\n4. Testing training step:")
    try:
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward
        model.train()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']
        print(f"   Loss: {loss.item():.4f}")

        # Backward
        loss.backward()
        print(f"   ✓ Backward pass complete")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        print(f"   ✓ Optimizer step complete")

        # Second step to verify it works consistently
        outputs2 = model(input_ids, labels=input_ids)
        loss2 = outputs2['loss']
        loss2.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"   ✓ Second step complete (loss: {loss2.item():.4f})")

    except Exception as e:
        print(f"   ✗ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("TEST PASSED!")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ MoE Transformer model created")
    print(f"  ✓ Found {len(conv3d_params)} 3D conv parameters (Mamba)")
    print("  ✓ Muon optimizer created without errors")
    print("  ✓ Training step (forward + backward + step) successful")
    print("  ✓ Ready for actual training!")

    return True

if __name__ == "__main__":
    success = test_muon_with_moe_model()
    if not success:
        exit(1)
