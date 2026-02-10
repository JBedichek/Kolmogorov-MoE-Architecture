"""
Test that Muon optimizer correctly handles 3D conv1d parameters.
"""

import torch
import torch.nn as nn
from moe_arch.training.muon_optimizer import get_muon_optimizer

def test_conv_classification():
    """Test that conv1d 3D parameters are correctly classified to AdamW."""
    print("=" * 70)
    print("TESTING MUON OPTIMIZER WITH 3D CONV PARAMETERS")
    print("=" * 70)

    # Create a model with various parameter shapes like Mamba
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 2D linear weights (should go to Muon)
            self.linear1 = nn.Linear(256, 512, bias=False)
            self.linear2 = nn.Linear(512, 256, bias=False)

            # 1D parameters (should go to AdamW)
            self.norm = nn.LayerNorm(256)

            # 3D conv1d weights like Mamba (should go to AdamW)
            self.conv1d = nn.Conv1d(512, 512, kernel_size=4, groups=512, bias=False)

            # Embedding (should go to AdamW)
            self.embedding = nn.Embedding(1000, 256)

        def forward(self, x):
            return x

    model = TestModel()

    print("\n1. Model Parameter Shapes:")
    for name, param in model.named_parameters():
        print(f"   {name:30s} shape: {str(list(param.shape)):20s} dim: {param.dim()}")

    # Get optimizer
    print("\n2. Creating Hybrid Muon + AdamW Optimizer:")
    optimizer = get_muon_optimizer(model, lr=1e-3)

    # Verify parameter groups
    print("\n3. Verifying Parameter Classification:")

    # Check that conv1d went to AdamW (not Muon)
    conv_param = model.conv1d.weight
    print(f"\n   Conv1d weight shape: {list(conv_param.shape)} (dim={conv_param.dim()})")

    # Find which optimizer has the conv parameter
    found_in_muon = False
    found_in_adamw = False

    if optimizer.muon_opt:
        for group in optimizer.muon_opt.param_groups:
            for p in group['params']:
                if p is conv_param:
                    found_in_muon = True
                    break

    if optimizer.adamw_opt:
        for group in optimizer.adamw_opt.param_groups:
            for p in group['params']:
                if p is conv_param:
                    found_in_adamw = True
                    break

    print(f"   Conv1d in Muon: {found_in_muon}")
    print(f"   Conv1d in AdamW: {found_in_adamw}")

    if found_in_adamw and not found_in_muon:
        print(f"   ✓ Conv1d correctly classified to AdamW!")
    else:
        print(f"   ✗ ERROR: Conv1d incorrectly classified!")
        return False

    # Check that 2D linear weights went to Muon
    linear_param = model.linear1.weight
    print(f"\n   Linear weight shape: {list(linear_param.shape)} (dim={linear_param.dim()})")

    found_in_muon = False
    found_in_adamw = False

    if optimizer.muon_opt:
        for group in optimizer.muon_opt.param_groups:
            for p in group['params']:
                if p is linear_param:
                    found_in_muon = True
                    break

    if optimizer.adamw_opt:
        for group in optimizer.adamw_opt.param_groups:
            for p in group['params']:
                if p is linear_param:
                    found_in_adamw = True
                    break

    print(f"   Linear in Muon: {found_in_muon}")
    print(f"   Linear in AdamW: {found_in_adamw}")

    if found_in_muon and not found_in_adamw:
        print(f"   ✓ Linear correctly classified to Muon!")
    else:
        print(f"   ✗ ERROR: Linear incorrectly classified!")
        return False

    # Test optimization step
    print("\n4. Testing Optimization Step:")
    try:
        # Create input that exercises all parameters
        input_ids = torch.randint(0, 1000, (2, 10))
        x = model.embedding(input_ids)  # (2, 10, 256)
        x = model.linear1(x)  # (2, 10, 512)
        x = x.transpose(1, 2)  # (2, 512, 10) for conv1d
        x = model.conv1d(x)  # (2, 512, 7)
        x = x.transpose(1, 2)  # (2, 7, 512)
        x = model.linear2(x)  # (2, 7, 256)
        x = model.norm(x)  # (2, 7, 256)
        loss = x.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"   ✓ Optimization step successful!")
    except Exception as e:
        print(f"   ✗ ERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("TEST PASSED!")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ 3D Conv1d parameters correctly routed to AdamW")
    print("  ✓ 2D Linear parameters correctly routed to Muon")
    print("  ✓ Optimization step works without errors")
    print("  ✓ Fix resolves PyTorch Muon 2D-only constraint")

    return True

if __name__ == "__main__":
    success = test_conv_classification()
    if not success:
        exit(1)
