"""
Debug training issues by inspecting checkpoints.
Check if parameters are actually updating during training.
"""

import torch
import numpy as np
from pathlib import Path
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import get_3b_config

def analyze_checkpoint(checkpoint_path: str):
    """Analyze a checkpoint to see if training is working."""
    print("="*70)
    print("CHECKPOINT TRAINING ANALYSIS")
    print("="*70)

    # Load checkpoint
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    # Get training info
    if 'step' in checkpoint:
        print(f"   Training step: {checkpoint['step']}")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"   Loss: {checkpoint['loss']:.4f}")

    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\n2. Model config:")
        print(f"   d_model: {config.d_model}")
        print(f"   n_layers: {config.n_layers}")
        print(f"   n_experts: {config.n_experts}")
        print(f"   moe_top_k: {config.moe_top_k}")
    else:
        print("\n2. No config in checkpoint, using default 3B config")
        config = get_3b_config()

    # Load model state
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model', None))
    if model_state is None:
        print("\n✗ ERROR: No model state found in checkpoint!")
        return

    print(f"\n3. Model state dict has {len(model_state)} parameters")

    # Create fresh model for comparison
    print("\n4. Creating fresh model with same config...")
    fresh_model = MoETransformer(config)
    fresh_state = fresh_model.state_dict()

    # Compare parameters
    print("\n5. Comparing checkpoint vs fresh initialization...")

    differences = []
    zeros = []
    unchanged = []

    for name in model_state.keys():
        if name not in fresh_state:
            print(f"   Warning: {name} in checkpoint but not in fresh model")
            continue

        ckpt_param = model_state[name]
        fresh_param = fresh_state[name]

        # Check if shapes match
        if ckpt_param.shape != fresh_param.shape:
            print(f"   Warning: Shape mismatch for {name}")
            print(f"      Checkpoint: {ckpt_param.shape}, Fresh: {fresh_param.shape}")
            continue

        # Calculate difference
        diff = (ckpt_param - fresh_param).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Check if parameter moved at all
        if max_diff < 1e-8:
            unchanged.append(name)
        elif max_diff < 1e-6:
            zeros.append((name, max_diff, mean_diff))
        else:
            differences.append((name, max_diff, mean_diff, ckpt_param.abs().mean().item()))

    # Report findings
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\n✓ Parameters that changed significantly: {len(differences)}")
    print(f"⚠ Parameters with tiny changes: {len(zeros)}")
    print(f"✗ Parameters completely unchanged: {len(unchanged)}")

    if unchanged:
        print(f"\n{'='*70}")
        print("FROZEN PARAMETERS (NOT MOVING AT ALL)")
        print(f"{'='*70}")
        for name in unchanged[:10]:
            print(f"   {name}")
        if len(unchanged) > 10:
            print(f"   ... and {len(unchanged) - 10} more")

    if zeros:
        print(f"\n{'='*70}")
        print("BARELY MOVING PARAMETERS (may indicate problem)")
        print(f"{'='*70}")
        for name, max_d, mean_d in zeros[:10]:
            print(f"   {name:50s} max_diff={max_d:.2e}, mean={mean_d:.2e}")
        if len(zeros) > 10:
            print(f"   ... and {len(zeros) - 10} more")

    if differences:
        print(f"\n{'='*70}")
        print("PROPERLY UPDATING PARAMETERS")
        print(f"{'='*70}")
        # Sort by magnitude of change
        differences.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 largest changes:")
        for name, max_d, mean_d, param_mag in differences[:10]:
            relative_change = max_d / (param_mag + 1e-8)
            print(f"   {name:50s}")
            print(f"      max_diff={max_d:.2e}, mean={mean_d:.2e}, param_mag={param_mag:.2e}")
            print(f"      relative_change={relative_change:.2%}")

        print(f"\nBottom 10 smallest changes (but still moving):")
        for name, max_d, mean_d, param_mag in differences[-10:]:
            relative_change = max_d / (param_mag + 1e-8)
            print(f"   {name:50s}")
            print(f"      max_diff={max_d:.2e}, mean={mean_d:.2e}")

    # Overall statistics
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}")

    total_params = len(model_state)
    pct_changed = len(differences) / total_params * 100
    pct_frozen = len(unchanged) / total_params * 100

    print(f"Total parameters: {total_params}")
    print(f"Changed significantly: {pct_changed:.1f}%")
    print(f"Frozen/unchanged: {pct_frozen:.1f}%")

    if pct_frozen > 50:
        print(f"\n✗ CRITICAL: {pct_frozen:.0f}% of parameters are frozen!")
        print("   Possible causes:")
        print("   - Requires_grad=False on some parameters")
        print("   - Optimizer not including all parameters")
        print("   - Zero gradients (check gradient flow)")
    elif pct_frozen > 10:
        print(f"\n⚠ WARNING: {pct_frozen:.0f}% of parameters are frozen")
        print("   This might be intentional (e.g., embeddings) or a bug")
    else:
        print(f"\n✓ Good: Most parameters are updating")

    if len(differences) > 0:
        all_max_diffs = [d[1] for d in differences]
        all_mean_diffs = [d[2] for d in differences]
        all_param_mags = [d[3] for d in differences]

        print(f"\nParameter update statistics (for moving params):")
        print(f"   Max difference range: [{min(all_max_diffs):.2e}, {max(all_max_diffs):.2e}]")
        print(f"   Mean difference range: [{min(all_mean_diffs):.2e}, {max(all_mean_diffs):.2e}]")
        print(f"   Median max diff: {np.median(all_max_diffs):.2e}")
        print(f"   Median mean diff: {np.median(all_mean_diffs):.2e}")

        # Check if updates are too small
        median_max = np.median(all_max_diffs)
        median_param_mag = np.median(all_param_mags)
        relative_update = median_max / median_param_mag

        print(f"\nRelative update size: {relative_update:.2%}")
        if relative_update < 0.001:
            print("   ✗ PROBLEM: Updates are extremely small relative to parameter magnitude")
            print("   Possible causes:")
            print("   - Learning rate too low")
            print("   - Gradients too small (vanishing gradients)")
            print("   - Loss not flowing back properly")
        elif relative_update < 0.01:
            print("   ⚠ Updates are small but might be okay for early training")
        else:
            print("   ✓ Update magnitude seems reasonable")

    # Check optimizer state
    if 'optimizer_state_dict' in checkpoint:
        print(f"\n{'='*70}")
        print("OPTIMIZER STATE")
        print(f"{'='*70}")
        opt_state = checkpoint['optimizer_state_dict']
        print(f"Optimizer state keys: {list(opt_state.keys())}")

        if 'param_groups' in opt_state:
            for i, group in enumerate(opt_state['param_groups']):
                print(f"\nParam group {i}:")
                print(f"   lr: {group.get('lr', 'N/A')}")
                print(f"   weight_decay: {group.get('weight_decay', 'N/A')}")
                print(f"   params: {len(group.get('params', []))}")

        # Check if optimizer has accumulated state (momentum, etc.)
        if 'state' in opt_state:
            opt_states = opt_state['state']
            print(f"\nOptimizer state for {len(opt_states)} parameters")
            if len(opt_states) == 0:
                print("   ⚠ WARNING: Optimizer has no accumulated state (no steps taken?)")
            else:
                print("   ✓ Optimizer has accumulated state")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_interrupted.pt"

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nLooking for checkpoints...")
        checkpoints = list(Path("checkpoints").glob("*.pt")) if Path("checkpoints").exists() else []
        if checkpoints:
            print(f"Found {len(checkpoints)} checkpoint(s):")
            for cp in checkpoints:
                print(f"   {cp}")
            checkpoint_path = str(checkpoints[0])
            print(f"\nUsing: {checkpoint_path}")
        else:
            print("No checkpoints found!")
            exit(1)

    analyze_checkpoint(checkpoint_path)
