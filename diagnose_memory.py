#!/usr/bin/env python3
"""
Detailed memory leak diagnosis script.
Tracks memory at each phase of training to identify where growth occurs.
"""

import torch
import gc
import sys
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_mem():
    """Return allocated and reserved memory in GB."""
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
        torch.cuda.max_memory_allocated() / 1e9,
    )

def print_mem(label, prev_alloc=None):
    alloc, reserved, peak = get_mem()
    delta = f" (Δ{alloc - prev_alloc:+.3f}GB)" if prev_alloc is not None else ""
    print(f"  {label}: alloc={alloc:.3f}GB, reserved={reserved:.3f}GB, peak={peak:.3f}GB{delta}")
    return alloc

def main():
    print("=" * 70)
    print("MEMORY LEAK DIAGNOSIS")
    print("=" * 70)

    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/production_training.yaml'
    print(f"\nUsing config: {config_path}")

    # Load config
    config_dict = load_config(config_path)
    model_cfg = config_dict['model']

    print(f"\nModel config:")
    print(f"  n_layers: {model_cfg.get('n_layers')}")
    print(f"  n_experts: {model_cfg.get('n_experts')}")
    print(f"  d_model: {model_cfg.get('d_model')}")
    print(f"  max_seq_len: {model_cfg.get('max_seq_len')}")
    print(f"  mod_enabled: {model_cfg.get('mod_enabled')}")

    # Clear everything
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    prev = print_mem("Initial state")

    # Create model config
    config = AdvancedMoEConfig(
        vocab_size=model_cfg.get('vocab_size', 50000),
        d_model=model_cfg.get('d_model', 2048),
        n_layers=model_cfg.get('n_layers', 32),
        n_heads=model_cfg.get('n_heads', 16),
        n_kv_heads=model_cfg.get('n_kv_heads', 4),
        head_dim=model_cfg.get('head_dim', 128),
        d_ff=model_cfg.get('d_ff', 5632),
        d_ff_expert=model_cfg.get('d_ff_expert', 2816),
        max_seq_len=model_cfg.get('max_seq_len', 2048),
        n_experts=model_cfg.get('n_experts', 16),
        moe_top_k=model_cfg.get('moe_top_k', 2),
        moe_capacity_factor=model_cfg.get('moe_capacity_factor', 1.25),
        moe_load_balance_loss_weight=model_cfg.get('moe_load_balance_loss_weight', 0.01),
        moe_router_z_loss_weight=model_cfg.get('moe_router_z_loss_weight', 0.001),
        moe_layers=tuple(model_cfg.get('moe_layers', [])),
        moe_implementation=model_cfg.get('moe_implementation', 'batched'),
        mod_enabled=model_cfg.get('mod_enabled', True),
        mod_capacity_factor=model_cfg.get('mod_capacity_factor', 0.75),
        mod_router_hidden_dim=model_cfg.get('mod_router_hidden_dim', 128),
        mod_load_balance_loss_weight=model_cfg.get('mod_load_balance_loss_weight', 0.001),
        mamba_enabled=model_cfg.get('mamba_enabled', False),
        mamba_layers=tuple(model_cfg.get('mamba_layers', [])),
        n_pred_tokens=model_cfg.get('n_pred_tokens', 1),
        aux_loss_weights=tuple(model_cfg.get('aux_loss_weights', [1.0])),
        use_flash_attention=model_cfg.get('use_flash_attention', True),
        dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )

    # Create model
    print("\n--- Model Creation ---")
    model = MoETransformer(config)
    prev = print_mem("After model creation (CPU)", prev)

    model = model.cuda()
    prev = print_mem("After model.cuda()", prev)

    model = model.to(torch.bfloat16)
    prev = print_mem("After to(bfloat16)", prev)

    model.train()
    model.gradient_checkpointing_enable()
    prev = print_mem("After gradient_checkpointing_enable()", prev)

    # Create optimizer
    print("\n--- Optimizer Creation ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    prev = print_mem("After optimizer creation", prev)

    # Training simulation
    print("\n--- Training Steps ---")
    batch_size = 1
    seq_len = config.max_seq_len
    vocab_size = config.vocab_size

    # Track memory per step
    step_memory = []

    num_steps = 50
    grad_accum = 8  # Simulate gradient accumulation

    for step in range(num_steps):
        # Create input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        labels = input_ids.clone()

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        # Backward pass (with gradient accumulation)
        loss = loss / grad_accum
        loss.backward()

        # Optimizer step every grad_accum steps
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Force cleanup
            gc.collect()
            torch.cuda.empty_cache()

            alloc, reserved, peak = get_mem()
            step_memory.append(alloc)
            print(f"  Step {step+1:3d} (opt step): alloc={alloc:.3f}GB, reserved={reserved:.3f}GB, peak={peak:.3f}GB")
        elif step < 5 or step % 10 == 0:
            alloc, reserved, peak = get_mem()
            print(f"  Step {step+1:3d} (accum):    alloc={alloc:.3f}GB, reserved={reserved:.3f}GB")

        # Clear references
        del outputs, loss, input_ids, labels

    # Analysis
    print("\n--- Analysis ---")
    if len(step_memory) >= 2:
        growth_per_opt_step = (step_memory[-1] - step_memory[0]) / (len(step_memory) - 1)
        print(f"Memory at first opt step: {step_memory[0]:.3f}GB")
        print(f"Memory at last opt step:  {step_memory[-1]:.3f}GB")
        print(f"Growth per opt step: {growth_per_opt_step*1000:.1f}MB")

        if growth_per_opt_step > 0.01:  # >10MB per opt step
            print(f"\n⚠️  MEMORY LEAK DETECTED: ~{growth_per_opt_step*1000:.0f}MB per optimizer step")
            print("    This will cause OOM over time!")

            # Try to identify cause
            print("\nPossible causes:")
            print("  1. Computation graphs not being freed")
            print("  2. Tensors stored on modules accumulating")
            print("  3. Optimizer state growing")
            print("  4. CUDA memory fragmentation")
        else:
            print("\n✓ No significant memory leak detected")
            print(f"    Memory is stable at ~{step_memory[-1]:.2f}GB")

    # Check for any growing attributes on model
    print("\n--- Checking for growing state in model ---")
    for name, module in model.named_modules():
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, list) and len(attr) > 0:
                    print(f"  {name}.{attr_name}: list with {len(attr)} items")
                elif isinstance(attr, dict) and len(attr) > 0:
                    print(f"  {name}.{attr_name}: dict with {len(attr)} items")
            except:
                pass

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
