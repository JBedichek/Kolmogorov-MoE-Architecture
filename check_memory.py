#!/usr/bin/env python3
"""Quick script to check memory allocation during a few forward/backward passes."""

import torch
import gc
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def print_memory(step, label):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_alloc = torch.cuda.max_memory_allocated() / 1e9
    print(f"  [{step:02d}] {label}: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB, max={max_alloc:.2f}GB")

def main():
    print("=" * 60)
    print("MEMORY LEAK DETECTION")
    print("=" * 60)

    # Load config
    config_dict = load_config('configs/production_training.yaml')
    model_cfg = config_dict['model']

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

    print(f"\nConfig: {model_cfg.get('n_layers')} layers, {model_cfg.get('n_experts')} experts")
    print(f"MoD capacity: {model_cfg.get('mod_capacity_factor')}")
    print(f"Seq len: {model_cfg.get('max_seq_len')}")

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_memory(0, "Initial")

    # Create model
    model = MoETransformer(config).cuda().to(torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable()

    print_memory(0, "After model creation")

    # Create input
    batch_size = 1
    seq_len = config.max_seq_len
    vocab_size = config.vocab_size

    print(f"\nRunning {40} forward/backward passes...")
    print(f"Batch: {batch_size}, Seq: {seq_len}\n")

    prev_alloc = torch.cuda.memory_allocated()

    for step in range(40):
        # Create fresh input each step
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        labels = input_ids.clone()

        # Forward
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        # Backward
        loss.backward()

        # Simulate gradient accumulation (no optimizer step)
        if (step + 1) % 8 == 0:
            # Every 8 steps, zero grads (simulate optimizer step)
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            print_memory(step + 1, f"After optimizer step")

        # Check memory growth
        curr_alloc = torch.cuda.memory_allocated()
        growth = (curr_alloc - prev_alloc) / 1e6

        if step < 5 or step % 5 == 0 or abs(growth) > 100:
            print_memory(step + 1, f"Step (growth: {growth:+.1f}MB)")

        prev_alloc = curr_alloc

        # Clear references
        del outputs, loss, input_ids, labels

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    final_alloc = torch.cuda.memory_allocated() / 1e9
    max_alloc = torch.cuda.max_memory_allocated() / 1e9
    print(f"Final allocated: {final_alloc:.2f}GB")
    print(f"Peak allocated: {max_alloc:.2f}GB")

    if max_alloc > final_alloc * 1.5:
        print("\n⚠️  Peak much higher than final - possible memory spike during forward/backward")

    # Check for memory growth pattern
    gc.collect()
    torch.cuda.empty_cache()
    after_cleanup = torch.cuda.memory_allocated() / 1e9
    print(f"After cleanup: {after_cleanup:.2f}GB")

if __name__ == '__main__':
    main()
