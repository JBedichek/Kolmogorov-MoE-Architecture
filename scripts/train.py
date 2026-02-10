"""
Main training script for MoE Transformer.

Usage:
    python scripts/train.py --config configs/training_dolma.yaml
    python scripts/train.py --config configs/training_dolma.yaml --resume checkpoints/checkpoint_step_1000.pt
"""

import argparse
import sys
import os
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.model.transformer import MoETransformer
from moe_arch.data.tokenizer import TokenizerWrapper
from moe_arch.data.dolma_dataset import get_dataloaders
from moe_arch.training.muon_optimizer import get_muon_optimizer
from moe_arch.training.lr_schedule import get_lr_scheduler
from moe_arch.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train MoE Transformer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_1.5b.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MoE Transformer Training")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading config from {args.config}...")
    try:
        train_config = load_config(args.config)
        print("  ✓ Config loaded")
    except FileNotFoundError:
        print(f"  ✗ Config file not found: {args.config}")
        print("  Using default configuration...")
        train_config = {
            "model": {},
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "total_tokens": 100_000_000_000,  # 100B tokens
                "warmup_steps": 2000,
                "max_lr": 1e-3,
                "min_lr_ratio": 0.1,
                "gradient_clip_norm": 1.0,
                "checkpoint_dir": "./checkpoints",
                "log_interval": 10,
                "eval_interval": 1000,
                "save_interval": 1000,
            },
            "optimizer": {
                "type": "muon",
                "momentum": 0.95,
                "weight_decay": 0.01,
            },
            "data": {
                "tokenizer_name": "gpt2",
                "use_memory_mapped": False,
                "data_path": None,
                "num_workers": 4,
            },
        }

    # Create model configuration
    print("\nInitializing model configuration...")
    model_config = AdvancedMoEConfig(**train_config.get("model", {}))
    print(f"  Model: {model_config.n_layers} layers, {model_config.d_model} dim")
    print(f"  MoE: {model_config.n_experts} experts, top-{model_config.moe_top_k}")
    print(f"  MoD: {model_config.mod_enabled}, capacity={model_config.mod_capacity_factor}")
    print(f"  Mamba: {model_config.mamba_enabled}")
    print(f"  Multi-token: {model_config.n_pred_tokens} heads")

    # Initialize model
    print("\nInitializing model...")
    model = MoETransformer(model_config)
    model = model.to(args.device)

    # Keep model in FP32 for optimizer precision
    # Autocast in trainer.py handles BF16 compute (forward pass only)
    # This prevents BFloat16 underflow on small parameter updates (e.g., norms at 1.0)
    print("  ✓ Model weights in FP32 (autocast handles BF16 compute)")

    # Count parameters
    params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,} ({params['total_billions']:.3f}B)")
    print(f"  Active: {params['active']:,} ({params['active_billions']:.3f}B)")
    print(f"  Sparsity: {params['sparsity']:.1%} (inactive per forward pass)")
    print(f"  MoE layers: {params['n_moe_layers']}")
    print(f"  Mamba layers: {params['n_mamba_layers']}")

    # Compile model with max-autotune for maximum performance
    print("\nCompiling model with torch.compile (max-autotune mode)...")
    print("  This may take a few minutes on first forward pass...")
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    print("  ✓ Model compiled (expect 20-50% speedup after warmup)")

    # Enable gradient checkpointing for memory efficiency (saves ~25GB for 5B model)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  ✓ Gradient checkpointing enabled (saves ~30-40% memory)")

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer_config = train_config.get("data", {})
    tokenizer = TokenizerWrapper(
        tokenizer_name=tokenizer_config.get("tokenizer_name", "gpt2"),
        vocab_size=model_config.vocab_size,
    )

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        tokenizer=tokenizer,
        config=model_config,
        batch_size=train_config["training"]["batch_size"],
        num_workers=tokenizer_config.get("num_workers", 4),
        use_memory_mapped=tokenizer_config.get("use_memory_mapped", False),
        data_path=tokenizer_config.get("data_path", None),
    )
    print("  ✓ Dataloaders created")

    # Initialize optimizer
    print("\nInitializing optimizer...")
    optimizer_config = train_config.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "muon")

    if optimizer_type == "muon":
        # Use hybrid Muon + AdamW (Muon for 2D, AdamW for 1D)
        optimizer = get_muon_optimizer(
            model,
            lr=train_config["training"]["max_lr"],
            momentum=optimizer_config.get("momentum", 0.95),
            weight_decay=optimizer_config.get("weight_decay", 0.01),
        )
    else:
        # Fallback to AdamW
        print("  Using AdamW optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config["training"]["max_lr"],
            betas=(0.9, 0.95),
            weight_decay=optimizer_config.get("weight_decay", 0.01),
        )

    # Initialize learning rate scheduler
    print("\nInitializing LR scheduler...")
    # Use effective batch size (batch_size * gradient_accumulation_steps) for scheduler
    effective_batch_size = train_config["training"]["batch_size"] * train_config["training"]["gradient_accumulation_steps"]
    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        total_tokens=train_config["training"]["total_tokens"],
        batch_size=effective_batch_size,  # Use effective batch size!
        seq_len=model_config.max_seq_len,
        warmup_steps=train_config["training"]["warmup_steps"],
        max_lr=train_config["training"]["max_lr"],
        min_lr_ratio=train_config["training"]["min_lr_ratio"],
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=args.device,
        use_amp=True,  # Always use BF16 AMP
        gradient_accumulation_steps=train_config["training"]["gradient_accumulation_steps"],
        gradient_clip_norm=train_config["training"]["gradient_clip_norm"],
        checkpoint_dir=train_config["training"]["checkpoint_dir"],
        log_interval=train_config["training"]["log_interval"],
        eval_interval=train_config["training"]["eval_interval"],
        save_interval=train_config["training"]["save_interval"],
        use_wandb=args.use_wandb,
    )

    # Load checkpoint if resuming
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Calculate max steps
    total_tokens = train_config["training"]["total_tokens"]
    batch_size = train_config["training"]["batch_size"]
    grad_accum = train_config["training"]["gradient_accumulation_steps"]
    seq_len = model_config.max_seq_len

    effective_batch_size = batch_size * grad_accum
    tokens_per_step = effective_batch_size * seq_len
    max_steps = total_tokens // tokens_per_step

    print("\nTraining configuration:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Max steps: {max_steps:,}")

    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    try:
        trainer.train(max_steps=max_steps)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint before exit...")
        trainer.save_checkpoint("checkpoint_interrupted.pt")
        print("✓ Checkpoint saved")
        sys.exit(0)

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
