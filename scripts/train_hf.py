"""
Full-scale training script using HuggingFace Trainer.

Usage:
    python scripts/train_hf.py --config configs/training_1.5b_hf.yaml
    python scripts/train_hf.py --config configs/training_1.5b_hf.yaml --resume checkpoints/checkpoint-1000
"""

import argparse
import sys
import os
import yaml
import torch
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.model.transformer import MoETransformer
from moe_arch.data.tokenizer import TokenizerWrapper
from moe_arch.data.dolma_dataset import get_dataloaders
from moe_arch.training.muon_optimizer import get_muon_optimizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class MoEModelWrapper(torch.nn.Module):
    """
    Wrapper to make MoE model compatible with HuggingFace Trainer.

    Handles:
    - Multi-token prediction losses
    - MoE auxiliary losses (load balancing, router z-loss)
    - MoD auxiliary losses
    """
    def __init__(self, model: MoETransformer):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with HF Trainer.

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) - same as input_ids for causal LM

        Returns:
            dict with 'loss' and 'logits'
        """
        # Call model
        outputs = self.model(input_ids, labels=labels)

        # Combine all losses for HF Trainer
        total_loss = outputs['loss']  # Already includes lm_loss + aux_losses

        return {
            'loss': total_loss,
            'logits': outputs['logits'],  # (batch, seq, vocab)
        }


class MoETrainer(Trainer):
    """
    Custom HuggingFace Trainer for MoE model.

    Overrides:
    - create_optimizer: Use Muon optimizer instead of AdamW
    - create_scheduler: Use custom LR scheduler compatible with Muon
    - compute_loss: Already handled by model wrapper
    """

    def __init__(self, muon_config: dict, lr_config: dict, *args, **kwargs):
        self.muon_config = muon_config
        self.lr_config = lr_config
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Create Muon optimizer instead of default AdamW."""
        if self.optimizer is None:
            self.optimizer = get_muon_optimizer(
                self.model.model,  # Unwrap from MoEModelWrapper
                lr=self.args.learning_rate,
                momentum=self.muon_config.get('momentum', 0.95),
                weight_decay=self.muon_config.get('weight_decay', 0.01),
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Create custom LR scheduler compatible with HybridOptimizer.

        Since HybridOptimizer isn't recognized by HF's get_scheduler,
        we use our custom LR scheduler from the training module.
        """
        from moe_arch.training.lr_schedule import get_lr_scheduler

        if self.lr_scheduler is None:
            # Use our custom LR scheduler
            self.lr_scheduler = get_lr_scheduler(
                optimizer=self.optimizer,
                total_tokens=self.lr_config['total_tokens'],
                batch_size=self.lr_config['effective_batch_size'],
                seq_len=self.lr_config['seq_len'],
                warmup_steps=self.args.warmup_steps,
                max_lr=self.args.learning_rate,
                min_lr_ratio=self.lr_config.get('min_lr_ratio', 0.1),
            )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss - model wrapper already handles everything.

        Note: HuggingFace Trainer handles gradient accumulation scaling automatically.
        We just return the mean loss per sample.
        """
        # Remove any extra keys that aren't model inputs
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'labels': inputs['labels'] if 'labels' in inputs else inputs['input_ids'],
        }

        # CRITICAL: Skip empty sequences (can cause NaN loss)
        seq_len = model_inputs['input_ids'].shape[1]
        if seq_len == 0:
            # Return zero loss for empty sequences
            return torch.tensor(0.0, device=model_inputs['input_ids'].device, requires_grad=True)

        outputs = model(**model_inputs)
        loss = outputs['loss']

        # Return raw loss - HF Trainer handles all scaling automatically
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train MoE with HuggingFace Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_1.5b_hf.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to HF checkpoint to resume from",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MoE Transformer Training (HuggingFace Trainer)")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading config from {args.config}...")
    train_config = load_config(args.config)
    print("  ✓ Config loaded")

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

    # Keep in FP32 (HF Trainer handles BF16 via training_args.bf16)
    print("  ✓ Model weights in FP32 (HF Trainer handles BF16 compute)")

    # Count parameters
    params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,} ({params['total_billions']:.3f}B)")
    print(f"  Active: {params['active']:,} ({params['active_billions']:.3f}B)")
    print(f"  Sparsity: {params['sparsity']:.1%}")

    # Wrap model for HF Trainer
    wrapped_model = MoEModelWrapper(model)
    print("  ✓ Model wrapped for HF Trainer")

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

    # Convert to HF dataset format
    # Since get_dataloaders returns PyTorch DataLoader, we'll use the underlying dataset
    # For now, we'll just use the DataLoader as-is with HF Trainer
    print("  ✓ Dataloaders created")

    # Calculate training steps
    total_tokens = train_config["training"]["total_tokens"]
    batch_size = train_config["training"]["batch_size"]
    grad_accum = train_config["training"]["gradient_accumulation_steps"]
    seq_len = model_config.max_seq_len

    effective_batch_size = batch_size * grad_accum
    tokens_per_step = effective_batch_size * seq_len
    max_steps = total_tokens // tokens_per_step

    print(f"\nTraining configuration:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Max steps: {max_steps:,}")

    # HuggingFace TrainingArguments
    hf_config = train_config.get("huggingface", {})
    training_args = TrainingArguments(
        # Output
        output_dir=train_config["training"]["checkpoint_dir"],
        run_name=train_config.get("wandb", {}).get("run_name", "moe-1.5b"),

        # Training
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        # Learning rate
        learning_rate=train_config["training"]["max_lr"],
        warmup_steps=train_config["training"]["warmup_steps"],
        lr_scheduler_type="cosine",

        # Optimizer (will be overridden by create_optimizer)
        optim="adamw_torch",
        weight_decay=train_config["optimizer"].get("weight_decay", 0.01),
        max_grad_norm=train_config["training"]["gradient_clip_norm"],

        # Mixed precision
        bf16=hf_config.get("bf16", True),
        bf16_full_eval=False,

        # Logging
        logging_dir=hf_config.get("logging_dir", "./logs"),
        logging_steps=train_config["training"]["log_interval"],
        logging_first_step=True,
        report_to=["wandb"] if hf_config.get("use_wandb", False) else ["none"],

        # Checkpointing
        save_strategy="steps",
        save_steps=train_config["training"]["save_interval"],
        save_total_limit=hf_config.get("save_total_limit", 3),

        # Evaluation
        eval_strategy="steps" if val_loader is not None else "no",
        eval_steps=train_config["training"].get("eval_interval", 1000),

        # Performance
        dataloader_num_workers=tokenizer_config.get("num_workers", 4),
        remove_unused_columns=False,
        dataloader_pin_memory=True,

        # Other
        seed=hf_config.get("seed", 42),
        disable_tqdm=False,
    )

    print(f"\nHuggingFace TrainingArguments:")
    print(f"  Max steps: {training_args.max_steps:,}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Warmup steps: {training_args.warmup_steps}")
    print(f"  BF16: {training_args.bf16}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer.tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Extract datasets from dataloaders
    # For streaming datasets, we need to pass the dataset directly
    train_dataset = train_loader.dataset if hasattr(train_loader, 'dataset') else None

    # For eval, only use if it exists
    # Don't check val_loader with boolean (causes len() call on IterableDataset)
    eval_dataset = None
    if val_loader is not None:
        if hasattr(val_loader, 'dataset'):
            eval_dataset = val_loader.dataset

    # Prepare LR config for custom scheduler
    lr_config = {
        'total_tokens': total_tokens,
        'effective_batch_size': effective_batch_size,
        'seq_len': seq_len,
        'min_lr_ratio': train_config["training"].get("min_lr_ratio", 0.1),
    }

    # Create trainer
    print("\nInitializing MoE Trainer...")
    trainer = MoETrainer(
        muon_config=train_config.get("optimizer", {}),
        lr_config=lr_config,
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    print("  ✓ Trainer created")

    # Resume from checkpoint if specified
    resume_from_checkpoint = args.resume
    if resume_from_checkpoint:
        print(f"\nResuming from checkpoint: {resume_from_checkpoint}")

    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("HuggingFace Trainer auto-saves checkpoints")
        sys.exit(0)


if __name__ == "__main__":
    main()
