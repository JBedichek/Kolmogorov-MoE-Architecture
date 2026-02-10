"""
Main training loop for MoE model.

Features:
- Multi-token prediction loss
- Gradient accumulation
- Gradient checkpointing
- Mixed precision (BFloat16)
- Checkpointing and resumption
- Weights & Biases logging
- Evaluation during training
"""

import os
import time
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from pathlib import Path
from torch.amp import autocast, GradScaler
from tqdm import tqdm


class Trainer:
    """
    Main training loop for MoE Transformer.

    Handles:
    - Training with multi-token prediction
    - Gradient accumulation for large effective batch sizes
    - Mixed precision training
    - Checkpointing and resumption
    - Logging to Weights & Biases
    - Periodic evaluation
    """

    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        device: str = "cuda",
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        use_wandb: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            model: MoE Transformer model
            optimizer: Optimizer (Muon or AdamW)
            lr_scheduler: Learning rate scheduler
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            device: Device to train on
            use_amp: Whether to use automatic mixed precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
            gradient_clip_norm: Max gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints
            log_interval: Log every N steps
            eval_interval: Evaluate every N steps
            save_interval: Save checkpoint every N steps
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_wandb = use_wandb

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize gradient scaler for mixed precision
        # Note: GradScaler only needed for FP16, not BF16 (BF16 has same range as FP32)
        # Since we're using BF16 in autocast, we don't need the scaler
        self.scaler = None  # Disabled for BF16

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0

        # Initialize W&B if enabled
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="moe-architecture",
                    config=config,
                    name=config.get("run_name", "moe-5b"),
                )
                self.wandb = wandb
                print("✓ Weights & Biases initialized")
            except Exception as e:
                print(f"Warning: Could not initialize W&B: {e}")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  AMP: {use_amp}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Gradient clip norm: {gradient_clip_norm}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Log interval: {log_interval}")
        print(f"  Eval interval: {eval_interval}")
        print(f"  Save interval: {save_interval}")

    def train(self, max_steps: Optional[int] = None):
        """
        Main training loop.

        Args:
            max_steps: Maximum number of training steps (None = train forever)
        """
        print(f"\nStarting training...")
        print(f"  Max steps: {max_steps if max_steps else 'unlimited'}")
        print(f"  Starting from step {self.global_step}")

        self.model.train()
        train_iter = iter(self.train_loader)

        start_time = time.time()
        accumulated_loss = 0.0
        accumulated_lm_loss = 0.0
        accumulated_aux_loss = 0.0

        # Create progress bar
        pbar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

        while max_steps is None or self.global_step < max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with AMP
            if self.use_amp:
                with autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"] / self.gradient_accumulation_steps
            else:
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"] / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp and self.scaler is not None:
                # FP16 path (needs gradient scaling)
                self.scaler.scale(loss).backward()
            else:
                # BF16 or FP32 path (no scaling needed)
                loss.backward()

            # Accumulate losses for logging
            accumulated_loss += loss.item() * self.gradient_accumulation_steps
            accumulated_lm_loss += outputs["lm_loss"].item() / self.gradient_accumulation_steps
            accumulated_aux_loss += outputs["aux_loss"] / self.gradient_accumulation_steps

            # Update tokens seen
            batch_size, seq_len = input_ids.shape
            self.tokens_seen += batch_size * seq_len

            # Optimizer step every gradient_accumulation_steps
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp and self.scaler is not None:
                    # FP16: Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

                # Optimizer step
                if self.use_amp and self.scaler is not None:
                    # FP16: Use scaler for step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # BF16 or FP32: Direct optimizer step
                    self.optimizer.step()

                # LR scheduler step
                current_lr = self.lr_scheduler.step()

                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                # Average accumulated losses
                avg_loss = accumulated_loss / self.gradient_accumulation_steps
                avg_lm_loss = accumulated_lm_loss / self.gradient_accumulation_steps
                avg_aux_loss = accumulated_aux_loss / self.gradient_accumulation_steps

                # Update progress bar (inside optimizer step block)
                elapsed = time.time() - start_time
                tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0
                pbar.set_description(f"Training | Loss: {avg_loss:.4f} | {tokens_per_sec:,.0f} tok/s")
                pbar.set_postfix({
                    "lm": f"{avg_lm_loss:.4f}",
                    "aux": f"{avg_aux_loss:.6f}",
                    "lr": f"{current_lr:.2e}",
                })

                # Logging
                if self.global_step % self.log_interval == 0:
                    log_dict = {
                        "step": self.global_step,
                        "epoch": self.epoch,
                        "loss": avg_loss,
                        "lm_loss": avg_lm_loss,
                        "aux_loss": avg_aux_loss,
                        "lr": current_lr,
                        "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "tokens_seen": self.tokens_seen,
                        "tokens_per_sec": tokens_per_sec,
                    }

                    # Add multi-token losses if available
                    if "multitoken_loss_dict" in outputs and outputs["multitoken_loss_dict"]:
                        for key, value in outputs["multitoken_loss_dict"].items():
                            log_dict[f"mt_{key}"] = value

                    # W&B logging
                    if self.use_wandb:
                        self.wandb.log(log_dict, step=self.global_step)

                # Reset accumulators
                accumulated_loss = 0.0
                accumulated_lm_loss = 0.0
                accumulated_aux_loss = 0.0

                # Evaluation
                if self.global_step % self.eval_interval == 0 and self.global_step > 0:
                    eval_metrics = self.evaluate()
                    tqdm.write(f"  Eval | Loss {eval_metrics['eval_loss']:.4f} | PPL {eval_metrics['eval_ppl']:.2f}")

                    if self.use_wandb:
                        self.wandb.log(eval_metrics, step=self.global_step)

                    self.model.train()

                # Save checkpoint
                if self.global_step % self.save_interval == 0 and self.global_step > 0:
                    tqdm.write(f"  Saving checkpoint: checkpoint_step_{self.global_step}.pt")
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        print(f"\nTraining completed!")
        print(f"  Total steps: {self.global_step}")
        print(f"  Total tokens: {self.tokens_seen:,}")

        # Final checkpoint
        self.save_checkpoint("checkpoint_final.pt")

    @torch.no_grad()
    def evaluate(self, num_batches: int = 100) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            num_batches: Number of batches to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_lm_loss = 0.0
        total_tokens = 0

        val_iter = iter(self.val_loader)

        for i in range(num_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, labels=labels)

            batch_size, seq_len = input_ids.shape
            total_loss += outputs["loss"].item() * batch_size * seq_len
            total_lm_loss += outputs["lm_loss"].item() * batch_size * seq_len
            total_tokens += batch_size * seq_len

        avg_loss = total_loss / total_tokens
        avg_lm_loss = total_lm_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_lm_loss)).item()

        return {
            "eval_loss": avg_loss,
            "eval_lm_loss": avg_lm_loss,
            "eval_ppl": perplexity,
        }

    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "tokens_seen": self.tokens_seen,
            "config": self.config,
        }

        if self.use_amp and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.tokens_seen = checkpoint["tokens_seen"]

        if self.use_amp and self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"  ✓ Checkpoint loaded")
        print(f"  Resuming from step {self.global_step}, epoch {self.epoch}")
        print(f"  Tokens seen: {self.tokens_seen:,}")


if __name__ == "__main__":
    # Test trainer with dummy data
    from moe_arch.model.config import get_test_config
    from moe_arch.model.transformer import MoETransformer
    from moe_arch.training.muon_optimizer import get_muon_optimizer
    from moe_arch.training.lr_schedule import get_lr_scheduler
    from torch.utils.data import DataLoader, TensorDataset

    print("Testing Trainer...")

    # Create small model
    config = get_test_config()
    config.use_flash_attention = False
    config.n_layers = 2
    config.max_seq_len = 64

    model = MoETransformer(config)

    # Create optimizer and scheduler
    optimizer = get_muon_optimizer(model, lr=1e-3)
    lr_scheduler = get_lr_scheduler(
        optimizer,
        total_tokens=1_000_000,
        batch_size=2,
        seq_len=64,
        warmup_steps=10,
        max_lr=1e-3,
    )

    # Create dummy data
    train_data = TensorDataset(
        torch.randint(0, config.vocab_size, (100, 64)),
        torch.randint(0, config.vocab_size, (100, 64)),
    )
    val_data = TensorDataset(
        torch.randint(0, config.vocab_size, (20, 64)),
        torch.randint(0, config.vocab_size, (20, 64)),
    )

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2)

    # Wrap datasets to match expected format
    def collate_fn(batch):
        return {"input_ids": batch[0], "labels": batch[1]}

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=lambda x: {"input_ids": torch.stack([b[0] for b in x]), "labels": torch.stack([b[1] for b in x])})
    val_loader = DataLoader(val_data, batch_size=2, collate_fn=lambda x: {"input_ids": torch.stack([b[0] for b in x]), "labels": torch.stack([b[1] for b in x])})

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config={"test": True},
        device="cpu",
        use_amp=False,
        gradient_accumulation_steps=2,
        checkpoint_dir="/tmp/test_checkpoints",
        log_interval=5,
        eval_interval=20,
        save_interval=50,
        use_wandb=False,
    )

    # Train for a few steps
    print("\nTraining for 30 steps...")
    trainer.train(max_steps=30)

    # Test checkpoint loading
    print("\nTesting checkpoint loading...")
    checkpoint_path = "/tmp/test_checkpoints/checkpoint_final.pt"
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
        print("  ✓ Checkpoint loaded successfully")
    else:
        print(f"  Warning: Checkpoint not found at {checkpoint_path}")

    print("\n✓ Trainer tests passed!")
