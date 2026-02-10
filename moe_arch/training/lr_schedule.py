"""
Learning rate scheduling.

Implements cosine decay with linear warmup, standard for LLM training.
"""

import math
from typing import Optional


class CosineScheduleWithWarmup:
    """
    Cosine learning rate schedule with linear warmup.

    Schedule:
    1. Linear warmup from 0 to max_lr over warmup_steps
    2. Cosine decay from max_lr to min_lr over remaining steps
    3. Optional constant min_lr after decay

    This is the standard LR schedule used in GPT-3, LLaMA, etc.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
        warmup_init_lr: float = 0.0,
    ):
        """
        Initialize LR scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            max_lr: Maximum learning rate (reached after warmup)
            min_lr: Minimum learning rate (final LR after decay)
            warmup_init_lr: Initial LR at step 0 (usually 0)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_init_lr = warmup_init_lr

        self.current_step = 0

        print(f"Initialized CosineScheduleWithWarmup:")
        print(f"  Warmup steps: {warmup_steps:,}")
        print(f"  Max steps: {max_steps:,}")
        print(f"  Max LR: {max_lr:.2e}")
        print(f"  Min LR: {min_lr:.2e}")
        print(f"  Warmup init LR: {warmup_init_lr:.2e}")

    def get_lr(self, step: Optional[int] = None) -> float:
        """
        Get learning rate for current step.

        Args:
            step: Step number (uses internal counter if None)

        Returns:
            Learning rate for this step
        """
        if step is None:
            step = self.current_step

        if step < self.warmup_steps:
            # Linear warmup
            lr = self.warmup_init_lr + (self.max_lr - self.warmup_init_lr) * (
                step / self.warmup_steps
            )
        elif step < self.max_steps:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        else:
            # After max_steps, use min_lr
            lr = self.min_lr

        return lr

    def step(self):
        """
        Update learning rate for next step.

        Call this after each optimizer.step().
        """
        self.current_step += 1
        lr = self.get_lr()

        # Apply scheduled LR to ALL param groups
        # (No ratio preservation - all groups get same scheduled LR)
        self._last_lr = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            self._last_lr.append(lr)

        return lr

    def get_last_lr(self):
        """
        Return last computed learning rates (required by HuggingFace Trainer).

        Returns:
            List of learning rates for each parameter group
        """
        if not hasattr(self, '_last_lr'):
            # If step() hasn't been called yet, return LR for step 1 (the first actual step)
            # NOT step 0 (which would be warmup_init_lr = 0.0)
            current_lr = self.get_lr(1)
            return [current_lr] * len(self.optimizer.param_groups)
        return self._last_lr

    def state_dict(self):
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'warmup_init_lr': self.warmup_init_lr,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.max_steps = state_dict['max_steps']
        self.max_lr = state_dict['max_lr']
        self.min_lr = state_dict['min_lr']
        self.warmup_init_lr = state_dict['warmup_init_lr']


def get_lr_scheduler(
    optimizer,
    total_tokens: int,
    batch_size: int,
    seq_len: int,
    warmup_steps: int = 2000,
    max_lr: float = 3e-4,
    min_lr_ratio: float = 0.1,
):
    """
    Create LR scheduler with automatic step calculation.

    Args:
        optimizer: PyTorch optimizer
        total_tokens: Total training tokens
        batch_size: Batch size
        seq_len: Sequence length
        warmup_steps: Warmup steps (default: 2000)
        max_lr: Maximum learning rate
        min_lr_ratio: Minimum LR as ratio of max_lr (default: 0.1)

    Returns:
        CosineScheduleWithWarmup scheduler
    """
    # Calculate total steps
    tokens_per_step = batch_size * seq_len
    max_steps = total_tokens // tokens_per_step

    min_lr = max_lr * min_lr_ratio

    print(f"\nLR scheduler configuration:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Total steps: {max_steps:,}")

    return CosineScheduleWithWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_init_lr=0.0,
    )


if __name__ == "__main__":
    # Test LR scheduler
    import torch
    import matplotlib.pyplot as plt

    print("Testing CosineScheduleWithWarmup...")

    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create scheduler
    warmup_steps = 100
    max_steps = 1000
    max_lr = 1e-3
    min_lr = 1e-5

    scheduler = CosineScheduleWithWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=min_lr,
    )

    # Simulate training
    lrs = []
    for step in range(max_steps + 100):
        lr = scheduler.step()
        lrs.append(lr)

        # Verify optimizer LR matches
        assert optimizer.param_groups[0]['lr'] == lr

    # Check key properties
    print("\nVerifying schedule properties...")

    # 1. Warmup: LR should increase linearly
    warmup_lrs = lrs[:warmup_steps]
    assert warmup_lrs[0] == 0.0, f"Initial LR should be 0, got {warmup_lrs[0]}"
    assert abs(warmup_lrs[-1] - max_lr) < 1e-6, f"End of warmup should reach max_lr"
    print(f"  ✓ Warmup: 0 -> {max_lr:.2e}")

    # 2. Cosine decay: LR should decrease smoothly
    decay_lrs = lrs[warmup_steps:max_steps]
    assert decay_lrs[0] <= max_lr, "Decay should start from max_lr"
    assert decay_lrs[-1] >= min_lr, "Decay should end at min_lr"
    print(f"  ✓ Decay: {max_lr:.2e} -> {min_lr:.2e}")

    # 3. After max_steps: LR should be constant at min_lr
    post_lrs = lrs[max_steps:]
    assert all(abs(lr - min_lr) < 1e-6 for lr in post_lrs), "LR should be constant after max_steps"
    print(f"  ✓ Post-training: constant at {min_lr:.2e}")

    # Test state dict
    print("\nTesting state dict...")
    state = scheduler.state_dict()
    assert state['current_step'] == max_steps + 100
    print(f"  ✓ State dict works (step={state['current_step']})")

    # Plot schedule
    print("\nPlotting LR schedule...")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(lrs)
        plt.axvline(warmup_steps, color='r', linestyle='--', label='End of warmup')
        plt.axvline(max_steps, color='g', linestyle='--', label='Max steps')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Cosine Schedule with Warmup')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig('/tmp/lr_schedule.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Plot saved to /tmp/lr_schedule.png")
    except Exception as e:
        print(f"  Could not create plot: {e}")

    print("\n✓ All LR scheduler tests passed!")
