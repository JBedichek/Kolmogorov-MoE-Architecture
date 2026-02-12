#!/usr/bin/env python3
"""
Quick training test to check if loss decreases.
Runs for configurable steps and prints loss every 50 steps.
Uses real text data from FineWeb.

Features:
- Tokens/sec tracking
- torch.compile support
- WandB logging (optional)
- Epoch-based loss tracking
"""

import torch
import torch.nn.functional as F
import yaml
import argparse
import gc
import time
import os
import signal
import sys
import atexit
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
from datasets import load_dataset
from transformers import AutoTokenizer

# Optional WandB
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# Global training state for signal handler access
_training_state = {
    'model': None,
    'optimizer': None,
    'step': 0,
    'epoch': 0,
    'data_idx': 0,
    'losses': [],
    'total_tokens': 0,
    'config_path': None,
    'checkpoint_dir': None,
    'should_stop': False,
    'checkpoint_saved': False,
    'args': None,
}


def get_model_state_dict(model):
    """Get state dict, handling compiled models."""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    return model.state_dict()


def save_checkpoint(state, path, reason="scheduled"):
    """Save training checkpoint."""
    if state['model'] is None:
        print(f"\n[Checkpoint] No model to save")
        return False

    checkpoint = {
        'step': state['step'],
        'epoch': state['epoch'],
        'data_idx': state['data_idx'],
        'losses': state['losses'][-1000:],  # Keep last 1000 losses
        'total_tokens': state['total_tokens'],
        'config_path': state['config_path'],
        'model_state_dict': get_model_state_dict(state['model']),
        'optimizer_state_dict': state['optimizer'].state_dict() if state['optimizer'] else None,
    }

    torch.save(checkpoint, path)
    loss_str = f"{state['losses'][-1]:.4f}" if state['losses'] else 'N/A'
    print(f"\n[Checkpoint] Saved ({reason}): {path}")
    print(f"  Step: {state['step']}, Epoch: {state['epoch']}, Loss: {loss_str}")
    state['checkpoint_saved'] = True
    return True


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    sig_name = signal.Signals(signum).name
    print(f"\n\n[Signal] Received {sig_name}, saving checkpoint and exiting...")
    _training_state['should_stop'] = True

    # Save checkpoint
    if not _training_state['checkpoint_saved'] and _training_state['model'] is not None:
        ckpt_path = os.path.join(
            _training_state['checkpoint_dir'],
            f"checkpoint-{_training_state['step']}-interrupted.pt"
        )
        save_checkpoint(_training_state, ckpt_path, reason=f"interrupted by {sig_name}")

    sys.exit(0)


def get_dataloader(tokenizer, seq_len, batch_size, max_examples=10000):
    """Load real text data from FineWeb."""
    print("Loading FineWeb dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=False,
    )
    dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Tokenize
    print(f"Tokenizing {len(dataset)} examples...")

    # Process in batches
    all_input_ids = []
    for i in range(0, len(dataset), 100):
        batch = dataset[i:i+100]
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        all_input_ids.extend(tokens["input_ids"])

    print(f"Created {len(all_input_ids)} training examples")
    return all_input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug_no_mod.yaml')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb-project', type=str, default='Kolmogorov-Testing', help='WandB project name')
    parser.add_argument('--wandb-run', type=str, default=None, help='WandB run name')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--grad-checkpoint', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--max-examples', type=int, default=10000, help='Max training examples to load')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_quick', help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=500, help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup signal handlers for graceful shutdown
    _training_state['checkpoint_dir'] = args.checkpoint_dir
    _training_state['config_path'] = args.config
    _training_state['args'] = args
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 70)
    print("QUICK TRAINING TEST")
    print("=" * 70)

    # Load config
    config_dict = load_config(args.config)
    model_cfg = config_dict['model']
    train_cfg = config_dict.get('training', {})

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.grad_accum
    tokens_per_step = effective_batch_size * args.seq_len

    print(f"\nConfig: {args.config}")
    print(f"MoD enabled: {model_cfg.get('mod_enabled', True)}")
    print(f"MoD capacity: {model_cfg.get('mod_capacity_factor', 0.75)}")
    print(f"n_experts: {model_cfg.get('n_experts')}")
    print(f"n_layers: {model_cfg.get('n_layers')}")
    print(f"d_model: {model_cfg.get('d_model')}")
    print(f"\nTraining settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Tokens per optimizer step: {tokens_per_step:,}")
    print(f"  torch.compile: {args.compile}")
    print(f"  Gradient checkpointing: {args.grad_checkpoint}")
    print(f"  Learning rate: {train_cfg.get('max_lr', 0.0003)}")

    # Initialize WandB
    if args.wandb:
        if not HAS_WANDB:
            print("\nWandB not installed. Install with: pip install wandb")
            args.wandb = False
        else:
            run_name = args.wandb_run or f"quick-train-{model_cfg.get('n_layers')}L-{model_cfg.get('n_experts')}E"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": model_cfg,
                    "training": {
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "grad_accum": args.grad_accum,
                        "effective_batch_size": effective_batch_size,
                        "tokens_per_step": tokens_per_step,
                        "compile": args.compile,
                        "grad_checkpoint": args.grad_checkpoint,
                        "lr": train_cfg.get('max_lr', 0.0003),
                    },
                },
            )
            print(f"\nWandB initialized: {args.wandb_project}/{run_name}")

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Performance settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
        max_seq_len=args.seq_len,
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
    print("\nCreating model...")
    model = MoETransformer(config).cuda().to(torch.bfloat16)
    model.train()

    # Count parameters
    params = model.count_parameters()
    print(f"  Total params: {params['total_billions']:.3f}B")
    print(f"  Active params: {params['active_billions']:.3f}B ({params['sparsity']:.1%} sparsity)")

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # Apply torch.compile
    if args.compile:
        print("\nCompiling model with torch.compile...")
        compile_start = time.time()
        model = torch.compile(model, mode="default")
        print(f"  Compilation setup done in {time.time() - compile_start:.1f}s")
        print("  (Full compilation happens on first forward pass)")

    # Load tokenizer and data
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token

    # Load real data
    all_input_ids = get_dataloader(tokenizer, args.seq_len, args.batch_size, max_examples=args.max_examples)
    num_examples = len(all_input_ids)
    examples_per_epoch = num_examples // args.batch_size

    print(f"\nDataset:")
    print(f"  Total examples: {num_examples:,}")
    print(f"  Examples per epoch: {examples_per_epoch:,}")
    print(f"  Optimizer steps per epoch: {examples_per_epoch // args.grad_accum:,}")

    # Create optimizer
    lr = train_cfg.get('max_lr', 0.0003)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Training state
    losses = []
    epoch_losses = []
    current_epoch_losses = []
    data_idx = 0
    micro_step = 0
    current_epoch = 0
    start_step = 0
    total_tokens = 0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cuda', weights_only=False)

            # Load model state
            if hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            if checkpoint.get('optimizer_state_dict'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore training state
            start_step = checkpoint.get('step', 0)
            current_epoch = checkpoint.get('epoch', 0)
            data_idx = checkpoint.get('data_idx', 0)
            losses = checkpoint.get('losses', [])
            total_tokens = checkpoint.get('total_tokens', 0)

            print(f"  Resumed at step {start_step}, epoch {current_epoch}")
            print(f"  Last loss: {losses[-1] if losses else 'N/A':.4f}")
        else:
            print(f"\n[Warning] Checkpoint not found: {args.resume}")

    # Update global training state for signal handler
    _training_state['model'] = model
    _training_state['optimizer'] = optimizer
    _training_state['losses'] = losses
    _training_state['data_idx'] = data_idx
    _training_state['epoch'] = current_epoch
    _training_state['total_tokens'] = total_tokens

    print(f"\nTraining for {args.steps} optimizer steps (starting from {start_step})...")
    if args.save_interval > 0:
        print(f"Checkpoints: every {args.save_interval} steps to {args.checkpoint_dir}")
    print("-" * 70)

    # Timing
    start_time = time.time()
    step_start_time = time.time()
    log_interval = 50

    # Warmup for torch.compile (first few steps are slow)
    if args.compile:
        print("Running warmup steps for torch.compile...")

    for opt_step in range(start_step, args.steps):
        # Check for graceful shutdown
        if _training_state['should_stop']:
            print("\n[Training] Stopping due to signal...")
            break
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_lm_loss = 0.0
        step_aux_loss = 0.0

        # Gradient accumulation loop
        for accum_step in range(args.grad_accum):
            # Build batch
            batch_input_ids = []
            for _ in range(args.batch_size):
                batch_input_ids.append(all_input_ids[data_idx % num_examples])
                data_idx += 1

                # Check for epoch boundary
                if data_idx % num_examples == 0:
                    if len(current_epoch_losses) > 0:
                        epoch_avg = sum(current_epoch_losses) / len(current_epoch_losses)
                        epoch_losses.append(epoch_avg)
                        print(f"\n{'='*70}")
                        print(f"EPOCH {current_epoch} COMPLETE - Average Loss: {epoch_avg:.4f}")
                        if len(epoch_losses) > 1:
                            delta = epoch_avg - epoch_losses[-2]
                            print(f"  Change from previous epoch: {delta:+.4f}")
                        print(f"{'='*70}\n")

                        if args.wandb:
                            wandb.log({
                                "epoch": current_epoch,
                                "epoch/avg_loss": epoch_avg,
                            }, step=opt_step)

                        current_epoch_losses = []
                        current_epoch += 1

            input_ids = torch.tensor(batch_input_ids, device='cuda')
            labels = input_ids.clone()

            # Forward
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss'] / args.grad_accum
            lm_loss = outputs.get('lm_loss', outputs['loss'])
            aux_loss = outputs.get('aux_loss', 0)

            # Backward
            loss.backward()

            step_loss += loss.item() * args.grad_accum
            step_lm_loss += lm_loss.item() / args.grad_accum
            if isinstance(aux_loss, torch.Tensor):
                step_aux_loss += aux_loss.item() / args.grad_accum
            else:
                step_aux_loss += aux_loss / args.grad_accum

            micro_step += 1
            total_tokens += args.batch_size * args.seq_len

            del outputs, loss, input_ids, labels

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()

        losses.append(step_loss)
        current_epoch_losses.append(step_loss)

        # Update global training state
        _training_state['step'] = opt_step + 1
        _training_state['epoch'] = current_epoch
        _training_state['data_idx'] = data_idx
        _training_state['losses'] = losses
        _training_state['total_tokens'] = total_tokens
        _training_state['checkpoint_saved'] = False  # Reset for next interval

        # Save checkpoint at intervals
        if args.save_interval > 0 and (opt_step + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{opt_step + 1}.pt")
            save_checkpoint(_training_state, ckpt_path, reason="interval")

        # Calculate metrics
        step_time = time.time() - step_start_time
        tokens_per_sec = tokens_per_step / step_time if step_time > 0 else 0
        elapsed = time.time() - start_time

        # Log to WandB
        if args.wandb:
            wandb.log({
                "step": opt_step,
                "loss": step_loss,
                "lm_loss": step_lm_loss,
                "aux_loss": step_aux_loss,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "tokens_per_sec": tokens_per_sec,
                "total_tokens": total_tokens,
                "lr": lr,
                "memory_gb": torch.cuda.memory_allocated() / 1e9,
            }, step=opt_step)

        # Print progress
        if (opt_step + 1) % log_interval == 0 or opt_step == 0:
            avg_loss = sum(losses[-log_interval:]) / len(losses[-log_interval:])
            mem = torch.cuda.memory_allocated() / 1e9

            print(f"Step {opt_step+1:5d} | "
                  f"loss={step_loss:.4f} avg{log_interval}={avg_loss:.4f} | "
                  f"lm={step_lm_loss:.4f} aux={step_aux_loss:.4f} | "
                  f"gnorm={grad_norm:.2f} | "
                  f"{tokens_per_sec:,.0f} tok/s | "
                  f"mem={mem:.1f}GB | "
                  f"epoch={current_epoch}")

        step_start_time = time.time()

    # Training complete
    total_time = time.time() - start_time
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    # Save final checkpoint
    final_step = opt_step + 1 if 'opt_step' in dir() else start_step
    final_ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{final_step}-final.pt")
    _training_state['step'] = final_step
    save_checkpoint(_training_state, final_ckpt_path, reason="training complete")

    print("-" * 70)
    print("\nTRAINING COMPLETE")
    print("=" * 70)

    # Final epoch stats
    if len(current_epoch_losses) > 0:
        final_epoch_avg = sum(current_epoch_losses) / len(current_epoch_losses)
        epoch_losses.append(final_epoch_avg)
        print(f"\nFinal partial epoch {current_epoch} - Average Loss: {final_epoch_avg:.4f}")

    # Summary statistics
    print(f"\nResults:")
    if losses:
        print(f"  Initial loss (step 1):  {losses[0]:.4f}")
        print(f"  Final loss (step {final_step}): {losses[-1]:.4f}")
        print(f"  Change: {losses[-1] - losses[0]:+.4f}")
    else:
        print("  No training steps completed")

    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average throughput: {avg_tokens_per_sec:,.0f} tokens/sec")
    print(f"  Steps completed: {args.steps}")
    print(f"  Epochs completed: {current_epoch} + partial")

    # Epoch loss progression
    if len(epoch_losses) > 0:
        print(f"\nEpoch Loss Progression:")
        for i, eloss in enumerate(epoch_losses):
            delta = ""
            if i > 0:
                d = eloss - epoch_losses[i-1]
                delta = f" ({d:+.4f})"
            status = "improving" if i > 0 and eloss < epoch_losses[i-1] else ""
            print(f"  Epoch {i}: {eloss:.4f}{delta} {status}")

    # Learning assessment
    if len(losses) >= 2:
        if losses[-1] < losses[0] - 0.5:
            print("\n[OK] Loss is DECREASING - model is learning!")
        elif losses[-1] < losses[0] - 0.1:
            print("\n[WARN] Loss decreased slightly - learning slowly")
        else:
            print("\n[ERROR] Loss NOT decreasing - something is wrong!")
            print("   Possible issues:")
            print("   - Learning rate too low or too high")
            print("   - MoD blocking gradient flow")
            print("   - Router collapse")
            print("   - Architecture issue")

        # Show loss curve
        print("\nLoss curve (every 100 steps):")
        for i in range(0, len(losses), 100):
            bar_len = int((12 - losses[i]) * 3)  # Scale for display
            bar = "#" * max(0, bar_len)
            print(f"  {i+1:5d}: {losses[i]:.2f} {bar}")

    # Print checkpoint info
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"  Resume with: python quick_train_test.py --resume {final_ckpt_path} --config {args.config}")

    # Close WandB
    if args.wandb:
        wandb.log({
            "final/initial_loss": losses[0] if losses else 0,
            "final/final_loss": losses[-1] if losses else 0,
            "final/loss_change": (losses[-1] - losses[0]) if len(losses) > 1 else 0,
            "final/total_time_s": total_time,
            "final/avg_tokens_per_sec": avg_tokens_per_sec,
            "final/epochs_completed": current_epoch,
        })
        wandb.finish()
        print(f"\nWandB run completed: {args.wandb_project}")


if __name__ == '__main__':
    main()
