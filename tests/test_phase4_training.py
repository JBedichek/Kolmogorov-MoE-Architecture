"""
Comprehensive test suite for Phase 4: Training Infrastructure.

Tests all Phase 4 features:
1. Tokenizer wrapper
2. Dolma dataset loading
3. Learning rate scheduling
4. Muon optimizer
5. Training loop with checkpointing
6. Full training pipeline
"""

import torch
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.model.transformer import MoETransformer
from moe_arch.data.tokenizer import TokenizerWrapper
from moe_arch.data.dolma_dataset import DolmaStreamingDataset, get_dataloaders
from moe_arch.training.muon_optimizer import Muon, get_muon_optimizer
from moe_arch.training.lr_schedule import CosineScheduleWithWarmup, get_lr_scheduler
from moe_arch.training.trainer import Trainer
from torch.utils.data import DataLoader


def test_tokenizer():
    """Test 1: Tokenizer wrapper."""
    print("=" * 70)
    print("TEST 1: Tokenizer Wrapper")
    print("=" * 70)

    tokenizer = TokenizerWrapper("gpt2")

    # Test encoding
    text = "Hello, world! This is a test."
    encoded = tokenizer.encode(text, return_tensors="pt")

    assert "input_ids" in encoded
    assert encoded["input_ids"].dim() == 2
    print(f"  ✓ Encoding works (shape: {encoded['input_ids'].shape})")

    # Test decoding
    decoded = tokenizer.decode(encoded["input_ids"][0])
    assert isinstance(decoded, str)
    print(f"  ✓ Decoding works")

    # Test batch encoding
    texts = ["First text.", "Second text with more tokens."]
    batch = tokenizer.encode(texts, padding=True, return_tensors="pt")
    assert batch["input_ids"].shape[0] == 2
    print(f"  ✓ Batch encoding works (shape: {batch['input_ids'].shape})")


def test_dataset():
    """Test 2: Dolma dataset."""
    print("\n" + "=" * 70)
    print("TEST 2: Dolma Streaming Dataset")
    print("=" * 70)

    tokenizer = TokenizerWrapper("gpt2")
    dataset = DolmaStreamingDataset(
        tokenizer=tokenizer,
        seq_len=128,
        n_pred_tokens=4,
        vocab_size=1000,  # Use small vocab for testing
    )

    # Test iteration
    samples = []
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        samples.append(sample)
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (128,)

    print(f"  ✓ Dataset iteration works")
    print(f"  ✓ Sample shape: {samples[0]['input_ids'].shape}")

    # Test dataloader
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 128)
    print(f"  ✓ Dataloader works (batch shape: {batch['input_ids'].shape})")


def test_lr_scheduler():
    """Test 3: Learning rate scheduler."""
    print("\n" + "=" * 70)
    print("TEST 3: Learning Rate Scheduler")
    print("=" * 70)

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    scheduler = CosineScheduleWithWarmup(
        optimizer=optimizer,
        warmup_steps=10,
        max_steps=100,
        max_lr=1e-3,
        min_lr=1e-5,
    )

    # Test warmup
    lrs = []
    for i in range(15):
        lr = scheduler.step()
        lrs.append(lr)

    # First step gives LR for step 1 (after warmup starts)
    assert lrs[0] > 0.0, "Should start warming up"
    assert lrs[0] < 1e-3, "Should be less than max_lr"
    assert abs(lrs[9] - 1e-3) < 1e-6, "Should reach max_lr around step 10"
    assert lrs[10] < lrs[9], "Should start decaying after warmup"
    print(f"  ✓ Warmup works ({lrs[0]:.2e} -> {lrs[9]:.2e})")
    print(f"  ✓ Decay starts after warmup")

    # Test state dict
    state = scheduler.state_dict()
    assert "current_step" in state
    print(f"  ✓ State dict works (step={state['current_step']})")


def test_muon_optimizer():
    """Test 4: Muon optimizer."""
    print("\n" + "=" * 70)
    print("TEST 4: Muon Optimizer")
    print("=" * 70)

    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )

    optimizer = get_muon_optimizer(model, lr=1e-3)

    # Test optimization step
    x = torch.randn(8, 64)
    target = torch.randint(0, 10, (8,))

    loss1 = torch.nn.functional.cross_entropy(model(x), target)
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss2 = torch.nn.functional.cross_entropy(model(x), target)

    assert loss1.item() != loss2.item(), "Parameters should update"
    print(f"  ✓ Optimizer step works")
    print(f"  ✓ Loss: {loss1.item():.4f} -> {loss2.item():.4f}")

    # Test orthogonalization
    matrix = torch.randn(32, 64)
    ortho = Muon._newton_schulz_orthogonalize(matrix, num_iters=10)
    product = ortho @ ortho.T
    identity = torch.eye(32)
    error = (product - identity).abs().max().item()
    assert error < 0.1, f"Orthogonalization error too large: {error}"
    print(f"  ✓ Orthogonalization works (error={error:.6f})")


def test_trainer():
    """Test 5: Training loop."""
    print("\n" + "=" * 70)
    print("TEST 5: Training Loop")
    print("=" * 70)

    # Create small model
    config = AdvancedMoEConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,  # d_model / n_heads = 128 / 4
        d_ff=256,
        d_ff_expert=128,
        max_seq_len=64,
        n_experts=2,
        moe_layers=(1,),
        mamba_layers=tuple(),
        mamba_enabled=False,
        mod_enabled=False,
        n_pred_tokens=2,
        aux_loss_weights=(1.0, 0.5),
        use_flash_attention=False,
    )

    model = MoETransformer(config)
    optimizer = get_muon_optimizer(model, lr=1e-3)
    lr_scheduler = get_lr_scheduler(
        optimizer,
        total_tokens=100000,
        batch_size=2,
        seq_len=64,
        warmup_steps=5,
        max_lr=1e-3,
    )

    # Create dataset
    tokenizer = TokenizerWrapper("gpt2")
    dataset = DolmaStreamingDataset(tokenizer, seq_len=64, n_pred_tokens=2, vocab_size=1000)
    loader = DataLoader(dataset, batch_size=2, num_workers=0)

    # Create trainer
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=loader,
            val_loader=loader,
            config={"test": True},
            device="cpu",
            use_amp=False,  # Disable AMP on CPU
            gradient_accumulation_steps=2,
            checkpoint_dir=tmpdir,
            log_interval=5,
            eval_interval=20,
            save_interval=50,
            use_wandb=False,
        )

        print(f"  Training for 10 steps...")
        trainer.train(max_steps=10)

        assert trainer.global_step == 10
        print(f"  ✓ Training completed {trainer.global_step} steps")
        print(f"  ✓ Tokens seen: {trainer.tokens_seen:,}")

        # Test checkpoint saving/loading
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
        trainer.save_checkpoint("test_checkpoint.pt")
        assert os.path.exists(checkpoint_path)
        print(f"  ✓ Checkpoint saved")

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        assert trainer.global_step == 10
        print(f"  ✓ Checkpoint loaded (step={trainer.global_step})")


def test_full_pipeline():
    """Test 6: Full training pipeline."""
    print("\n" + "=" * 70)
    print("TEST 6: Full Training Pipeline")
    print("=" * 70)

    config = AdvancedMoEConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,  # d_model / n_heads = 128 / 4
        d_ff=256,
        d_ff_expert=128,
        max_seq_len=64,
        n_experts=2,
        moe_layers=(1,),
        mamba_layers=tuple(),
        mamba_enabled=False,
        mod_enabled=False,
        n_pred_tokens=2,
        aux_loss_weights=(1.0, 0.5),
        use_flash_attention=False,
    )

    print(f"  Model: {config.n_layers} layers, {config.d_model} dim")

    # Initialize model
    model = MoETransformer(config)
    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,}")

    # Initialize components
    tokenizer = TokenizerWrapper("gpt2")
    train_loader, val_loader = get_dataloaders(
        tokenizer=tokenizer,
        config=config,
        batch_size=2,
        num_workers=0,
    )

    optimizer = get_muon_optimizer(model, lr=1e-3)
    lr_scheduler = get_lr_scheduler(
        optimizer,
        total_tokens=10000,
        batch_size=2,
        seq_len=64,
        warmup_steps=5,
        max_lr=1e-3,
    )

    print(f"  ✓ All components initialized")

    # Quick training test
    model.train()
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"  ✓ Forward/backward pass works")
    print(f"  ✓ Loss: {loss.item():.4f}")

    # Evaluation test
    model.eval()
    with torch.no_grad():
        eval_batch = next(iter(val_loader))
        eval_outputs = model(eval_batch["input_ids"], labels=eval_batch["labels"])
        eval_loss = eval_outputs["loss"]

    print(f"  ✓ Evaluation works (loss={eval_loss.item():.4f})")


def main():
    """Run all Phase 4 tests."""
    print("\n" + "=" * 70)
    print("PHASE 4: TRAINING INFRASTRUCTURE TEST SUITE")
    print("=" * 70)

    test_tokenizer()
    test_dataset()
    test_lr_scheduler()
    test_muon_optimizer()
    test_trainer()
    test_full_pipeline()

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 4 TEST SUMMARY")
    print("=" * 70)
    print("✓ All Phase 4 tests passed!")
    print("\nPhase 4 complete:")
    print("  ✓ Tokenizer wrapper (HuggingFace)")
    print("  ✓ Dolma dataset loader with streaming")
    print("  ✓ Learning rate scheduler (cosine + warmup)")
    print("  ✓ Muon optimizer with momentum orthogonalization")
    print("  ✓ Training loop with gradient accumulation")
    print("  ✓ Checkpointing and resumption")
    print("  ✓ Full training pipeline")
    print("\nReady for full-scale training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
