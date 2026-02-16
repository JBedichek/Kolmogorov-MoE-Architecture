#!/usr/bin/env python3
"""
Custom high-performance training script for MoE models.

Features:
- FSDP distributed training with optimal sharding
- Sequence packing for maximum GPU utilization
- 8-bit Muon optimizer with custom implementation
- Gradient checkpointing for memory efficiency
- Efficient data loading with prefetching
- WandB integration
- Configurable evaluation suite
- Token-level loss weighting
- Checkpoint save/resume

Usage:
    # Single GPU
    python train_custom.py --config configs/production_training.yaml

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=2 train_custom.py --config configs/production_training.yaml
"""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# Local imports
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.training.muon_optimizer import Muon8bit


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Training
    total_tokens: int = 1_000_000_000
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 4096

    # Optimizer
    max_lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    use_8bit_optimizer: bool = True

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "sample-10BT"
    tokenizer_name: str = "gpt2"
    max_examples: Optional[int] = None
    num_workers: int = 4
    use_sequence_packing: bool = True
    use_document_attention_mask: bool = False  # Disable to use Flash Attention (17x faster)

    # Checkpointing
    checkpoint_dir: str = "./checkpoints_custom"
    save_interval: int = 1000
    resume_from: Optional[str] = None

    # Logging
    log_interval: int = 10
    wandb_enabled: bool = False
    wandb_project: str = "moe-training"
    wandb_run_name: Optional[str] = None

    # Evaluation
    eval_enabled: bool = False
    eval_interval: int = 500
    eval_datasets: List[str] = field(default_factory=list)

    # Loss weighting
    use_token_weights: bool = False
    token_weight_decay: float = 0.0  # 0 = uniform, >0 = later tokens weighted more

    # Performance
    use_gradient_checkpointing: bool = True
    compile_model: bool = False
    profile_timing: bool = False  # Enable detailed timing (adds sync overhead)

    # Device
    gpu: Optional[int] = None  # GPU index to use (None = auto)

    # Mixed precision
    dtype: str = "bfloat16"  # bfloat16, float16, float32

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()
        config.model_config = data.get('model', {})

        train = data.get('training', {})
        config.total_tokens = train.get('total_tokens', config.total_tokens)
        config.batch_size = train.get('batch_size', config.batch_size)
        config.gradient_accumulation_steps = train.get('gradient_accumulation_steps', config.gradient_accumulation_steps)
        config.max_lr = train.get('max_lr', config.max_lr)
        config.min_lr_ratio = train.get('min_lr_ratio', config.min_lr_ratio)
        config.warmup_steps = train.get('warmup_steps', config.warmup_steps)
        config.weight_decay = train.get('weight_decay', config.weight_decay)
        config.grad_clip_norm = train.get('gradient_clip_norm', config.grad_clip_norm)
        config.checkpoint_dir = train.get('checkpoint_dir', config.checkpoint_dir)
        config.save_interval = train.get('save_interval', config.save_interval)
        config.log_interval = train.get('log_interval', config.log_interval)
        config.resume_from = train.get('resume_from', config.resume_from)

        data_cfg = data.get('data', {})
        config.dataset_name = data_cfg.get('dataset_name', config.dataset_name)
        config.dataset_config = data_cfg.get('dataset_config', config.dataset_config)
        config.tokenizer_name = data_cfg.get('tokenizer_name', config.tokenizer_name)
        config.max_examples = data_cfg.get('max_examples', config.max_examples)
        config.num_workers = data_cfg.get('num_workers', config.num_workers)
        config.max_seq_len = data.get('model', {}).get('max_seq_len', config.max_seq_len)
        config.use_document_attention_mask = data_cfg.get('use_document_attention_mask', config.use_document_attention_mask)

        wandb_cfg = data.get('wandb', {})
        config.wandb_enabled = wandb_cfg.get('enabled', config.wandb_enabled)
        config.wandb_project = wandb_cfg.get('project', config.wandb_project)
        config.wandb_run_name = wandb_cfg.get('run_name', config.wandb_run_name)

        eval_cfg = data.get('evaluation', {})
        config.eval_enabled = eval_cfg.get('enabled', config.eval_enabled)
        config.eval_interval = eval_cfg.get('eval_interval', config.eval_interval)
        config.eval_datasets = eval_cfg.get('datasets', config.eval_datasets)

        mp_cfg = data.get('mixed_precision', {})
        if mp_cfg.get('bf16', False):
            config.dtype = 'bfloat16'
        elif mp_cfg.get('fp16', False):
            config.dtype = 'float16'

        return config


# =============================================================================
# Distributed Utilities
# =============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


# =============================================================================
# Sequence Packing Dataset
# =============================================================================

class PackedSequenceDataset(IterableDataset):
    """
    Dataset that packs multiple sequences into fixed-length chunks.

    This maximizes GPU utilization by eliminating padding waste.
    Each batch contains densely packed tokens with separator tokens between documents.

    IMPORTANT: Creates document_ids to track document boundaries for proper attention masking.
    Without this, documents can attend to each other (cross-contamination).
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        tokenizer,
        max_seq_len: int,
        max_examples: Optional[int] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_examples = max_examples
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.eos_token_id = tokenizer.eos_token_id

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield packed sequences with document boundaries."""
        from datasets import load_dataset

        # Load streaming dataset
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split='train',
            streaming=True,
        )

        # Shard for distributed training
        if self.world_size > 1:
            dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

        # Shuffle with seed
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)

        # Token buffer for packing (stores (token_id, doc_id) pairs)
        token_buffer = []
        doc_id_buffer = []
        current_doc_id = 0
        examples_seen = 0

        for example in dataset:
            if self.max_examples and examples_seen >= self.max_examples:
                break

            text = example.get('text', '')
            if not text.strip():
                continue

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Add EOS and append to buffer with doc_id tracking
            tokens.append(self.eos_token_id)
            token_buffer.extend(tokens)
            doc_id_buffer.extend([current_doc_id] * len(tokens))

            current_doc_id += 1
            examples_seen += 1

            # Yield packed sequences when buffer is full
            while len(token_buffer) >= self.max_seq_len:
                packed_tokens = token_buffer[:self.max_seq_len]
                packed_doc_ids = doc_id_buffer[:self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len:]
                doc_id_buffer = doc_id_buffer[self.max_seq_len:]

                input_ids = torch.tensor(packed_tokens, dtype=torch.long)
                labels = input_ids.clone()
                doc_ids = torch.tensor(packed_doc_ids, dtype=torch.long)

                # Reset labels at document boundaries (don't predict across docs)
                # Find positions where next token is from a different document
                doc_boundaries = (doc_ids[:-1] != doc_ids[1:])
                labels[:-1][doc_boundaries] = -100  # Don't predict first token of new doc

                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                    'doc_ids': doc_ids,  # For attention mask creation
                }

        # Yield remaining tokens if any (padded)
        if len(token_buffer) > 0:
            pad_len = self.max_seq_len - len(token_buffer)
            packed_tokens = token_buffer + [self.tokenizer.pad_token_id] * pad_len
            packed_doc_ids = doc_id_buffer + [-1] * pad_len  # -1 for padding

            input_ids = torch.tensor(packed_tokens, dtype=torch.long)
            labels = input_ids.clone()
            labels[-pad_len:] = -100  # Ignore padding in loss
            doc_ids = torch.tensor(packed_doc_ids, dtype=torch.long)

            # Reset labels at document boundaries
            valid_len = self.max_seq_len - pad_len
            if valid_len > 1:
                doc_boundaries = (doc_ids[:valid_len-1] != doc_ids[1:valid_len])
                labels[:valid_len-1][doc_boundaries] = -100

            yield {
                'input_ids': input_ids,
                'labels': labels,
                'doc_ids': doc_ids,
            }


class StandardDataset(IterableDataset):
    """Standard dataset without packing (for comparison)."""

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        tokenizer,
        max_seq_len: int,
        max_examples: Optional[int] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_examples = max_examples
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        from datasets import load_dataset

        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split='train',
            streaming=True,
        )

        if self.world_size > 1:
            dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)

        examples_seen = 0
        for example in dataset:
            if self.max_examples and examples_seen >= self.max_examples:
                break

            text = example.get('text', '')
            if not text.strip():
                continue

            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                padding='max_length',
                return_tensors='pt',
            )

            input_ids = tokens['input_ids'].squeeze(0)
            labels = input_ids.clone()

            # Mask padding tokens in labels
            labels[input_ids == self.tokenizer.pad_token_id] = -100

            examples_seen += 1
            yield {
                'input_ids': input_ids,
                'labels': labels,
            }


def create_dataloader(
    config: TrainConfig,
    tokenizer,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create optimized dataloader."""

    DatasetClass = PackedSequenceDataset if config.use_sequence_packing else StandardDataset

    dataset = DatasetClass(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        max_examples=config.max_examples,
        rank=rank,
        world_size=world_size,
    )

    # Capture config flag for closure
    use_doc_mask = config.use_document_attention_mask

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x['input_ids'] for x in batch])
        labels = torch.stack([x['labels'] for x in batch])

        result = {'input_ids': input_ids, 'labels': labels}

        # Create document-aware attention mask if enabled and doc_ids present
        # WARNING: This disables Flash Attention and is 17x slower!
        if use_doc_mask and 'doc_ids' in batch[0]:
            doc_ids = torch.stack([x['doc_ids'] for x in batch])
            # Create block-diagonal attention mask
            # Each position can only attend to positions with same doc_id (and earlier)
            # Shape: (batch, 1, seq_len, seq_len) for broadcasting
            batch_size, seq_len = doc_ids.shape

            # doc_ids: (batch, seq_len) -> compare each position pair
            # mask[i,j] = 1 if doc_ids[i] == doc_ids[j] (same document)
            doc_ids_row = doc_ids.unsqueeze(2)  # (batch, seq_len, 1)
            doc_ids_col = doc_ids.unsqueeze(1)  # (batch, 1, seq_len)
            same_doc = (doc_ids_row == doc_ids_col)  # (batch, seq_len, seq_len)

            # Combine with causal mask: can only attend to same doc AND earlier positions
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            attention_mask = same_doc & causal.to(same_doc.device)

            # Convert to additive mask (0 for attend, -inf for don't attend)
            attention_mask = attention_mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            attention_mask = torch.where(
                attention_mask,
                torch.zeros_like(attention_mask, dtype=torch.float),
                torch.full_like(attention_mask, float('-inf'), dtype=torch.float),
            )
            result['attention_mask'] = attention_mask

        return result

    # IterableDataset with streaming needs num_workers=0
    # For map-style datasets, we can use multiple workers
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,  # Threaded prefetcher handles async loading
        pin_memory=True,
        collate_fn=collate_fn,
    )


class CUDAPrefetcher:
    """
    Prefetches batches to GPU asynchronously.

    This overlaps data transfer with computation, hiding data loading latency.
    """

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.iter = None
        self.next_batch = None

    def __iter__(self):
        self.iter = iter(self.dataloader)
        self._preload()
        return self

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
            }

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        if self.next_batch is None:
            raise StopIteration

        batch = self.next_batch
        self._preload()
        return batch


class ThreadedPrefetcher:
    """
    Prefetches batches using a background thread.

    This is essential for streaming datasets where data loading is the bottleneck.
    The background thread continuously fetches batches while GPU processes current batch.
    """

    def __init__(self, dataloader, device, buffer_size: int = 3):
        import threading
        import queue

        self.dataloader = dataloader
        self.device = device
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.thread = None
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _producer(self):
        """Background thread that fetches and transfers batches."""
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break

                # Transfer to GPU asynchronously (don't sync here!)
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        gpu_batch = {
                            k: v.to(self.device, non_blocking=True)
                            for k, v in batch.items()
                        }
                    # Record event for consumer to wait on
                    event = torch.cuda.Event()
                    event.record(self.stream)
                    self.queue.put((gpu_batch, event))
                else:
                    gpu_batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.queue.put((gpu_batch, None))

        except Exception as e:
            self.queue.put(e)
        finally:
            self.queue.put(None)  # Signal end

    def __iter__(self):
        import threading

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        item = self.queue.get()

        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item

        # Unpack batch and event
        batch, event = item
        # Wait for GPU transfer to complete (only blocks if transfer not done)
        if event is not None:
            event.wait()

        return batch

    def stop(self):
        """Stop the background thread."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class TimingStats:
    """Track timing breakdown for profiling."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.data_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.optimizer_time = 0.0
        self.total_time = 0.0
        self.count = 0

    def update(self, data_t, fwd_t, bwd_t, opt_t, total_t):
        self.data_time += data_t
        self.forward_time += fwd_t
        self.backward_time += bwd_t
        self.optimizer_time += opt_t
        self.total_time += total_t
        self.count += 1

    def summary(self) -> str:
        if self.count == 0:
            return "No timing data"

        total = self.total_time / self.count * 1000  # ms
        data = self.data_time / self.count * 1000
        fwd = self.forward_time / self.count * 1000
        bwd = self.backward_time / self.count * 1000
        opt = self.optimizer_time / self.count * 1000

        # Calculate percentages
        data_pct = data / total * 100 if total > 0 else 0
        fwd_pct = fwd / total * 100 if total > 0 else 0
        bwd_pct = bwd / total * 100 if total > 0 else 0
        opt_pct = opt / total * 100 if total > 0 else 0

        return (
            f"data: {data:.1f}ms ({data_pct:.0f}%) | "
            f"fwd: {fwd:.1f}ms ({fwd_pct:.0f}%) | "
            f"bwd: {bwd:.1f}ms ({bwd_pct:.0f}%) | "
            f"opt: {opt:.1f}ms ({opt_pct:.0f}%)"
        )


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class CosineWarmupScheduler:
    """Cosine learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * self.current_step / self.warmup_steps
        elif self.current_step >= self.max_steps:
            return self.min_lr
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# Token-Level Loss Weighting
# =============================================================================

def compute_token_weights(seq_len: int, decay: float, device: torch.device) -> torch.Tensor:
    """
    Compute token-level loss weights.

    Args:
        seq_len: Sequence length
        decay: Weight decay factor (0 = uniform, >0 = later tokens weighted more)
        device: Target device

    Returns:
        Weight tensor of shape (seq_len,)
    """
    if decay == 0:
        return torch.ones(seq_len, device=device)

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    weights = 1.0 + decay * positions / seq_len
    weights = weights / weights.mean()  # Normalize to mean=1
    return weights


# =============================================================================
# Training Utilities
# =============================================================================

def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    token_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss with optional token-level weighting.

    Returns:
        loss: Scalar loss tensor
        metrics: Dictionary of metrics
    """
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch.get('attention_mask', None)

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    if token_weights is not None and 'logits' in outputs:
        # Recompute loss with token weights
        logits = outputs['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Apply token weights
        loss = loss.view(shift_labels.shape)
        weights = token_weights[:loss.shape[1]].unsqueeze(0)
        loss = (loss * weights).sum() / (shift_labels != -100).sum()
    else:
        loss = outputs['loss']

    # Compute perplexity
    with torch.no_grad():
        ppl = torch.exp(loss).item()

    metrics = {
        'loss': loss.item(),
        'ppl': ppl,
    }

    # Add auxiliary losses if present
    if 'moe_loss' in outputs:
        metrics['moe_loss'] = outputs['moe_loss'].item()
    if 'mod_loss' in outputs:
        metrics['mod_loss'] = outputs['mod_loss'].item()

    return loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler: CosineWarmupScheduler,
    step: int,
    config: TrainConfig,
    metrics: Dict[str, float],
    is_fsdp: bool = False,
):
    """Save training checkpoint."""
    if not is_main_process():
        return

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint-{step}"
    checkpoint_path.mkdir(exist_ok=True)

    # Save model
    if is_fsdp:
        # FSDP requires special handling
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, checkpoint_path / "model.pt")

    # Save optimizer
    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")

    # Save scheduler and training state
    training_state = {
        'step': step,
        'scheduler_step': scheduler.current_step,
        'metrics': metrics,
        'config': vars(config),
    }
    torch.save(training_state, checkpoint_path / "training_state.pt")

    print_rank0(f"  [Checkpoint] Saved to {checkpoint_path}")

    # Clean up old checkpoints (keep last 3)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[1]))
    for old_checkpoint in checkpoints[:-3]:
        import shutil
        shutil.rmtree(old_checkpoint)


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler: CosineWarmupScheduler,
    checkpoint_path: str,
    is_fsdp: bool = False,
) -> int:
    """Load training checkpoint. Returns step number."""
    checkpoint_path = Path(checkpoint_path)

    # Load model
    model_state = torch.load(checkpoint_path / "model.pt", map_location='cpu')
    if is_fsdp:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # Load optimizer
    optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt", map_location='cpu'))

    # Load training state
    training_state = torch.load(checkpoint_path / "training_state.pt")
    scheduler.current_step = training_state['scheduler_step']

    print_rank0(f"  [Checkpoint] Resumed from {checkpoint_path} at step {training_state['step']}")

    return training_state['step']


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch['input_ids'], labels=batch['labels'])

        loss = outputs['loss']
        num_tokens = (batch['labels'] != -100).sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        'eval_loss': avg_loss,
        'eval_ppl': math.exp(avg_loss),
    }


# =============================================================================
# FSDP Setup
# =============================================================================

def setup_fsdp(
    model: nn.Module,
    config: TrainConfig,
    local_rank: int,
) -> nn.Module:
    """Wrap model with FSDP."""
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from moe_arch.model.layers import TransformerBlock

    # Mixed precision policy
    if config.dtype == 'bfloat16':
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif config.dtype == 'float16':
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_policy = None

    # Auto wrap policy - wrap each transformer block
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock},
    )

    # Create FSDP model
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        use_orig_params=True,
        device_id=local_rank,
        sync_module_states=True,
    )

    return model


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainConfig):
    """Main training function."""

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    is_distributed = world_size > 1

    # Set device - use --gpu arg for single GPU, or local_rank for distributed
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    elif config.gpu is not None:
        device = torch.device(f'cuda:{config.gpu}')
        torch.cuda.set_device(config.gpu)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print_rank0("=" * 80)
    print_rank0("CUSTOM MoE TRAINING SCRIPT")
    print_rank0("=" * 80)
    print_rank0(f"\nDevice: {device}")
    print_rank0(f"Distributed: {is_distributed} (world_size={world_size})")

    # Set dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    dtype = dtype_map[config.dtype]
    print_rank0(f"Mixed precision: {config.dtype}")

    # Create model
    print_rank0("\nInitializing model...")
    model_config = AdvancedMoEConfig(**config.model_config)
    model = MoETransformer(model_config)

    params = model.count_parameters()
    print_rank0(f"  Total params: {params['total_billions']:.3f}B")
    print_rank0(f"  Active params: {params['active_billions']:.3f}B ({params['sparsity']:.1%} sparsity)")

    # Enable gradient checkpointing
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print_rank0("  Gradient checkpointing: enabled")

    # Move to device
    model = model.to(device)

    # Setup FSDP if distributed
    if is_distributed:
        print_rank0("\nSetting up FSDP...")
        model = setup_fsdp(model, config, local_rank)
        print_rank0("  FSDP: enabled (FULL_SHARD)")
    elif config.compile_model:
        print_rank0("\nCompiling model with torch.compile...")
        model = torch.compile(model, mode='max-autotune-no-cudagraphs')
        print_rank0("  torch.compile: enabled")

    # Create tokenizer
    print_rank0("\nInitializing tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_rank0(f"  Tokenizer: {config.tokenizer_name} (vocab_size={len(tokenizer)})")

    # Create dataloader
    print_rank0("\nCreating dataloader...")
    dataloader = create_dataloader(config, tokenizer, rank, world_size)
    print_rank0(f"  Sequence packing: {config.use_sequence_packing}")
    print_rank0(f"  Document attention masks: {config.use_document_attention_mask}")
    if config.use_document_attention_mask:
        print_rank0(f"  WARNING: Document masks disable Flash Attention (17x slower)!")
    print_rank0(f"  Batch size: {config.batch_size}")
    print_rank0(f"  Streaming mode: enabled (num_workers=0 for compatibility)")

    # Calculate training steps
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps * world_size
    tokens_per_step = effective_batch_size * config.max_seq_len
    max_steps = config.total_tokens // tokens_per_step

    print_rank0(f"\nTraining configuration:")
    print_rank0(f"  Total tokens: {config.total_tokens:,}")
    print_rank0(f"  Effective batch size: {effective_batch_size}")
    print_rank0(f"  Tokens per step: {tokens_per_step:,}")
    print_rank0(f"  Max steps: {max_steps:,}")

    # Create optimizer
    print_rank0("\nCreating optimizer...")
    if config.use_8bit_optimizer:
        optimizer = Muon8bit(
            model.parameters(),
            lr=config.max_lr,
            momentum=0.95,
            weight_decay=config.weight_decay,
            use_8bit=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.max_lr,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )
        print_rank0("  Using AdamW optimizer")

    # Create scheduler
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=max_steps,
        max_lr=config.max_lr,
        min_lr=config.max_lr * config.min_lr_ratio,
    )
    print_rank0(f"  Warmup steps: {config.warmup_steps:,}")
    print_rank0(f"  LR: {config.max_lr:.2e} -> {config.max_lr * config.min_lr_ratio:.2e}")

    # Token weights
    token_weights = None
    if config.use_token_weights and config.token_weight_decay > 0:
        token_weights = compute_token_weights(
            config.max_seq_len,
            config.token_weight_decay,
            device,
        )
        print_rank0(f"  Token weighting: enabled (decay={config.token_weight_decay})")

    # Resume from checkpoint
    start_step = 0
    if config.resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, config.resume_from, is_distributed)

    # WandB setup
    if config.wandb_enabled and is_main_process():
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(config),
        )
        print_rank0("  WandB: enabled")

    # Training loop
    print_rank0("\n" + "=" * 80)
    print_rank0("STARTING TRAINING")
    print_rank0("=" * 80 + "\n")

    model.train()
    optimizer.zero_grad()

    print_rank0("Initializing data iterator with threaded prefetching...")
    prefetcher = ThreadedPrefetcher(dataloader, device, buffer_size=4)
    data_iter = iter(prefetcher)

    # Pre-fetch first batch to ensure data loading works
    first_batch = next(data_iter)
    print_rank0(f"  First batch loaded: {first_batch['input_ids'].shape}")
    print_rank0(f"  First batch device: {first_batch['input_ids'].device}")
    print_rank0(f"  Prefetch buffer: 4 batches")

    # Put first batch back by creating a chain
    from itertools import chain
    data_iter = chain([first_batch], data_iter)

    accumulated_loss = 0.0
    accumulated_metrics = {}
    step_times = []
    seq_lengths = []
    timing_stats = TimingStats()

    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    pbar = tqdm(range(start_step, max_steps), disable=not is_main_process())

    for step in pbar:
        step_start = time.time()
        data_time = 0.0
        forward_time = 0.0
        backward_time = 0.0

        # Accumulation loop
        for micro_step in range(config.gradient_accumulation_steps):
            # Data loading (already on GPU via prefetcher)
            data_start = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                prefetcher.stop()
                prefetcher = ThreadedPrefetcher(dataloader, device, buffer_size=4)
                data_iter = iter(prefetcher)
                batch = next(data_iter)
            data_time += time.time() - data_start

            # Track sequence lengths (with packed sequences, all positions are used)
            seq_len = batch['input_ids'].shape[1]
            batch_size = batch['input_ids'].shape[0]
            seq_lengths.extend([seq_len] * batch_size)

            # Forward pass with autocast
            if config.profile_timing:
                torch.cuda.synchronize()
            fwd_start = time.time()
            with torch.autocast(device_type='cuda', dtype=dtype):
                loss, metrics = compute_loss(model, batch, token_weights)
                loss = loss / config.gradient_accumulation_steps
            if config.profile_timing:
                torch.cuda.synchronize()
            forward_time += time.time() - fwd_start

            # Backward pass
            bwd_start = time.time()
            loss.backward()
            if config.profile_timing:
                torch.cuda.synchronize()
            backward_time += time.time() - bwd_start

            accumulated_loss += loss.item()
            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v / config.gradient_accumulation_steps

        # Gradient clipping
        if config.grad_clip_norm > 0:
            if is_distributed:
                grad_norm = model.clip_grad_norm_(config.grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        else:
            grad_norm = 0.0

        # Optimizer step
        opt_start = time.time()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        if config.profile_timing:
            torch.cuda.synchronize()
        opt_time = time.time() - opt_start

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Update timing stats
        timing_stats.update(data_time, forward_time, backward_time, opt_time, step_time)

        # Logging
        if (step + 1) % config.log_interval == 0:
            # Average loss over log_interval steps
            num_steps_logged = len(step_times)
            avg_loss = accumulated_loss / num_steps_logged if num_steps_logged > 0 else 0.0
            avg_time = sum(step_times) / len(step_times)
            tokens_per_sec = tokens_per_step / avg_time
            current_lr = scheduler.get_lr()

            # Average metrics
            avg_metrics = {k: v / num_steps_logged for k, v in accumulated_metrics.items()}

            # Memory stats
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(device) / 1e9
            else:
                mem_alloc = mem_reserved = 0

            # Sequence length stats
            if seq_lengths:
                avg_seq_len = sum(seq_lengths) / len(seq_lengths)
                min_seq_len = min(seq_lengths)
                max_seq_len = max(seq_lengths)
            else:
                avg_seq_len = min_seq_len = max_seq_len = 0

            log_msg = (
                f"step {step+1}/{max_steps} | "
                f"loss: {avg_loss:.4f} | "
                f"ppl: {math.exp(avg_loss):.2f} | "
                f"lr: {current_lr:.2e} | "
                f"tok/s: {tokens_per_sec:,.0f} | "
                f"mem: {mem_alloc:.1f}GB | "
                f"seq: {avg_seq_len:.0f}"
            )
            print_rank0(log_msg)
            if config.profile_timing:
                print_rank0(f"  [Timing] {timing_stats.summary()}")

            # WandB logging
            if config.wandb_enabled and is_main_process():
                import wandb
                wandb.log({
                    'train/loss': avg_loss,
                    'train/ppl': math.exp(avg_loss),
                    'train/lr': current_lr,
                    'train/grad_norm': grad_norm,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/step_time': avg_time,
                    'memory/allocated_gb': mem_alloc,
                    'memory/reserved_gb': mem_reserved,
                    'data/avg_seq_len': avg_seq_len,
                    'data/min_seq_len': min_seq_len,
                    'data/max_seq_len': max_seq_len,
                    **{f'train/{k}': v for k, v in avg_metrics.items()},
                }, step=step+1)

            # Reset accumulators
            accumulated_loss = 0.0
            accumulated_metrics = {}
            step_times = []
            seq_lengths = []
            timing_stats.reset()

        # Update progress bar with running average
        steps_since_log = (step + 1) % config.log_interval
        if steps_since_log == 0:
            steps_since_log = config.log_interval
        running_avg_loss = accumulated_loss / steps_since_log if steps_since_log > 0 else 0.0
        pbar.set_postfix({
            'loss': f"{running_avg_loss:.4f}" if accumulated_loss > 0 else "...",
            'lr': f"{scheduler.get_lr():.2e}",
        })

        # Checkpointing
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, step + 1, config,
                accumulated_metrics, is_distributed,
            )

        # Evaluation
        if config.eval_enabled and (step + 1) % config.eval_interval == 0:
            print_rank0("\n  [Eval] Running evaluation...")
            # TODO: Implement evaluation on configured datasets
            print_rank0("  [Eval] Evaluation not yet implemented for custom datasets\n")

    # Stop prefetcher
    prefetcher.stop()

    # Final save
    save_checkpoint(model, optimizer, scheduler, max_steps, config, accumulated_metrics, is_distributed)

    print_rank0("\n" + "=" * 80)
    print_rank0("TRAINING COMPLETE")
    print_rank0("=" * 80)

    # Cleanup
    if config.wandb_enabled and is_main_process():
        import wandb
        wandb.finish()

    cleanup_distributed()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Custom MoE Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--no-packing", action="store_true", help="Disable sequence packing")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--profile", action="store_true", help="Enable detailed timing (adds overhead)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use (e.g., 0 or 1)")
    args = parser.parse_args()

    # Load config
    config = TrainConfig.from_yaml(args.config)

    # Override with CLI args
    if args.resume:
        config.resume_from = args.resume
    if args.wandb:
        config.wandb_enabled = True
    if args.no_packing:
        config.use_sequence_packing = False
    if args.compile:
        config.compile_model = True
    if args.profile:
        config.profile_timing = True
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.gpu is not None:
        config.gpu = args.gpu

    # Run training
    train(config)


if __name__ == "__main__":
    main()
