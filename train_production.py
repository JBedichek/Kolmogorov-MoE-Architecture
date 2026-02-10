"""
Production training script with FineWeb dataset.

Features:
- Configurable via YAML config file
- Checkpoint saving and resuming
- Weights & Biases logging
- Validation during training
- Automatic step calculation from total tokens
- Progress tracking

Usage:
    python train_production.py --config configs/production_training.yaml
    python train_production.py --config configs/production_training.yaml --resume checkpoints_production/checkpoint-1000
"""

import argparse
import yaml
import os
import torch
from typing import Optional, Dict, List, Any
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, GPT2TokenizerFast
from datasets import load_dataset, Dataset
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.training.muon_optimizer import get_muon_optimizer


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class OnTheFlyDataCollator:
    """
    Data collator that tokenizes text on-the-fly during training.

    This is more memory-efficient than pre-tokenizing the entire dataset,
    especially for large datasets, but may be slightly slower per batch.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract text from examples
        texts = [example["text"] for example in examples]

        # Tokenize on-the-fly
        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # For language modeling, labels are the same as input_ids
        batch["labels"] = batch["input_ids"].clone()

        return batch


class MoEModelWrapper(torch.nn.Module):
    """Wrapper to make MoE model compatible with HuggingFace Trainer."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.model(input_ids, labels=labels)
        return {
            'loss': outputs['loss'],
            'logits': outputs['logits'],
        }


class MoETrainer(Trainer):
    """Custom HuggingFace Trainer with Muon optimizer and custom LR scheduler."""

    def __init__(self, muon_config: dict, lr_config: dict, *args, **kwargs):
        self.muon_config = muon_config
        self.lr_config = lr_config
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Create Muon optimizer instead of default AdamW."""
        if self.optimizer is None:
            print("\n[Creating Muon optimizer...]")
            self.optimizer = get_muon_optimizer(
                self.model.model,
                lr=self.args.learning_rate,
                momentum=self.muon_config.get('momentum', 0.95),
                weight_decay=self.muon_config.get('weight_decay', 0.01),
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create custom cosine LR scheduler with warmup."""
        from moe_arch.training.lr_schedule import CosineScheduleWithWarmup

        if self.lr_scheduler is None:
            self.lr_scheduler = CosineScheduleWithWarmup(
                optimizer=self.optimizer,
                warmup_steps=self.args.warmup_steps,
                max_steps=num_training_steps,
                max_lr=self.args.learning_rate,
                min_lr=self.args.learning_rate * self.lr_config.get('min_lr_ratio', 0.1),
                warmup_init_lr=0.0,
            )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with empty sequence protection."""
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'labels': inputs['labels'] if 'labels' in inputs else inputs['input_ids'],
        }

        # Skip empty sequences
        if model_inputs['input_ids'].shape[1] == 0:
            return torch.tensor(0.0, device=model_inputs['input_ids'].device, requires_grad=True)

        outputs = model(**model_inputs)
        loss = outputs['loss']

        return (loss, outputs) if return_outputs else loss


def load_and_prepare_dataset(config: dict, tokenizer, split: str = "train"):
    """Load dataset and optionally tokenize it upfront."""
    data_config = config['data']
    tokenize_on_fly = data_config.get('tokenize_on_fly', False)

    print(f"\nLoading {split} dataset...")
    print(f"  Dataset: {data_config['dataset_name']}")
    print(f"  Tokenization: {'on-the-fly' if tokenize_on_fly else 'upfront'}")

    # Load dataset
    dataset = load_dataset(
        data_config['dataset_name'],
        name=data_config.get('dataset_config'),
        split=split,
        streaming=data_config.get('use_streaming', False),
    )

    # Limit dataset size if specified
    max_examples = data_config.get('max_examples')
    if max_examples:
        print(f"  Limiting to {max_examples:,} examples...")
        if data_config.get('use_streaming', False):
            dataset = dataset.take(max_examples)
        else:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

    print(f"  ✓ Dataset loaded")

    # If tokenizing on-the-fly, just filter empty texts and return
    if tokenize_on_fly:
        print(f"  Preparing for on-the-fly tokenization...")

        # Filter empty texts
        def has_text(example):
            return example.get("text", "").strip() != ""

        if data_config.get('use_streaming', False):
            # For streaming, convert to list with filtering
            print(f"  Converting streaming dataset to list...")
            dataset_list = []
            for item in dataset:
                if has_text(item):
                    dataset_list.append(item)
                if max_examples and len(dataset_list) >= max_examples:
                    break
            dataset = Dataset.from_list(dataset_list)
        else:
            dataset = dataset.filter(has_text)

        print(f"  ✓ {len(dataset):,} examples ready for on-the-fly tokenization")
        return dataset

    # Otherwise, tokenize upfront (original behavior)
    print(f"  Tokenizing upfront...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['model']['max_seq_len'],
            padding=False,
        )

    # Get column names to remove
    if data_config.get('use_streaming', False):
        # For streaming, we need to peek at first example
        first_example = next(iter(dataset))
        columns_to_remove = list(first_example.keys())
    else:
        columns_to_remove = dataset.column_names

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
    )

    # Convert streaming to list if needed
    if data_config.get('use_streaming', False):
        print(f"  Converting streaming dataset to list...")
        tokenized_list = []
        for item in tokenized:
            if len(item['input_ids']) > 0:  # Filter empty sequences
                tokenized_list.append(item)
            if max_examples and len(tokenized_list) >= max_examples:
                break
        tokenized = Dataset.from_list(tokenized_list)
    else:
        # Filter empty sequences for non-streaming
        tokenized = tokenized.filter(lambda x: len(x['input_ids']) > 0)

    print(f"  ✓ {len(tokenized):,} examples tokenized")

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Production MoE Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    print("=" * 80)
    print("MoE PRODUCTION TRAINING")
    print("=" * 80)

    print(f"\nLoading config from {args.config}...")
    config = load_config(args.config)
    print("  ✓ Config loaded")

    # Create model configuration
    print("\nInitializing model...")
    model_config = AdvancedMoEConfig(**config['model'])

    # Create model
    model = MoETransformer(model_config)
    params = model.count_parameters()
    print(f"  ✓ Model created: {params['total_billions']:.3f}B params")
    print(f"    Active: {params['active_billions']:.3f}B ({params['sparsity']:.1%} sparsity)")

    # Wrap model
    wrapped_model = MoEModelWrapper(model)

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(config['data']['tokenizer_name'])
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer: {config['data']['tokenizer_name']}")

    # Load and prepare datasets
    train_dataset = load_and_prepare_dataset(config, tokenizer, split="train")

    # Evaluation dataset
    eval_dataset = None
    if config['evaluation'].get('enabled', False):
        try:
            eval_dataset = load_and_prepare_dataset(config, tokenizer, split="validation")
        except:
            print("  No validation split, using subset of train for eval...")
            eval_size = min(config['evaluation'].get('eval_steps', 100) * 10, len(train_dataset) // 10)
            eval_dataset = train_dataset.select(range(eval_size))

    # Calculate training steps
    train_config = config['training']
    total_tokens = train_config['total_tokens']
    batch_size = train_config['batch_size']
    grad_accum = train_config['gradient_accumulation_steps']
    seq_len = model_config.max_seq_len

    effective_batch_size = batch_size * grad_accum
    tokens_per_step = effective_batch_size * seq_len
    max_steps = total_tokens // tokens_per_step

    print(f"\nTraining configuration:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Total steps: {max_steps:,}")

    # Create checkpoint directory
    os.makedirs(train_config['checkpoint_dir'], exist_ok=True)

    # Weights & Biases setup
    report_to = ["wandb"] if config['wandb'].get('enabled', False) else ["none"]
    run_name = config['wandb'].get('run_name', 'moe-training')

    # Training arguments
    training_args = TrainingArguments(
        # Output
        output_dir=train_config['checkpoint_dir'],
        run_name=run_name,

        # Training
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        # Learning rate
        learning_rate=train_config['max_lr'],
        warmup_steps=train_config['warmup_steps'],

        # Optimization
        max_grad_norm=train_config['gradient_clip_norm'],

        # Mixed precision
        bf16=config['mixed_precision'].get('bf16', True),

        # Logging
        logging_dir=os.path.join(train_config['checkpoint_dir'], "logs"),
        logging_steps=train_config['log_interval'],
        logging_first_step=True,
        report_to=report_to,

        # Checkpointing
        save_strategy="steps",
        save_steps=train_config['save_interval'],
        save_total_limit=3,

        # Evaluation
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_config.get('eval_interval', 500) if eval_dataset is not None else None,
        per_device_eval_batch_size=batch_size,

        # Performance
        dataloader_num_workers=config['data'].get('num_workers', 4),
        dataloader_pin_memory=True,
        remove_unused_columns=False,

        # Other
        seed=42,
        disable_tqdm=False,
        load_best_model_at_end=False,
    )

    print(f"\nHuggingFace Trainer configuration:")
    print(f"  Max steps: {training_args.max_steps:,}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Warmup steps: {training_args.warmup_steps}")
    print(f"  BF16: {training_args.bf16}")
    print(f"  Save interval: {train_config['save_interval']} steps")
    print(f"  Log interval: {train_config['log_interval']} steps")

    # Data collator (choose based on tokenization strategy)
    tokenize_on_fly = config['data'].get('tokenize_on_fly', False)
    if tokenize_on_fly:
        print(f"\nUsing on-the-fly tokenization data collator")
        data_collator = OnTheFlyDataCollator(
            tokenizer=tokenizer,
            max_length=model_config.max_seq_len,
        )
    else:
        print(f"\nUsing standard data collator (pre-tokenized)")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    # LR config for custom scheduler
    lr_config = {
        'min_lr_ratio': train_config.get('min_lr_ratio', 0.1),
    }

    # Create trainer
    print("\nInitializing trainer...")
    trainer = MoETrainer(
        muon_config=config['optimizer'],
        lr_config=lr_config,
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    print("  ✓ Trainer created")

    # Resume from checkpoint
    resume_from_checkpoint = args.resume or train_config.get('resume_from')
    if resume_from_checkpoint:
        print(f"\n⚠ Resuming from checkpoint: {resume_from_checkpoint}")

    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"\nCheckpoints will be saved to: {train_config['checkpoint_dir']}")
    print(f"Logs will be saved to: {training_args.logging_dir}")
    if config['wandb'].get('enabled', False):
        print(f"W&B project: {config['wandb']['project']}")
        print(f"W&B run: {run_name}")
    print()

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        # Print training summary
        print(f"\nTraining summary:")
        print(f"  Total steps: {trainer.state.global_step}")
        print(f"  Final loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")
        print(f"  Checkpoints saved to: {train_config['checkpoint_dir']}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 80)
        print(f"\nLast checkpoint saved to: {train_config['checkpoint_dir']}")
        print(f"Resume with: python train_production.py --config {args.config} --resume <checkpoint-path>")

    except Exception as e:
        print(f"\n\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
