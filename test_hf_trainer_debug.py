"""
Debug script that matches actual HF Trainer setup exactly.
"""

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, GPT2TokenizerFast
from datasets import load_dataset
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.training.muon_optimizer import get_muon_optimizer

print("=" * 70)
print("HF TRAINER DEBUG - MATCHING ACTUAL SETUP")
print("=" * 70)

# Use SAME config as actual training (but smaller for speed)
config = AdvancedMoEConfig(
    vocab_size=50257,
    d_model=512,
    n_layers=6,
    n_heads=8,
    n_kv_heads=2,
    head_dim=64,
    n_experts=11,
    moe_top_k=2,
    moe_layers=(1, 2, 4, 5),

    # KEY: Same as actual training
    mod_enabled=True,  # User has this enabled
    mod_capacity_factor=0.75,

    mamba_enabled=False,  # Simplify for now
    mamba_layers=(),

    n_pred_tokens=4,  # Same as actual
    aux_loss_weights=(1.0, 0.5, 0.3, 0.2),

    max_seq_len=512,
    use_flash_attention=False,
)

print(f"\nConfig:")
print(f"  d_model={config.d_model}, n_layers={config.n_layers}, n_experts={config.n_experts}")
print(f"  MoD enabled: {config.mod_enabled}")
print(f"  Multi-token: {config.n_pred_tokens} heads")

# Create model
model = MoETransformer(config)
print(f"\nModel created: {model.count_parameters()['total_billions']:.3f}B params")

# Wrapper
class SimpleModelWrapper(torch.nn.Module):
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

wrapped_model = SimpleModelWrapper(model)

# Custom Trainer with Muon
class MuonTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            print("\n[DEBUG] Creating Muon optimizer...")
            self.optimizer = get_muon_optimizer(
                self.model.model,
                lr=self.args.learning_rate,
                momentum=0.95,
                weight_decay=0.01,
            )
            print(f"[DEBUG] Optimizer created")
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        # Use our custom LR scheduler (same as actual training)
        from moe_arch.training.lr_schedule import CosineScheduleWithWarmup

        if self.lr_scheduler is None:
            self.lr_scheduler = CosineScheduleWithWarmup(
                optimizer=self.optimizer,
                warmup_steps=self.args.warmup_steps,
                max_steps=num_training_steps,
                max_lr=self.args.learning_rate,
                min_lr=self.args.learning_rate * 0.1,
                warmup_init_lr=0.0,
            )
            print(f"[DEBUG] Custom LR scheduler created")
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Debug version to track loss values."""
        print(f"\n[COMPUTE_LOSS] Called with return_outputs={return_outputs}")
        print(f"[COMPUTE_LOSS] Input shape: {inputs['input_ids'].shape}")
        print(f"[COMPUTE_LOSS] Batch size: {inputs['input_ids'].shape[0]}")

        model_inputs = {
            'input_ids': inputs['input_ids'],
            'labels': inputs['labels'] if 'labels' in inputs else inputs['input_ids'],
        }

        outputs = model(**model_inputs)
        loss = outputs['loss']

        print(f"[COMPUTE_LOSS] Loss from model: {loss.item():.4f}")
        print(f"[COMPUTE_LOSS] Loss dtype: {loss.dtype}")
        print(f"[COMPUTE_LOSS] Loss shape: {loss.shape}")
        print(f"[COMPUTE_LOSS] Loss requires_grad: {loss.requires_grad}")

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[COMPUTE_LOSS] WARNING: Loss is NaN or Inf!")

        return (loss, outputs) if return_outputs else loss

# Load tiny dataset
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)

# CRITICAL: Filter out empty sequences (causes NaN loss!)
print(f"\nDataset size before filtering: {len(tokenized)}")
tokenized = tokenized.filter(lambda x: len(x['input_ids']) > 0)
print(f"Dataset size after filtering: {len(tokenized)}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training args - SAME as actual
training_args = TrainingArguments(
    output_dir="./test_hf_debug",
    max_steps=20,  # Just 20 steps to test
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # SAME as actual!
    learning_rate=1e-3,
    warmup_steps=5,
    bf16=True,
    logging_steps=1,
    logging_first_step=True,
    save_strategy="no",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

print(f"\nTraining config:")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Grad accum: {training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  BF16: {training_args.bf16}")

# Create trainer
trainer = MuonTrainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print("\n" + "=" * 70)
print("STARTING TRAINING (20 steps)")
print("=" * 70)
print("\nIf loss decreases → Everything works!")
print("If loss stays flat → Problem with HF Trainer setup\n")

try:
    trainer.train()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    # Check loss progression
    metrics = trainer.state.log_history
    if metrics:
        losses = [m['loss'] for m in metrics if 'loss' in m]
        if len(losses) >= 2:
            print(f"\nLoss progression:")
            print(f"  Initial: {losses[0]:.4f}")
            print(f"  Final: {losses[-1]:.4f}")
            print(f"  Change: {losses[0] - losses[-1]:.4f}")

            if losses[-1] < losses[0] - 0.1:
                print("\n✅ PASS: Loss is decreasing with HF Trainer!")
            else:
                print("\n❌ FAIL: Loss not decreasing with HF Trainer")
                print("\nAll losses:")
                for i, l in enumerate(losses):
                    print(f"  Step {i}: {l:.4f}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
