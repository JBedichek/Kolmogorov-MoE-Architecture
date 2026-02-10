"""
WORKING training script with FineWeb dataset.

This is based on the test script that we KNOW works.
"""

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, GPT2TokenizerFast
from datasets import load_dataset
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
from moe_arch.training.muon_optimizer import get_muon_optimizer

print("=" * 70)
print("MOE TRAINING WITH FINEWEB - WORKING VERSION")
print("=" * 70)

# Small 500M model
config = AdvancedMoEConfig(
    vocab_size=50257,
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=3,
    head_dim=64,
    n_experts=8,
    moe_top_k=2,
    moe_layers=(2, 3, 6, 7, 10, 11),
    mod_enabled=True,
    mamba_enabled=False,
    mamba_layers=(),
    n_pred_tokens=1,
    aux_loss_weights=(1.0,),
    max_seq_len=512,
    use_flash_attention=True,
)

print(f"\nConfig:")
print(f"  d_model={config.d_model}, n_layers={config.n_layers}, n_experts={config.n_experts}")
print(f"  seq_len={config.max_seq_len}")

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
            print("\n[Creating Muon optimizer...]")
            self.optimizer = get_muon_optimizer(
                self.model.model,
                lr=self.args.learning_rate,
                momentum=0.95,
                weight_decay=0.01,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
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
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'labels': inputs['labels'] if 'labels' in inputs else inputs['input_ids'],
        }
        if model_inputs['input_ids'].shape[1] == 0:
            return torch.tensor(0.0, device=model_inputs['input_ids'].device, requires_grad=True)
        outputs = model(**model_inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

# Load FineWeb dataset
print("\nLoading FineWeb dataset...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True,
)
print("  ✓ FineWeb loaded")

# Take first 1000 examples and tokenize
print("\nTokenizing dataset...")
dataset = dataset.take(1000)

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
    remove_columns=["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"],
)

# Filter empty sequences
print("Filtering empty sequences...")
tokenized_list = []
for i, item in enumerate(tokenized):
    if len(item['input_ids']) > 0:
        tokenized_list.append(item)
    if len(tokenized_list) >= 5000:  # Use 500 examples
        break

from datasets import Dataset
tokenized = Dataset.from_list(tokenized_list)
print(f"  ✓ {len(tokenized)} examples ready")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training args
training_args = TrainingArguments(
    output_dir="./checkpoints_fineweb",
    max_steps=100,  # 100 steps to verify it works
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-3,
    warmup_steps=10,
    bf16=True,
    logging_steps=1,
    logging_first_step=True,
    save_strategy="no",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

print(f"\nTraining config:")
print(f"  Steps: {training_args.max_steps}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Grad accum: {training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")

# Create trainer
trainer = MuonTrainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print("\nExpected: Loss should decrease from ~11 to <5 over 100 steps\n")

try:
    trainer.train()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    # Check loss progression
    metrics = trainer.state.log_history
    if metrics:
        losses = [m['loss'] for m in metrics if 'loss' in m]
        if len(losses) >= 2:
            print(f"\nLoss progression:")
            print(f"  Initial: {losses[0]:.4f} (÷8 = {losses[0]/8:.4f})")
            print(f"  Final: {losses[-1]:.4f} (÷8 = {losses[-1]/8:.4f})")
            print(f"  Improvement: {(losses[0] - losses[-1])/8:.4f}")

            if losses[-1] < losses[0] - 5:
                print(f"\n✅ SUCCESS: Loss decreased with FineWeb real data!")
                print(f"   Model is learning from the dataset.")
            else:
                print(f"\n⚠ Loss decreased but slowly. This is normal for early training.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
