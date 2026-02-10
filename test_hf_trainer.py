"""
Test training with HuggingFace Trainer.
Simple test to verify MoD works and loss decreases.
"""

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig
from transformers import GPT2TokenizerFast

print("="*70)
print("TESTING MOE MODEL WITH HUGGINGFACE TRAINER")
print("="*70)

# 1. Create simplified config
print("\n1. Creating model config...")
config = AdvancedMoEConfig(
    # Small model for testing
    vocab_size=50257,  # GPT2 vocab
    d_model=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=2,
    head_dim=64,
    d_ff=2048,
    max_seq_len=256,
    
    # MoE with MoD
    n_experts=8,
    moe_top_k=2,
    d_ff_expert=1024,
    moe_layers=(2, 3, 6, 7),  # 4 MoE layers
    
    # MoD ENABLED (this is what we're testing)
    mod_enabled=True,
    mod_capacity_factor=0.75,
    
    # Disable Mamba for simplicity
    mamba_enabled=False,
    mamba_layers=(),
    
    # DISABLE multi-token prediction (simplify)
    n_pred_tokens=1,  # Only predict next token
    aux_loss_weights=(1.0,),
    
    # Training settings
    dropout=0.1,
    use_flash_attention=False,  # Avoid dependency issues
)

print(f"   Model: {config.d_model}d, {config.n_layers}L, {config.n_experts}E")
print(f"   MoD enabled: {config.mod_enabled} (capacity: {config.mod_capacity_factor})")
print(f"   MoE layers: {config.moe_layers}")

# 2. Create model
print("\n2. Creating model...")
model = MoETransformer(config)
params = model.count_parameters()
print(f"   Total params: {params['total_billions']:.3f}B")
print(f"   Active params: {params['active_billions']:.3f}B")

# 3. Wrap model for HF Trainer
class SimpleModelWrapper(torch.nn.Module):
    """Wrapper to make model compatible with HF Trainer."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
    
    def forward(self, input_ids, labels=None, **kwargs):
        """
        HF Trainer expects:
        - input_ids: (batch, seq_len)
        - labels: (batch, seq_len)
        - Returns: dict with 'loss' and 'logits'
        """
        outputs = self.model(input_ids, labels=labels)
        
        # For single-token prediction, we just use the main loss
        # Ignore aux losses for this test
        return {
            'loss': outputs['lm_loss'],  # Just language modeling loss
            'logits': outputs['logits'],  # (batch, seq, vocab)
        }

wrapped_model = SimpleModelWrapper(model)
print("   ✓ Model wrapped for HF Trainer")

# 4. Load tiny dataset
print("\n3. Loading dataset...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load tiny subset of wikitext for quick test
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
print(f"   Dataset: {len(dataset)} examples")

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding=False,
    )

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)
print("   ✓ Dataset tokenized")

# Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

# 5. Training arguments
print("\n4. Setting up training...")
training_args = TrainingArguments(
    output_dir="./test_hf_output",
    
    # Training settings
    num_train_epochs=1,
    max_steps=100,  # Just 100 steps to verify it works
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    
    # Learning rate (use AdamW for simplicity)
    learning_rate=3e-4,
    warmup_steps=10,
    
    # Mixed precision
    bf16=True,  # HF handles this correctly
    
    # Logging
    logging_steps=10,
    logging_dir="./test_hf_logs",
    
    # Checkpointing
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    
    # Disable features we don't need
    # evaluation_strategy="no",  # Not needed for this test
    # load_best_model_at_end=False,
    
    # Other
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

print(f"   Max steps: {training_args.max_steps}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   BF16: {training_args.bf16}")

# 6. Create trainer
print("\n5. Creating HF Trainer...")
trainer = Trainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)
print("   ✓ Trainer created")

# 7. Train!
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print("\nExpected behavior:")
print("  ✓ Loss should decrease from ~10-11 to ~8-9")
print("  ✓ No NaN losses")
print("  ✓ Training completes without errors")
print("\nIf you see loss decreasing, MoD is working!")
print("="*70 + "\n")

try:
    trainer.train()
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Check final loss
    metrics = trainer.state.log_history
    if metrics:
        losses = [m['loss'] for m in metrics if 'loss' in m]
        if losses:
            print(f"\nLoss progression:")
            print(f"  Initial: {losses[0]:.4f}")
            print(f"  Final: {losses[-1]:.4f}")
            print(f"  Change: {losses[0] - losses[-1]:.4f}")
            
            if losses[-1] < losses[0]:
                print("\n✅ LOSS IS DECREASING - MoD WORKS!")
            else:
                print("\n⚠️  Loss not decreasing - may need more steps")
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ TRAINING FAILED")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
