"""
Debug script to check if gradients are flowing and model is learning.
"""

import torch
from transformers import GPT2TokenizerFast
from moe_arch.model.transformer import MoETransformer
from moe_arch.model.config import AdvancedMoEConfig

print("=" * 70)
print("GRADIENT FLOW DEBUG")
print("=" * 70)

# Create small model
config = AdvancedMoEConfig(
    vocab_size=50257,
    d_model=256,
    n_layers=4,
    n_heads=8,
    n_kv_heads=2,
    head_dim=32,  # d_model / n_heads = 256 / 8 = 32
    n_experts=4,
    moe_layers=(1, 2),
    mod_enabled=False,
    mamba_enabled=False,
    mamba_layers=(),
    n_pred_tokens=1,
    aux_loss_weights=(1.0,),
    max_seq_len=128,
    use_flash_attention=False,
)

print(f"\nModel config:")
print(f"  d_model={config.d_model}, n_layers={config.n_layers}")
print(f"  MoD enabled: {config.mod_enabled}")

# Create model and optimizer
model = MoETransformer(config)
model.train()

# Simple optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Create dummy data
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
text = "The quick brown fox jumps over the lazy dog"
tokens = tokenizer.encode(text, return_tensors="pt")
if tokens.size(1) > config.max_seq_len:
    tokens = tokens[:, :config.max_seq_len]

print(f"\nInput shape: {tokens.shape}")

# Check initial state
print("\n" + "=" * 70)
print("INITIAL STATE (Before any training)")
print("=" * 70)

with torch.no_grad():
    outputs = model(tokens, labels=tokens)
    initial_loss = outputs['loss'].item()
    print(f"Initial loss: {initial_loss:.4f}")

    # Check a few parameter values
    for name, param in model.named_parameters():
        if 'token_embedding' in name:
            print(f"Embedding mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
            break

    for name, param in model.named_parameters():
        if 'layers.0' in name and 'weight' in name:
            print(f"Layer 0 weight mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
            break

# Train for 10 steps
print("\n" + "=" * 70)
print("TRAINING FOR 10 STEPS")
print("=" * 70)

losses = []
grad_norms = []

for step in range(10):
    optimizer.zero_grad()

    outputs = model(tokens, labels=tokens)
    loss = outputs['loss']

    loss.backward()

    # Check gradient norms
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    optimizer.step()

    losses.append(loss.item())
    grad_norms.append(total_norm)

    if step % 2 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}")

# Check final state
print("\n" + "=" * 70)
print("FINAL STATE (After 10 steps)")
print("=" * 70)

with torch.no_grad():
    outputs = model(tokens, labels=tokens)
    final_loss = outputs['loss'].item()
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss change: {initial_loss - final_loss:.4f}")

    # Check if parameters changed
    for name, param in model.named_parameters():
        if 'token_embedding' in name:
            print(f"Embedding mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
            break

    for name, param in model.named_parameters():
        if 'layers.0' in name and 'weight' in name:
            print(f"Layer 0 weight mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
            break

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if final_loss < initial_loss - 0.01:
    print("✅ PASS: Loss is decreasing (gradients are flowing)")
else:
    print("❌ FAIL: Loss is NOT decreasing")
    print("\nPossible issues:")
    print("  1. Learning rate too low")
    print("  2. Gradients not flowing through some layers")
    print("  3. Loss computation issue")
    print("  4. Optimizer issue")

if grad_norms[-1] > 0.1:
    print("✅ PASS: Gradients have reasonable magnitude")
else:
    print("❌ FAIL: Gradients are too small or vanishing")

print("\n" + "=" * 70)
print("Loss over steps:")
for i, l in enumerate(losses):
    print(f"  Step {i}: {l:.4f}")
print("=" * 70)
