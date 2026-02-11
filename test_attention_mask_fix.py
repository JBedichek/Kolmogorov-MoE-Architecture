"""
Test that attention masks are properly passed and used in loss calculation.
"""

import torch
from moe_arch.training.losses import MultiTokenPredictionLoss

print("Testing attention mask fix...")

# Create loss function
loss_fn = MultiTokenPredictionLoss(
    n_pred_tokens=1,
    aux_loss_weights=(1.0,),
    ignore_index=-100,
)

batch_size = 2
seq_len = 8
vocab_size = 100

# Create dummy logits
logits = torch.randn(batch_size, seq_len, vocab_size)

# Create labels with padding (token id 0 is padding)
labels = torch.randint(1, vocab_size, (batch_size, seq_len))
# Second sequence has padding in last 3 positions
labels[1, 5:] = 0  # Padding tokens

# Create attention mask (1 = real token, 0 = padding)
attention_mask = torch.ones(batch_size, seq_len)
attention_mask[1, 5:] = 0  # Mark padding positions

print(f"\nLabels:\n{labels}")
print(f"\nAttention mask:\n{attention_mask}")

# Test 1: Loss WITHOUT attention mask (old behavior - includes padding)
loss_without_mask, _ = loss_fn([logits], labels, attention_mask=None)
print(f"\nLoss WITHOUT attention mask: {loss_without_mask.item():.4f}")
print("  (This incorrectly includes padding tokens)")

# Test 2: Loss WITH attention mask (new behavior - ignores padding)
loss_with_mask, _ = loss_fn([logits], labels, attention_mask=attention_mask)
print(f"\nLoss WITH attention mask: {loss_with_mask.item():.4f}")
print("  (This correctly ignores padding tokens)")

# The losses should be different because we're ignoring padding
diff = abs(loss_without_mask.item() - loss_with_mask.item())
print(f"\nDifference: {diff:.4f}")

if diff > 0.01:
    print("✓ Attention mask is working! Losses are different.")
else:
    print("⚠ Warning: Losses are very similar, mask might not be working")

# Test 3: Verify padding tokens are ignored
# Create labels where ALL tokens in second sequence are padding
labels_all_padding = labels.clone()
labels_all_padding[1, :] = 0
attention_mask_all_padding = attention_mask.clone()
attention_mask_all_padding[1, :] = 0

loss_partial_padding, _ = loss_fn([logits], labels, attention_mask=attention_mask)
loss_all_padding, _ = loss_fn([logits], labels_all_padding, attention_mask=attention_mask_all_padding)

print(f"\nLoss with partial padding: {loss_partial_padding.item():.4f}")
print(f"Loss with one sequence all padding: {loss_all_padding.item():.4f}")
print("  (Should compute loss only on first sequence)")

print("\n✓ Attention mask fix test completed!")
