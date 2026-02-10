"""
Loss functions for training, including multi-token prediction.

Reference: "Better & Faster Large Language Models via Multi-token Prediction" (Meta, 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class MultiTokenPredictionLoss(nn.Module):
    """
    Multi-token prediction loss.

    Predicts multiple future tokens simultaneously (t+1, t+2, t+3, t+4)
    and combines their losses with configurable weights.

    Benefits:
    - Better gradient signal (4x more supervision per token)
    - Improved sample efficiency
    - Forces model to learn more robust representations
    - ~10-15% perplexity improvement (from Llama 3 findings)
    """

    def __init__(
        self,
        n_pred_tokens: int = 4,
        aux_loss_weights: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.2),
        ignore_index: int = -100,
    ):
        super().__init__()
        self.n_pred_tokens = n_pred_tokens

        # Normalize weights to sum to 1.0 to keep loss magnitude comparable to standard LM
        weight_sum = sum(aux_loss_weights)
        self.aux_loss_weights = tuple(w / weight_sum for w in aux_loss_weights)
        self.ignore_index = ignore_index

        assert len(aux_loss_weights) == n_pred_tokens, \
            f"aux_loss_weights length ({len(aux_loss_weights)}) must match n_pred_tokens ({n_pred_tokens})"

        print(f"Multi-token loss weights (normalized): {[f'{w:.3f}' for w in self.aux_loss_weights]}")

    def forward(
        self,
        logits_list: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-token prediction loss.

        Args:
            logits_list: List of logits tensors, each (batch, seq_len, vocab_size)
                        [logits_t1, logits_t2, logits_t3, logits_t4]
                        where logits_ti predicts token at position t+i
            labels: (batch, seq_len) - ground truth token IDs

        Returns:
            total_loss: Weighted sum of all prediction losses
            loss_dict: Dictionary with individual losses for logging
        """
        assert len(logits_list) == self.n_pred_tokens, \
            f"Expected {self.n_pred_tokens} logits tensors, got {len(logits_list)}"

        batch_size, seq_len = labels.shape
        individual_losses = []
        loss_dict = {}

        # Compute loss for each prediction head
        for i, (logits, weight) in enumerate(zip(logits_list, self.aux_loss_weights)):
            # Shift logits and labels for predicting token at t+i
            # logits_ti predicts labels at position t+i
            # So we need logits[:, :-i] to predict labels[:, i:]

            if i == 0:
                # Standard next-token prediction (t+1)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
            else:
                # Multi-token ahead prediction (t+2, t+3, etc.)
                # Need to ensure we don't go beyond sequence length
                if i >= seq_len:
                    # Skip this prediction if offset is too large
                    individual_losses.append(torch.tensor(0.0, device=logits.device))
                    loss_dict[f'loss_t{i+1}'] = 0.0
                    continue

                shift_logits = logits[:, :-i-1, :].contiguous()
                shift_labels = labels[:, i+1:].contiguous()

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=self.ignore_index,
                reduction='mean',
            )

            individual_losses.append(loss)
            loss_dict[f'loss_t{i+1}'] = loss.item()

        # Compute weighted total loss
        total_loss = sum(
            weight * loss
            for weight, loss in zip(self.aux_loss_weights, individual_losses)
        )

        loss_dict['total_multitoken_loss'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test multi-token prediction loss
    print("Testing Multi-Token Prediction Loss...")

    batch_size = 2
    seq_len = 16
    vocab_size = 1000
    n_pred_tokens = 4

    # Create loss module
    loss_fn = MultiTokenPredictionLoss(
        n_pred_tokens=n_pred_tokens,
        aux_loss_weights=(1.0, 0.5, 0.3, 0.2),
    )

    print(f"  Prediction heads: {n_pred_tokens}")
    print(f"  Loss weights: {loss_fn.aux_loss_weights}")

    # Create dummy logits for each prediction head
    logits_list = [
        torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        for _ in range(n_pred_tokens)
    ]

    # Create dummy labels
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Compute loss
    total_loss, loss_dict = loss_fn(logits_list, labels)

    print(f"\n  Total loss: {total_loss.item():.4f}")
    print(f"  Individual losses:")
    for key, value in loss_dict.items():
        if key != 'total_multitoken_loss':
            print(f"    {key}: {value:.4f}")

    # Test backward
    total_loss.backward()
    print("\n  ✓ Backward pass successful")

    # Test with different sequence lengths
    print("\n  Testing with seq_len=32...")
    seq_len_long = 32
    logits_list_long = [
        torch.randn(batch_size, seq_len_long, vocab_size)
        for _ in range(n_pred_tokens)
    ]
    labels_long = torch.randint(0, vocab_size, (batch_size, seq_len_long))

    total_loss_long, loss_dict_long = loss_fn(logits_list_long, labels_long)
    print(f"  Total loss: {total_loss_long.item():.4f}")

    # Verify loss is positive and reasonable
    assert total_loss > 0, "Loss should be positive"
    assert total_loss < 100, "Loss should be reasonable (< 100)"

    print("\n✓ All multi-token prediction loss tests passed!")
