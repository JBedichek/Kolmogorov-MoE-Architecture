"""
Mixture of Depths (MoD) - Conditional layer execution.

Implements learned routing that determines which tokens should be processed
through each layer, allowing adaptive computation depth per token.

Reference: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" (Google, 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import AdvancedMoEConfig


class MoDRouter(nn.Module):
    """
    Mixture of Depths router.

    Learns to score each token and selects top-k% for processing.
    Tokens not selected skip the layer computation (residual only).

    Benefits:
    - Adaptive depth: important tokens get more processing
    - Compute savings: ~25% reduction with capacity_factor=0.75
    - Complements MoE: MoD handles "which tokens", MoE handles "which experts"
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.capacity_factor = config.mod_capacity_factor
        self.hidden_dim = config.mod_router_hidden_dim

        # Scoring network: d_model -> hidden -> 1
        self.scorer = nn.Sequential(
            nn.Linear(config.d_model, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )

        # Load balancing loss weight
        self.load_balance_loss_weight = config.mod_load_balance_loss_weight

        # Track auxiliary loss
        self.aux_loss = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute which tokens should be processed through the layer.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            selected_mask: (batch, seq_len) - binary mask of selected tokens
            selected_indices: (batch, k) - indices of selected tokens (for gathering)
            scores: (batch, seq_len) - token importance scores (for loss)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Compute importance scores for each token
        scores = self.scorer(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Determine number of tokens to select per sequence
        k = max(1, int(seq_len * self.capacity_factor))

        # Select top-k tokens per sequence
        topk_scores, topk_indices = torch.topk(scores, k, dim=-1)  # Each: (batch, k)

        # Create binary mask for selected tokens
        selected_mask = torch.zeros(
            batch_size, seq_len,
            device=hidden_states.device,
            dtype=torch.bool,
        )
        selected_mask.scatter_(1, topk_indices, True)

        # Compute auxiliary loss
        if self.training:
            self.aux_loss = self._compute_load_balance_loss(scores, selected_mask)
        else:
            self.aux_loss = 0.0

        return selected_mask, topk_indices, scores

    def _compute_load_balance_loss(
        self,
        scores: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Load balancing loss to encourage diverse token selection.

        Encourages the router to select different tokens across the sequence,
        preventing it from always selecting the same positions.

        Args:
            scores: (batch, seq_len) - token importance scores
            selected_mask: (batch, seq_len) - binary mask of selected tokens

        Returns:
            loss: scalar
        """
        # Compute selection probability per position (averaged across batch)
        # This encourages the router to not always select the same positions
        selection_prob = selected_mask.float().mean(dim=0)  # (seq_len,)

        # Ideal uniform distribution
        target_prob = self.capacity_factor

        # L2 loss: encourage uniform selection across positions
        # This prevents the router from always selecting beginning/end tokens
        loss = ((selection_prob - target_prob) ** 2).mean()

        return self.load_balance_loss_weight * loss


class MoDLayer(nn.Module):
    """
    Mixture of Depths layer wrapper.

    Wraps any layer (attention, FFN, etc.) with MoD routing.
    Selected tokens are processed through the layer, others skip via residual.
    """

    def __init__(
        self,
        config: AdvancedMoEConfig,
        layer_module: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.layer = layer_module
        self.router = MoDRouter(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **layer_kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with conditional execution.

        Args:
            hidden_states: (batch, seq_len, d_model)
            **layer_kwargs: Additional arguments for the wrapped layer

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Route tokens
        selected_mask, selected_indices, scores = self.router(hidden_states)
        # selected_mask: (batch, seq_len)
        # selected_indices: (batch, k)
        # scores: (batch, seq_len)

        # Gather selected tokens
        # selected_indices: (batch, k) -> (batch, k, 1) -> (batch, k, d_model)
        selected_tokens = torch.gather(
            hidden_states,
            1,
            selected_indices.unsqueeze(-1).expand(-1, -1, d_model),
        )  # (batch, k, d_model)

        # Process selected tokens through the layer
        processed_tokens = self.layer(selected_tokens, **layer_kwargs)  # (batch, k, d_model)

        # Scatter processed tokens back to their positions
        output = hidden_states.clone()  # Start with input (residual for non-selected)
        output.scatter_(
            1,
            selected_indices.unsqueeze(-1).expand(-1, -1, d_model),
            processed_tokens,
        )

        return output


if __name__ == "__main__":
    # Test MoD router
    from .config import get_test_config

    config = get_test_config()
    print("Testing Mixture of Depths (MoD)...")
    print(f"  d_model={config.d_model}, capacity_factor={config.mod_capacity_factor}")

    # Test MoD router
    print("\n1. Testing MoD Router...")
    router = MoDRouter(config)
    print(f"   Parameters: {sum(p.numel() for p in router.parameters()):,}")

    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)

    router.train()
    selected_mask, selected_indices, scores = router(hidden_states)

    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Selected mask shape: {selected_mask.shape}")
    print(f"   Selected indices shape: {selected_indices.shape}")
    print(f"   Scores shape: {scores.shape}")

    # Check selection count
    k = int(seq_len * config.mod_capacity_factor)
    actual_counts = selected_mask.sum(dim=1)
    assert (actual_counts == k).all(), f"Selection count mismatch: got {actual_counts.tolist()}, expected {k}"
    print(f"   ✓ Selected {k}/{seq_len} tokens ({100*config.mod_capacity_factor:.0f}%)")

    # Check aux loss
    print(f"   Auxiliary loss: {router.aux_loss.item():.6f}")
    assert router.aux_loss > 0, "Should have aux loss in training mode"
    print("   ✓ MoD router works")

    # Test backward
    loss = scores.sum() + router.aux_loss
    loss.backward()
    print("   ✓ Backward pass works")

    # Test MoD layer wrapper
    print("\n2. Testing MoD Layer Wrapper...")

    # Create a simple test layer (just a linear layer)
    test_layer = nn.Linear(config.d_model, config.d_model)
    mod_layer = MoDLayer(config, test_layer)

    print(f"   MoD layer parameters: {sum(p.numel() for p in mod_layer.parameters()):,}")

    mod_layer.train()
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)
    output = mod_layer(hidden_states)

    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == hidden_states.shape
    print("   ✓ MoD layer wrapper works")

    # Check that non-selected tokens are unchanged (residual)
    selected_mask, _, _ = mod_layer.router(hidden_states)
    non_selected_mask = ~selected_mask

    # For non-selected tokens, output should equal input
    for i in range(batch_size):
        for j in range(seq_len):
            if non_selected_mask[i, j]:
                # This token was not selected, should be unchanged
                diff = (output[i, j] - hidden_states[i, j]).abs().max()
                assert diff < 1e-5, f"Non-selected token changed: diff={diff}"

    print("   ✓ Non-selected tokens use residual connection")

    # Test backward
    loss = output.sum() + mod_layer.router.aux_loss
    loss.backward()
    print("   ✓ Backward pass works")

    # Test selection distribution
    print("\n3. Testing selection distribution...")
    n_batches = 100
    position_selection_count = torch.zeros(seq_len)

    router.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            hidden_states = torch.randn(batch_size, seq_len, config.d_model)
            selected_mask, _, _ = router(hidden_states)

            # Count selections per position
            position_selection_count += selected_mask.sum(dim=0).float()

    position_selection_prob = position_selection_count / (n_batches * batch_size)

    print(f"   Selection probability per position:")
    print(f"     Min: {position_selection_prob.min().item():.3f}")
    print(f"     Max: {position_selection_prob.max().item():.3f}")
    print(f"     Mean: {position_selection_prob.mean().item():.3f}")
    print(f"     Target: {config.mod_capacity_factor:.3f}")
    print(f"     Std: {position_selection_prob.std().item():.3f}")

    print("\n✓ All MoD tests passed!")
