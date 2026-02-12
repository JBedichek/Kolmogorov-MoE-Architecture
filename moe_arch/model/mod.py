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

    Implementation follows the paper:
    "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
    (Raposo et al., 2024)

    Key design choices from paper:
    - Router is a simple linear projection (not MLP)
    - Output is multiplied by softmax of router weights (puts router in gradient path)
    - Auxiliary loss uses binary cross-entropy for causality fix

    Benefits:
    - Adaptive depth: important tokens get more processing
    - Compute savings: ~87.5% reduction with capacity_factor=0.125 (paper's optimal)
    - Complements MoE: MoD handles "which tokens", MoE handles "which experts"
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.capacity_factor = config.mod_capacity_factor

        # Router: simple linear projection (as per paper)
        # "the router weight for a given token embedding is a scalar produced
        #  as a result of a linear projection r_i = w^T x_i"
        self.router = nn.Linear(config.d_model, 1, bias=False)

        # Auxiliary router for causality fix (small MLP)
        # Used during training to predict which tokens will be selected
        # This helps with autoregressive sampling where we can't use top-k
        aux_hidden = config.mod_router_hidden_dim
        self.aux_router = nn.Sequential(
            nn.Linear(config.d_model, aux_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(aux_hidden, 1, bias=False),
        )

        # Loss weights
        self.aux_loss_weight = config.mod_load_balance_loss_weight

        # Track auxiliary loss
        self.aux_loss = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute which tokens should be processed through the layer.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            selected_indices: (batch, k) - indices of selected tokens (sorted for causality)
            router_weights: (batch, k) - softmax weights for selected tokens (for gradient flow)
            router_logits: (batch, seq_len) - raw router scores (for aux loss)
            topk_indices: (batch, k) - original unsorted indices (for aux loss target)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Handle empty sequences (can happen with padding)
        if seq_len == 0:
            empty_indices = torch.zeros(batch_size, 0, device=hidden_states.device, dtype=torch.long)
            empty_weights = torch.zeros(batch_size, 0, device=hidden_states.device, dtype=hidden_states.dtype)
            empty_logits = torch.zeros(batch_size, 0, device=hidden_states.device, dtype=hidden_states.dtype)
            return empty_indices, empty_weights, empty_logits, empty_indices

        # Compute router logits (scalar per token)
        router_logits = self.router(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Determine number of tokens to select per sequence
        # Clamp k to never exceed seq_len (handles short/padded sequences)
        k = max(1, min(seq_len, int(seq_len * self.capacity_factor)))

        # Select top-k tokens per sequence
        topk_logits, topk_indices = torch.topk(router_logits, k, dim=-1)  # Each: (batch, k)

        # Sort selected indices for causal consistency
        # (ensures tokens are processed in order, important for attention)
        sorted_indices, sort_order = torch.sort(topk_indices, dim=-1)

        # Reorder the logits according to sorted indices
        sorted_logits = torch.gather(topk_logits, dim=-1, index=sort_order)

        # Apply softmax to get router weights (as per paper)
        # "we multiply the output of the function f by the router weights"
        # This puts router weights in the gradient path
        router_weights = F.softmax(sorted_logits, dim=-1)  # (batch, k)

        # Compute auxiliary loss for causality fix
        if self.training:
            self.aux_loss = self._compute_auxiliary_loss(
                hidden_states, router_logits, topk_indices
            )
        else:
            self.aux_loss = 0.0

        return sorted_indices, router_weights, router_logits, topk_indices

    def _compute_auxiliary_loss(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary loss for causality fix (as per paper Section 3.5).

        Uses a small auxiliary MLP to predict whether each token will be selected.
        This is trained with binary cross-entropy where:
        - Targets: 1 if token was among top-k, 0 otherwise
        - Predictions: auxiliary router output

        The auxiliary router receives inputs with stop_gradient to avoid
        interfering with the main routing decisions.

        Args:
            hidden_states: (batch, seq_len, d_model) - input embeddings
            router_logits: (batch, seq_len) - main router outputs (unused here)
            topk_indices: (batch, k) - indices of selected tokens

        Returns:
            loss: scalar (differentiable w.r.t. auxiliary router)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        k = topk_indices.shape[1]

        # Create binary targets: 1 for selected tokens, 0 otherwise
        targets = torch.zeros(
            batch_size, seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        targets.scatter_(1, topk_indices, 1.0)

        # Auxiliary router predicts selection (with stop gradient on input)
        # "receives the same inputs as the router (with a stop gradient)"
        aux_logits = self.aux_router(hidden_states.detach()).squeeze(-1)  # (batch, seq_len)

        # Binary cross-entropy loss
        # "a binary cross-entropy loss wherein the router's outputs provide the logits,
        #  and the top-k selections provide the targets"
        aux_loss = F.binary_cross_entropy_with_logits(
            aux_logits, targets, reduction='mean'
        )

        return self.aux_loss_weight * aux_loss


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
