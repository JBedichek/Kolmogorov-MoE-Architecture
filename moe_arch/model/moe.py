"""
Mixture of Experts (MoE) implementation with load balancing.

Features:
- Expert FFN modules
- Softmax router with top-k selection
- Load balancing auxiliary loss (Switch Transformer)
- Router z-loss for stability
- Grouped GEMM optimization for efficient expert computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math

from .config import AdvancedMoEConfig
from .ffn import ExpertFFN


class Expert(nn.Module):
    """
    Single expert module (FFN).

    Each expert is a simple feed-forward network that can specialize
    in processing certain types of tokens.
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.expert_ffn = ExpertFFN(
            config.d_model,
            config.d_ff_expert,
            config.ffn_activation,
            config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.

        Args:
            x: (n_tokens, d_model)

        Returns:
            output: (n_tokens, d_model)
        """
        return self.expert_ffn(x)


class Router(nn.Module):
    """
    Learned router that determines which tokens go to which experts.

    Uses a simple MLP to produce routing scores, then applies top-k selection.
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k

        # Router network: d_model -> hidden -> n_experts
        hidden_dim = 128
        self.router = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.n_experts, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing probabilities for each token.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            routing_weights: (batch, seq_len, top_k) - weights for selected experts
            selected_experts: (batch, seq_len, top_k) - indices of selected experts
            router_logits: (batch, seq_len, n_experts) - raw router scores (for losses)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Compute router logits
        router_logits = self.router(hidden_states)  # (batch, seq, n_experts)

        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq, n_experts)

        # Select top-k experts per token
        routing_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # Each: (batch, seq, top_k)

        # Normalize routing weights to sum to 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts, router_logits


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with load balancing.

    Architecture:
    1. Router selects top-k experts for each token
    2. Tokens are dispatched to selected experts
    3. Expert outputs are weighted and combined
    4. Load balancing losses encourage uniform expert utilization
    """

    def __init__(self, config: AdvancedMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k
        self.capacity_factor = config.moe_capacity_factor

        # Create experts
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.n_experts)
        ])

        # Create router
        self.router = Router(config)

        # Loss weights
        self.load_balance_loss_weight = config.moe_load_balance_loss_weight
        self.router_z_loss_weight = config.moe_router_z_loss_weight

        # Track auxiliary losses
        self.aux_loss = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Route tokens to experts
        routing_weights, selected_experts, router_logits = self.router(hidden_states)
        # routing_weights: (batch, seq, top_k)
        # selected_experts: (batch, seq, top_k)
        # router_logits: (batch, seq, n_experts)

        # Reshape for expert processing
        # Flatten batch and sequence dimensions
        hidden_states_flat = hidden_states.view(-1, d_model)  # (batch*seq, d_model)
        routing_weights_flat = routing_weights.view(-1, self.top_k)  # (batch*seq, top_k)
        selected_experts_flat = selected_experts.view(-1, self.top_k)  # (batch*seq, top_k)

        # Process through experts
        if self.training and self.config.use_gradient_checkpointing:
            # Use efficient grouped computation
            output_flat = self._grouped_expert_forward(
                hidden_states_flat,
                routing_weights_flat,
                selected_experts_flat,
            )
        else:
            # Simple loop-based implementation (easier to understand)
            output_flat = self._simple_expert_forward(
                hidden_states_flat,
                routing_weights_flat,
                selected_experts_flat,
            )

        # Reshape back
        output = output_flat.view(batch_size, seq_len, d_model)

        # Compute auxiliary losses
        if self.training:
            self.aux_loss = self._compute_auxiliary_losses(
                router_logits,
                selected_experts,
            )
        else:
            self.aux_loss = 0.0

        return output

    def _simple_expert_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simple loop-based expert forward pass.

        Args:
            hidden_states: (n_tokens, d_model)
            routing_weights: (n_tokens, top_k)
            selected_experts: (n_tokens, top_k)

        Returns:
            output: (n_tokens, d_model)
        """
        n_tokens, d_model = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        # Process each token
        for i in range(n_tokens):
            token_output = torch.zeros(d_model, device=hidden_states.device, dtype=hidden_states.dtype)

            # Process through top-k experts
            for k in range(self.top_k):
                expert_idx = selected_experts[i, k].item()
                weight = routing_weights[i, k]

                # Forward through expert
                expert_output = self.experts[expert_idx](hidden_states[i:i+1])
                token_output += weight * expert_output.squeeze(0)

            output[i] = token_output

        return output

    def _grouped_expert_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Grouped GEMM-style expert forward pass (more efficient).

        Groups tokens by expert assignment and processes them in batches.

        Args:
            hidden_states: (n_tokens, d_model)
            routing_weights: (n_tokens, top_k)
            selected_experts: (n_tokens, top_k)

        Returns:
            output: (n_tokens, d_model)
        """
        n_tokens, d_model = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        # For each expert, gather tokens assigned to it
        for expert_idx in range(self.n_experts):
            # Find all tokens assigned to this expert
            expert_mask = (selected_experts == expert_idx)  # (n_tokens, top_k)
            token_indices, k_indices = torch.where(expert_mask)

            if len(token_indices) == 0:
                continue  # No tokens for this expert

            # Gather tokens for this expert
            expert_inputs = hidden_states[token_indices]  # (n_assigned, d_model)

            # Process through expert (batched)
            expert_outputs = self.experts[expert_idx](expert_inputs)  # (n_assigned, d_model)

            # Get weights for these tokens
            weights = routing_weights[token_indices, k_indices].unsqueeze(1)  # (n_assigned, 1)

            # Accumulate weighted outputs
            output.index_add_(0, token_indices, weights * expert_outputs)

        return output

    def _compute_auxiliary_losses(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary losses for load balancing and stability.

        Args:
            router_logits: (batch, seq, n_experts) - raw router scores
            selected_experts: (batch, seq, top_k) - selected expert indices

        Returns:
            total_aux_loss: scalar tensor
        """
        # Load balancing loss (Switch Transformer style)
        load_balance_loss = self._compute_load_balance_loss(
            router_logits,
            selected_experts,
        )

        # Router z-loss (penalize large logits for stability)
        router_z_loss = self._compute_router_z_loss(router_logits)

        # Combine losses
        total_aux_loss = (
            self.load_balance_loss_weight * load_balance_loss +
            self.router_z_loss_weight * router_z_loss
        )

        return total_aux_loss

    def _compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Load balancing loss to encourage uniform expert utilization.

        Loss = n_experts * sum_i(f_i * P_i)

        where:
        - f_i = fraction of tokens assigned to expert i
        - P_i = average router probability for expert i

        Minimizing this encourages uniform distribution.

        Args:
            router_logits: (batch, seq, n_experts)
            selected_experts: (batch, seq, top_k)

        Returns:
            loss: scalar
        """
        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq, n_experts)

        # Average router probability per expert (P_i)
        P = router_probs.mean(dim=[0, 1])  # (n_experts,)

        # Fraction of tokens assigned to each expert (f_i)
        batch_size, seq_len, top_k = selected_experts.shape
        total_tokens = batch_size * seq_len * top_k

        # Create one-hot encoding of expert assignments
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.n_experts
        ).float()  # (batch, seq, top_k, n_experts)

        # Sum over batch, seq, top_k to get counts per expert
        expert_counts = expert_mask.sum(dim=[0, 1, 2])  # (n_experts,)
        f = expert_counts / total_tokens  # (n_experts,)

        # Load balance loss
        loss = self.n_experts * (f * P).sum()

        return loss

    def _compute_router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Router z-loss to penalize large logits (improves stability).

        Loss = sum(log(sum(exp(logits)))^2)

        This encourages the router to not be overconfident.

        Args:
            router_logits: (batch, seq, n_experts)

        Returns:
            loss: scalar
        """
        # Compute log-sum-exp
        log_z = torch.logsumexp(router_logits, dim=-1)  # (batch, seq)

        # Square and average
        z_loss = (log_z ** 2).mean()

        return z_loss

    def get_expert_utilization(self) -> Dict[str, float]:
        """
        Get statistics about expert utilization (for monitoring).

        Returns:
            Dictionary with utilization statistics
        """
        # This would be called during evaluation to monitor expert usage
        # For now, return empty dict (will be populated during actual training)
        return {}


if __name__ == "__main__":
    # Test MoE layer
    from .config import get_test_config

    config = get_test_config()
    print(f"Testing MoE layer with config:")
    print(f"  d_model={config.d_model}, n_experts={config.n_experts}, top_k={config.moe_top_k}")

    # Create MoE layer
    moe_layer = MoELayer(config, layer_idx=0)
    print(f"  Parameters: {sum(p.numel() for p in moe_layer.parameters()):,}")

    # Count expert vs router parameters
    expert_params = sum(p.numel() for expert in moe_layer.experts for p in expert.parameters())
    router_params = sum(p.numel() for p in moe_layer.router.parameters())
    print(f"  Expert params: {expert_params:,} ({100*expert_params/(expert_params+router_params):.1f}%)")
    print(f"  Router params: {router_params:,} ({100*router_params/(expert_params+router_params):.1f}%)")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)

    print(f"\n1. Testing forward pass...")
    print(f"   Input shape: {hidden_states.shape}")

    output = moe_layer(hidden_states)
    print(f"   Output shape: {output.shape}")
    assert output.shape == hidden_states.shape
    print(f"   Auxiliary loss: {moe_layer.aux_loss.item():.6f}")
    print("   ✓ Forward pass successful")

    # Test backward pass
    print(f"\n2. Testing backward pass...")
    loss = output.sum() + moe_layer.aux_loss
    loss.backward()

    # Check gradients
    grad_count = sum(1 for p in moe_layer.parameters() if p.grad is not None)
    total_params = sum(1 for p in moe_layer.parameters())
    print(f"   Parameters with gradients: {grad_count}/{total_params}")
    print("   ✓ Backward pass successful")

    # Test routing behavior
    print(f"\n3. Testing routing behavior...")
    with torch.no_grad():
        routing_weights, selected_experts, router_logits = moe_layer.router(hidden_states)

    print(f"   Routing weights shape: {routing_weights.shape}")
    print(f"   Selected experts shape: {selected_experts.shape}")
    print(f"   Router logits shape: {router_logits.shape}")
    print(f"   Sample routing weights (first token): {routing_weights[0, 0].tolist()}")
    print(f"   Sample selected experts (first token): {selected_experts[0, 0].tolist()}")
    print("   ✓ Routing behavior verified")

    # Test expert utilization
    print(f"\n4. Testing expert utilization distribution...")
    expert_counts = torch.zeros(config.n_experts)
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(config.moe_top_k):
                expert_idx = selected_experts[i, j, k].item()
                expert_counts[expert_idx] += 1

    expert_usage = expert_counts / expert_counts.sum()
    print(f"   Expert usage distribution: {expert_usage.tolist()}")
    print(f"   Min usage: {expert_usage.min().item():.3f}, Max usage: {expert_usage.max().item():.3f}")
    print(f"   Std dev: {expert_usage.std().item():.3f} (lower is more balanced)")

    # Test with different batch sizes
    print(f"\n5. Testing with different batch sizes...")
    for test_batch_size in [1, 4, 8]:
        test_input = torch.randn(test_batch_size, seq_len, config.d_model)
        test_output = moe_layer(test_input)
        assert test_output.shape == test_input.shape
        print(f"   Batch size {test_batch_size}: ✓")

    print("\nAll MoE tests passed!")
