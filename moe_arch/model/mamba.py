"""
Mamba SSM (State Space Model) blocks with Routing Mamba (RoM) support.

Implements:
1. Basic Mamba block (wrapper around mamba-ssm library)
2. Routing Mamba (RoM) - MoE-style routing for Mamba blocks
3. Shared router for Mamba experts

Reference:
- Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- RoM: Routing Mamba paper (shared router across Mamba experts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .config import AdvancedMoEConfig


class MambaBlock(nn.Module):
    """
    Mamba block - efficient state space model for sequence modeling.

    Uses the mamba-ssm library for CUDA-optimized implementation.
    Falls back to simplified version if library not available.
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.ssm_state_dim
        self.d_conv = 4  # Convolution width
        self.expand = config.ssm_expansion

        # Try to use mamba-ssm library
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=config.d_model,
                d_state=config.ssm_state_dim,
                d_conv=self.d_conv,
                expand=config.ssm_expansion,
            )
            self.using_mamba_ssm = True
        except ImportError:
            # Fallback: simple SSM implementation
            print("Warning: mamba-ssm not available, using simplified SSM")
            self.mamba = self._create_simple_ssm()
            self.using_mamba_ssm = False

    def _create_simple_ssm(self) -> nn.Module:
        """Create a simplified SSM as fallback."""
        # This is a simplified version for testing
        # In production, you should install mamba-ssm
        d_inner = self.expand * self.d_model

        class SimpleSSM(nn.Module):
            def __init__(self, d_model, d_inner):
                super().__init__()
                self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
                self.conv1d = nn.Conv1d(
                    d_inner, d_inner,
                    kernel_size=4,
                    padding=3,
                    groups=d_inner,
                )
                self.out_proj = nn.Linear(d_inner, d_model, bias=False)
                self.act = nn.SiLU()

            def forward(self, x):
                # x: (batch, seq, d_model)
                z, x = self.in_proj(x).chunk(2, dim=-1)

                # Conv1d expects (batch, channels, seq)
                x = x.transpose(1, 2)
                x = self.conv1d(x)[:, :, :x.shape[-1]]  # Truncate to match input
                x = x.transpose(1, 2)

                x = self.act(x)
                x = x * F.silu(z)
                return self.out_proj(x)

        return SimpleSSM(self.d_model, d_inner)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        return self.mamba(hidden_states)


class MambaExpert(nn.Module):
    """
    Single Mamba expert for Routing Mamba (RoM).

    Each expert is a Mamba block that can specialize in different
    sequence modeling patterns.
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.mamba = MambaBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba expert.

        Args:
            x: (batch, seq_len, d_model) or (n_tokens, d_model)

        Returns:
            output: same shape as input
        """
        return self.mamba(x)


class RoutingMamba(nn.Module):
    """
    Routing Mamba (RoM) - MoE-style routing for Mamba blocks.

    Key difference from standard MoE:
    - Uses Mamba blocks as experts instead of FFNs
    - Shared router across all Mamba experts
    - Designed for efficient sequence modeling with specialization

    Reference: RoM paper - routing allows different Mamba experts to specialize
    in different temporal patterns (short-range vs long-range dependencies, etc.)
    """

    def __init__(self, config: AdvancedMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k
        self.d_model = config.d_model

        # Create Mamba experts
        self.experts = nn.ModuleList([
            MambaExpert(config) for _ in range(config.n_experts)
        ])

        # Shared router (same as MoE router)
        # d_model -> hidden -> n_experts
        hidden_dim = 128
        self.router = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.n_experts, bias=False),
        )

        # Loss weights
        self.load_balance_loss_weight = config.moe_load_balance_loss_weight
        self.router_z_loss_weight = config.moe_router_z_loss_weight

        # Track auxiliary loss
        self.aux_loss = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Routing Mamba.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Route tokens to Mamba experts
        routing_weights, selected_experts, router_logits = self._route(hidden_states)
        # routing_weights: (batch, seq, top_k)
        # selected_experts: (batch, seq, top_k)
        # router_logits: (batch, seq, n_experts)

        # Process through Mamba experts (maintain sequence structure)
        output = self._process_experts(
            hidden_states,
            routing_weights,
            selected_experts,
        )

        # Compute auxiliary losses
        if self.training:
            self.aux_loss = self._compute_auxiliary_losses(
                router_logits,
                selected_experts,
            )
        else:
            self.aux_loss = 0.0

        return output

    def _route(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing for Mamba experts.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            routing_weights: (batch, seq_len, top_k)
            selected_experts: (batch, seq_len, top_k)
            router_logits: (batch, seq_len, n_experts)
        """
        # Compute router logits
        router_logits = self.router(hidden_states)  # (batch, seq, n_experts)

        # Apply softmax
        router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq, n_experts)

        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # Each: (batch, seq, top_k)

        # Normalize weights to sum to 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts, router_logits

    def _process_experts(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process sequences through selected Mamba experts.

        CRITICAL: Each expert processes the FULL sequence to maintain temporal
        context. This is essential for Mamba's state space modeling capability.

        Different from standard MoE where tokens are processed independently,
        here we:
        1. Pass full sequences through each of the top-k experts
        2. Weight expert outputs per-token according to routing weights
        3. Combine weighted outputs

        This maintains:
        - Sequence context (experts see full sequences)
        - Token-level specialization (different weights per token)
        - Computational efficiency (only top-k experts activated)

        Args:
            hidden_states: (batch, seq_len, d_model)
            routing_weights: (batch, seq_len, top_k)
            selected_experts: (batch, seq_len, top_k)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        # Find all unique experts selected across the batch
        unique_experts = torch.unique(selected_experts)

        # Process each selected expert
        for expert_idx in unique_experts:
            # Check which tokens selected this expert
            expert_mask = (selected_experts == expert_idx)  # (batch, seq, top_k)

            # If no tokens selected this expert, skip
            if not expert_mask.any():
                continue

            # Process the full sequence through this expert
            # All sequences in the batch go through together
            expert_output = self.experts[expert_idx](hidden_states)  # (batch, seq, d_model)

            # For each position in the batch/sequence that selected this expert,
            # add the weighted contribution
            for k in range(self.top_k):
                # Mask for positions where this expert was selected at position k
                mask_k = (selected_experts[:, :, k] == expert_idx)  # (batch, seq)

                if not mask_k.any():
                    continue

                # Get routing weights for these positions
                weights = routing_weights[:, :, k]  # (batch, seq)

                # Apply mask and weights
                # (batch, seq, 1) * (batch, seq, d_model)
                weighted_output = (mask_k.unsqueeze(-1) * weights.unsqueeze(-1)) * expert_output

                # Accumulate
                output = output + weighted_output

        return output

    def _compute_auxiliary_losses(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary losses for load balancing.

        Args:
            router_logits: (batch, seq, n_experts)
            selected_experts: (batch, seq, top_k)

        Returns:
            total_aux_loss: scalar
        """
        # Load balancing loss
        load_balance_loss = self._compute_load_balance_loss(
            router_logits,
            selected_experts,
        )

        # Router z-loss
        router_z_loss = self._compute_router_z_loss(router_logits)

        # Combine
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
        """Load balancing loss (same as MoE)."""
        router_probs = F.softmax(router_logits, dim=-1)
        P = router_probs.mean(dim=[0, 1])

        batch_size, seq_len, top_k = selected_experts.shape
        total_tokens = batch_size * seq_len * top_k

        expert_mask = F.one_hot(selected_experts, num_classes=self.n_experts).float()
        expert_counts = expert_mask.sum(dim=[0, 1, 2])
        f = expert_counts / total_tokens

        loss = self.n_experts * (f * P).sum()
        return loss

    def _compute_router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Router z-loss (same as MoE)."""
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        return z_loss


if __name__ == "__main__":
    # Test Mamba blocks
    from .config import get_test_config

    config = get_test_config()
    print("Testing Mamba blocks...")
    print(f"  d_model={config.d_model}, ssm_state_dim={config.ssm_state_dim}")
    print(f"  ssm_expansion={config.ssm_expansion}")

    # Test basic Mamba block
    print("\n1. Testing basic Mamba block...")
    mamba_block = MambaBlock(config)
    print(f"   Using mamba-ssm: {mamba_block.using_mamba_ssm}")
    print(f"   Parameters: {sum(p.numel() for p in mamba_block.parameters()):,}")

    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)

    output = mamba_block(hidden_states)
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == hidden_states.shape
    print("   ✓ Basic Mamba block works")

    # Test backward
    loss = output.sum()
    loss.backward()
    print("   ✓ Backward pass works")

    # Test Routing Mamba
    print("\n2. Testing Routing Mamba (RoM)...")
    rom = RoutingMamba(config, layer_idx=0)
    print(f"   Experts: {config.n_experts}, Top-k: {config.moe_top_k}")
    print(f"   Parameters: {sum(p.numel() for p in rom.parameters()):,}")

    rom.train()
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)
    output = rom(hidden_states)

    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Auxiliary loss: {rom.aux_loss.item():.6f}")
    assert output.shape == hidden_states.shape
    print("   ✓ Routing Mamba works")

    # Test backward
    loss = output.sum() + rom.aux_loss
    loss.backward()
    print("   ✓ Backward pass works")

    # Test routing distribution
    print("\n3. Testing routing distribution...")
    with torch.no_grad():
        routing_weights, selected_experts, _ = rom._route(hidden_states)

    expert_counts = torch.zeros(config.n_experts)
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(config.moe_top_k):
                expert_idx = selected_experts[i, j, k].item()
                expert_counts[expert_idx] += 1

    expert_usage = expert_counts / expert_counts.sum()
    print(f"   Expert usage: {expert_usage.tolist()}")
    print(f"   Min: {expert_usage.min().item():.3f}, Max: {expert_usage.max().item():.3f}")

    print("\n✓ All Mamba tests passed!")
