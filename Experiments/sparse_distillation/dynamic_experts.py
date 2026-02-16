"""
Dynamic Expert Reallocation

Tracks per-expert loss and reallocates parameters:
- High-loss experts get more dimensions (grow d_ff)
- Low-loss experts get fewer dimensions (shrink d_ff)

Weight matrices are interpolated to preserve the learned function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class ExpertStats:
    """Track per-expert statistics for reallocation decisions."""
    expert_idx: int
    d_ff: int  # Current expert dimension
    total_loss: float = 0.0
    token_count: int = 0

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(self.token_count, 1)


@dataclass
class DynamicExpertConfig:
    """Configuration for dynamic expert reallocation."""
    # Reallocation frequency
    reallocate_every_n_steps: int = 500

    # How much to grow/shrink
    growth_factor: float = 1.25  # Grow by 25%
    shrink_factor: float = 0.8   # Shrink by 20%

    # Constraints
    min_d_ff: int = 256  # Minimum expert dimension
    max_d_ff: int = 8192  # Maximum expert dimension

    # Which experts to adjust (top/bottom k by loss)
    top_k_grow: int = 2   # Grow top-k highest loss experts
    top_k_shrink: int = 2  # Shrink top-k lowest loss experts

    # Total parameter budget (if set, maintains constant total params)
    maintain_param_budget: bool = True


class DynamicExpertLayer(nn.Module):
    """
    Expert layer with variable-sized experts.

    Each expert can have a different d_ff dimension.
    """

    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_ff_per_expert: List[int],  # Different d_ff for each expert
        activation: str = "swiglu",
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_ff_per_expert = list(d_ff_per_expert)
        self.activation = activation

        # Create separate weight tensors for each expert (variable sizes)
        self.experts_w1 = nn.ParameterList()
        self.experts_w2 = nn.ParameterList()
        self.experts_w3 = nn.ParameterList()

        for expert_idx in range(n_experts):
            d_ff = self.d_ff_per_expert[expert_idx]
            # SwiGLU: gate (w1), up (w2), down (w3)
            self.experts_w1.append(nn.Parameter(torch.empty(d_model, d_ff)))
            self.experts_w2.append(nn.Parameter(torch.empty(d_model, d_ff)))
            self.experts_w3.append(nn.Parameter(torch.empty(d_ff, d_model)))

        self._init_weights()

    def _init_weights(self):
        for expert_idx in range(self.n_experts):
            for w in [self.experts_w1[expert_idx], self.experts_w2[expert_idx], self.experts_w3[expert_idx]]:
                fan_in = w.shape[0]
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(w, std=std)

    def forward_expert(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Forward pass through a single expert."""
        w1 = self.experts_w1[expert_idx]
        w2 = self.experts_w2[expert_idx]
        w3 = self.experts_w3[expert_idx]

        # SwiGLU: (silu(x @ w1) * (x @ w2)) @ w3
        gate = F.silu(x @ w1)
        up = x @ w2
        hidden = gate * up
        output = hidden @ w3
        return output

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass with per-expert loss tracking.

        Returns:
            output: Combined output
            expert_outputs: Dict mapping expert_idx to its output (for loss tracking)
        """
        batch_size, seq_len, d_model = x.shape
        _, n_experts, capacity = token_indices.shape
        device = x.device
        dtype = x.dtype

        output = torch.zeros(batch_size, seq_len, d_model, device=device, dtype=dtype)
        expert_outputs = {}

        for expert_idx in range(self.n_experts):
            # Get tokens for this expert
            indices = token_indices[:, expert_idx, :]  # (batch, capacity)
            weights = expert_weights[:, expert_idx, :]  # (batch, capacity)

            # Gather tokens
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d_model)
            expert_input = torch.gather(x, 1, indices_expanded)  # (batch, capacity, d_model)

            # Forward through expert
            expert_out = self.forward_expert(expert_input, expert_idx)  # (batch, capacity, d_model)
            expert_outputs[expert_idx] = expert_out

            # Apply weights and scatter back
            weighted_out = expert_out * weights.unsqueeze(-1)
            output.scatter_add_(1, indices_expanded, weighted_out)

        return output, expert_outputs

    def get_expert_params(self, expert_idx: int) -> int:
        """Count parameters for a specific expert."""
        d_ff = self.d_ff_per_expert[expert_idx]
        # w1: (d_model, d_ff), w2: (d_model, d_ff), w3: (d_ff, d_model)
        return 3 * self.d_model * d_ff

    def get_total_params(self) -> int:
        """Total parameters across all experts."""
        return sum(self.get_expert_params(i) for i in range(self.n_experts))

    def get_avg_active_params(self, top_k: int = 2) -> float:
        """Average active parameters when top_k experts are used per token."""
        # For expert-choice, each expert is equally active
        # Active params = sum of top_k largest experts (worst case) or average
        sorted_params = sorted([self.get_expert_params(i) for i in range(self.n_experts)], reverse=True)
        return sum(sorted_params[:top_k])


def interpolate_expert_weights(
    old_w1: torch.Tensor,  # (d_model, old_d_ff)
    old_w2: torch.Tensor,  # (d_model, old_d_ff)
    old_w3: torch.Tensor,  # (old_d_ff, d_model)
    new_d_ff: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Interpolate expert weights to a new d_ff dimension.

    For growing: Use linear interpolation to expand
    For shrinking: Use importance-based compression

    The goal is to preserve the function as much as possible.
    """
    d_model = old_w1.shape[0]
    old_d_ff = old_w1.shape[1]
    device = old_w1.device
    dtype = old_w1.dtype

    if new_d_ff == old_d_ff:
        return old_w1.clone(), old_w2.clone(), old_w3.clone()

    if new_d_ff > old_d_ff:
        # Growing: interpolate and pad
        # Use linear interpolation along d_ff dimension

        # For w1 and w2: (d_model, old_d_ff) -> (d_model, new_d_ff)
        # Interpolate each row
        new_w1 = F.interpolate(
            old_w1.unsqueeze(0).unsqueeze(0),  # (1, 1, d_model, old_d_ff)
            size=(d_model, new_d_ff),
            mode='bilinear',
            align_corners=True,
        ).squeeze(0).squeeze(0)

        new_w2 = F.interpolate(
            old_w2.unsqueeze(0).unsqueeze(0),
            size=(d_model, new_d_ff),
            mode='bilinear',
            align_corners=True,
        ).squeeze(0).squeeze(0)

        # For w3: (old_d_ff, d_model) -> (new_d_ff, d_model)
        new_w3 = F.interpolate(
            old_w3.unsqueeze(0).unsqueeze(0),
            size=(new_d_ff, d_model),
            mode='bilinear',
            align_corners=True,
        ).squeeze(0).squeeze(0)

        # Scale to maintain approximate output magnitude
        scale = math.sqrt(old_d_ff / new_d_ff)
        new_w3 = new_w3 * scale

    else:
        # Shrinking: use importance-based selection
        # Compute importance as L2 norm of each hidden dimension
        importance_w1 = old_w1.norm(dim=0)  # (old_d_ff,)
        importance_w2 = old_w2.norm(dim=0)  # (old_d_ff,)
        importance_w3 = old_w3.norm(dim=1)  # (old_d_ff,)

        # Combined importance
        importance = importance_w1 + importance_w2 + importance_w3

        # Select top new_d_ff most important dimensions
        _, keep_indices = torch.topk(importance, new_d_ff, sorted=False)
        keep_indices = keep_indices.sort().values  # Keep original order

        new_w1 = old_w1[:, keep_indices]
        new_w2 = old_w2[:, keep_indices]
        new_w3 = old_w3[keep_indices, :]

    return new_w1.to(dtype), new_w2.to(dtype), new_w3.to(dtype)


class DynamicExpertTracker:
    """
    Tracks per-expert losses and handles reallocation.
    """

    def __init__(
        self,
        n_experts: int,
        initial_d_ff: int,
        d_model: int,
        config: DynamicExpertConfig,
    ):
        self.n_experts = n_experts
        self.d_model = d_model
        self.config = config

        # Initialize stats for each expert
        self.expert_stats: Dict[int, ExpertStats] = {
            i: ExpertStats(expert_idx=i, d_ff=initial_d_ff)
            for i in range(n_experts)
        }

        self.step_count = 0
        self.reallocation_history: List[Dict] = []

    def update_stats(
        self,
        expert_outputs: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
    ):
        """Update per-expert loss statistics."""
        for expert_idx, output in expert_outputs.items():
            if expert_idx in targets:
                target = targets[expert_idx]
                # Per-expert MSE loss
                loss = F.mse_loss(output, target, reduction='sum').item()
                token_count = output.numel() // self.d_model

                self.expert_stats[expert_idx].total_loss += loss
                self.expert_stats[expert_idx].token_count += token_count

        self.step_count += 1

    def should_reallocate(self) -> bool:
        """Check if it's time to reallocate."""
        return self.step_count > 0 and self.step_count % self.config.reallocate_every_n_steps == 0

    def compute_reallocation(self) -> Dict[int, int]:
        """
        Compute new d_ff for each expert based on loss statistics.

        Returns:
            Dict mapping expert_idx to new d_ff
        """
        # Sort experts by average loss
        sorted_experts = sorted(
            self.expert_stats.values(),
            key=lambda s: s.avg_loss,
            reverse=True,  # Highest loss first
        )

        # Current parameter budget
        current_total_params = sum(
            3 * self.d_model * s.d_ff for s in self.expert_stats.values()
        )

        # Compute new dimensions
        new_d_ff = {s.expert_idx: s.d_ff for s in self.expert_stats.values()}

        # Grow high-loss experts
        for i, stats in enumerate(sorted_experts[:self.config.top_k_grow]):
            new_dim = min(
                int(stats.d_ff * self.config.growth_factor),
                self.config.max_d_ff,
            )
            new_d_ff[stats.expert_idx] = new_dim

        # Shrink low-loss experts
        for i, stats in enumerate(reversed(sorted_experts[-self.config.top_k_shrink:])):
            new_dim = max(
                int(stats.d_ff * self.config.shrink_factor),
                self.config.min_d_ff,
            )
            new_d_ff[stats.expert_idx] = new_dim

        # If maintaining budget, scale to match original total
        if self.config.maintain_param_budget:
            new_total_params = sum(3 * self.d_model * d for d in new_d_ff.values())
            if new_total_params != current_total_params:
                scale = current_total_params / new_total_params
                # Apply scale to dimensions (approximately)
                for expert_idx in new_d_ff:
                    scaled = int(new_d_ff[expert_idx] * math.sqrt(scale))
                    new_d_ff[expert_idx] = max(
                        self.config.min_d_ff,
                        min(scaled, self.config.max_d_ff)
                    )

        return new_d_ff

    def reset_stats(self):
        """Reset statistics after reallocation."""
        for stats in self.expert_stats.values():
            stats.total_loss = 0.0
            stats.token_count = 0

    def get_report(self) -> str:
        """Get a formatted report of expert statistics."""
        lines = ["Expert Statistics:"]
        sorted_stats = sorted(
            self.expert_stats.values(),
            key=lambda s: s.avg_loss,
            reverse=True,
        )

        total_params = sum(3 * self.d_model * s.d_ff for s in self.expert_stats.values())

        for stats in sorted_stats:
            params = 3 * self.d_model * stats.d_ff
            lines.append(
                f"  Expert {stats.expert_idx:2d}: d_ff={stats.d_ff:5d}, "
                f"avg_loss={stats.avg_loss:.6f}, "
                f"params={params:,} ({100*params/total_params:.1f}%)"
            )

        # Summary
        d_ffs = [s.d_ff for s in self.expert_stats.values()]
        lines.append(f"\n  Total params: {total_params:,}")
        lines.append(f"  d_ff range: [{min(d_ffs)}, {max(d_ffs)}]")
        lines.append(f"  d_ff mean: {sum(d_ffs)/len(d_ffs):.0f}")

        return "\n".join(lines)


def reallocate_expert_layer(
    layer: DynamicExpertLayer,
    new_d_ff: Dict[int, int],
    device: str = "cuda",
) -> DynamicExpertLayer:
    """
    Create a new DynamicExpertLayer with reallocated dimensions.

    Interpolates weights to preserve learned functions.
    """
    d_model = layer.d_model
    n_experts = layer.n_experts

    # Create new layer with new dimensions
    new_d_ff_list = [new_d_ff[i] for i in range(n_experts)]
    new_layer = DynamicExpertLayer(
        n_experts=n_experts,
        d_model=d_model,
        d_ff_per_expert=new_d_ff_list,
        activation=layer.activation,
    )

    # Interpolate weights for each expert
    for expert_idx in range(n_experts):
        old_d_ff = layer.d_ff_per_expert[expert_idx]
        target_d_ff = new_d_ff[expert_idx]

        if old_d_ff != target_d_ff:
            new_w1, new_w2, new_w3 = interpolate_expert_weights(
                layer.experts_w1[expert_idx].data,
                layer.experts_w2[expert_idx].data,
                layer.experts_w3[expert_idx].data,
                target_d_ff,
            )
            new_layer.experts_w1[expert_idx].data = new_w1
            new_layer.experts_w2[expert_idx].data = new_w2
            new_layer.experts_w3[expert_idx].data = new_w3
        else:
            # Copy unchanged
            new_layer.experts_w1[expert_idx].data = layer.experts_w1[expert_idx].data.clone()
            new_layer.experts_w2[expert_idx].data = layer.experts_w2[expert_idx].data.clone()
            new_layer.experts_w3[expert_idx].data = layer.experts_w3[expert_idx].data.clone()

    return new_layer.to(device)


def print_active_params_report(
    layers: Dict[int, DynamicExpertLayer],
    top_k: int = 2,
):
    """Print report of total and average active parameters."""
    total_params = 0
    total_active = 0

    print("\nDynamic Expert Parameter Report:")
    print("-" * 60)

    for layer_idx, layer in sorted(layers.items()):
        layer_params = layer.get_total_params()
        layer_active = layer.get_avg_active_params(top_k)
        total_params += layer_params
        total_active += layer_active

        d_ffs = layer.d_ff_per_expert
        print(f"  Layer {layer_idx:2d}: {layer_params:,} total, "
              f"{layer_active:,} active (top-{top_k}), "
              f"d_ff=[{min(d_ffs)}-{max(d_ffs)}]")

    print("-" * 60)
    print(f"  Total:     {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Active:    {total_active:,} ({total_active/1e9:.2f}B) @ top-{top_k}")
    print(f"  Sparsity:  {100*(1 - total_active/total_params):.1f}%")
