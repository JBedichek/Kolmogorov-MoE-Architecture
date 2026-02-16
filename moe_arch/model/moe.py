"""
Mixture of Experts (MoE) implementation with load balancing.

Features:
- Expert FFN modules
- Softmax router with top-k selection
- Load balancing auxiliary loss (Switch Transformer)
- Router z-loss for stability
- Batched GEMM optimization for efficient expert computation (no Python loops)
- True sparse computation option (no padding waste)
- Expert parallelism across multiple GPUs
- Expert-choice routing with optional Triton kernels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math
import os

from .config import AdvancedMoEConfig
from .ffn import ExpertFFN

# Import sparse implementations (optional - falls back gracefully)
try:
    from .sparse_moe import SparseGroupedExpertsPyTorch, ExpertParallelMoE
    SPARSE_MOE_AVAILABLE = True
except ImportError:
    SPARSE_MOE_AVAILABLE = False

# Import Triton for optimized kernels (optional)
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Control whether to use torch.compile for expert-choice
# Set MOE_DISABLE_COMPILE=1 to disable (useful for debugging)
USE_TORCH_COMPILE = not os.environ.get('MOE_DISABLE_COMPILE', False)


# =============================================================================
# Triton Kernels for Expert-Choice Routing
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _scatter_add_kernel(
        # Pointers to tensors
        output_ptr,          # Output: (batch, seq_len, d_model)
        weighted_out_ptr,    # Input: (batch, n_experts, capacity, d_model)
        indices_ptr,         # Token indices: (batch, n_experts, capacity)
        # Shapes
        batch_size: tl.constexpr,
        seq_len: tl.constexpr,
        n_experts: tl.constexpr,
        capacity: tl.constexpr,
        d_model: tl.constexpr,
        # Block sizes
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused scatter-add kernel for expert-choice routing.

        Each program handles one (batch, expert, capacity_slot) combination
        and scatters its d_model elements to the correct token position.
        """
        # Program ID maps to (batch_idx * n_experts * capacity + expert_idx * capacity + cap_idx)
        pid = tl.program_id(0)

        cap_idx = pid % capacity
        expert_idx = (pid // capacity) % n_experts
        batch_idx = pid // (capacity * n_experts)

        # Load the token index for this (batch, expert, cap) slot
        idx_offset = batch_idx * n_experts * capacity + expert_idx * capacity + cap_idx
        token_idx = tl.load(indices_ptr + idx_offset)

        # Process d_model elements in blocks
        for d_start in range(0, d_model, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < d_model

            # Load weighted output for this slot
            weighted_offset = (
                batch_idx * n_experts * capacity * d_model +
                expert_idx * capacity * d_model +
                cap_idx * d_model +
                d_offsets
            )
            weighted_vals = tl.load(weighted_out_ptr + weighted_offset, mask=d_mask, other=0.0)

            # Compute output offset for this token
            out_offset = batch_idx * seq_len * d_model + token_idx * d_model + d_offsets

            # Atomic add to handle multiple experts writing to same token
            tl.atomic_add(output_ptr + out_offset, weighted_vals, mask=d_mask)

    def triton_scatter_add_experts(
        output: torch.Tensor,           # (batch, seq_len, d_model) - zeroed
        weighted_outputs: torch.Tensor, # (batch, n_experts, capacity, d_model)
        token_indices: torch.Tensor,    # (batch, n_experts, capacity)
    ):
        """
        Fused scatter-add using Triton kernel.

        Scatters expert outputs to token positions with proper accumulation
        when multiple experts select the same token.
        """
        batch_size, seq_len, d_model = output.shape
        _, n_experts, capacity, _ = weighted_outputs.shape

        # Total number of scatter operations
        n_programs = batch_size * n_experts * capacity

        # Choose block size for d_model dimension
        BLOCK_D = min(128, triton.next_power_of_2(d_model))

        # Launch kernel
        _scatter_add_kernel[(n_programs,)](
            output, weighted_outputs.contiguous(), token_indices.contiguous(),
            batch_size, seq_len, n_experts, capacity, d_model,
            BLOCK_D=BLOCK_D,
        )

        return output


def _pytorch_scatter_add_experts(
    output: torch.Tensor,           # (batch, seq_len, d_model) - zeroed
    weighted_outputs: torch.Tensor, # (batch, n_experts, capacity, d_model)
    token_indices: torch.Tensor,    # (batch, n_experts, capacity)
) -> torch.Tensor:
    """
    PyTorch fallback for scatter-add (vectorized, no Python loops).

    Uses advanced indexing with scatter_add_ for efficiency.
    """
    batch_size, seq_len, d_model = output.shape
    _, n_experts, capacity, _ = weighted_outputs.shape

    # Flatten expert and capacity dimensions for batched scatter
    # weighted_outputs: (batch, n_experts * capacity, d_model)
    weighted_flat = weighted_outputs.view(batch_size, n_experts * capacity, d_model)

    # token_indices: (batch, n_experts * capacity)
    indices_flat = token_indices.view(batch_size, n_experts * capacity)

    # Expand indices for scatter_add: (batch, n_experts * capacity, d_model)
    indices_expanded = indices_flat.unsqueeze(-1).expand(-1, -1, d_model)

    # Scatter add - fully vectorized, no loops!
    output.scatter_add_(1, indices_expanded, weighted_flat)

    return output


class GroupedExperts(nn.Module):
    """
    Grouped expert computation with stacked weight matrices.

    Uses stacked weights but processes tokens grouped by expert to avoid
    memory-intensive weight gathering. This is a balance between:
    - Fully batched (too much memory)
    - Per-token loops (too slow)

    For SwiGLU: output = (silu(x @ W1) * (x @ W2)) @ W3
    """

    def __init__(self, n_experts: int, d_model: int, d_ff: int, activation: str = "swiglu"):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        if activation == "swiglu":
            # SwiGLU has 3 weight matrices per expert
            # Stack them: (n_experts, d_model, d_ff) etc.
            self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
            self.w2 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
            self.w3 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        else:
            # Standard FFN has 2 weight matrices
            self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
            self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
            self.w3 = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling.

        IMPORTANT: For 3D tensors (n_experts, in_dim, out_dim), PyTorch's
        kaiming_uniform computes fan incorrectly (includes all dims).
        We need to init each expert slice separately with correct fan.
        """
        for w in [self.w1, self.w2, self.w3]:
            if w is not None:
                # Init each expert's weights separately with correct fan
                # w shape: (n_experts, in_features, out_features)
                fan_in = w.shape[1]  # in_features
                # Kaiming uniform bounds: [-bound, bound] where bound = sqrt(3/fan_in) * gain
                # For LeakyReLU with a=sqrt(5), gain = sqrt(2 / (1 + a^2)) = sqrt(1/3)
                gain = math.sqrt(1.0 / 3.0)  # For a=sqrt(5) in kaiming
                std = gain / math.sqrt(fan_in)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(w, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched expert forward pass using torch.bmm.

        Uses sorted indices + padded batches for a single bmm call across all experts.
        This is ~2x faster than Python loops for many experts (tested: 110 experts).

        Args:
            x: (n_tokens, d_model) - flattened input tokens
            expert_indices: (n_tokens, top_k) - which experts each token uses
            expert_weights: (n_tokens, top_k) - weight for each expert

        Returns:
            output: (n_tokens, d_model)
        """
        n_tokens = x.shape[0]
        _, top_k = expert_indices.shape
        device = x.device
        dtype = x.dtype

        # Sort tokens by expert for efficient batching
        flat_experts = expert_indices.reshape(-1)
        flat_weights = expert_weights.reshape(-1)
        token_indices = torch.arange(n_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)

        # Sort by expert ID
        sorted_order = torch.argsort(flat_experts, stable=True)
        sorted_token_idx = token_indices[sorted_order]
        sorted_expert_ids = flat_experts[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        # Compute segment boundaries
        expert_counts = torch.bincount(flat_experts, minlength=self.n_experts)
        seg_ends = torch.cumsum(expert_counts, dim=0)
        seg_starts = torch.zeros_like(seg_ends)
        seg_starts[1:] = seg_ends[:-1]

        # Gather inputs in sorted order
        all_inputs = x[sorted_token_idx]
        total_assignments = len(sorted_token_idx)

        # Compute max batch size for padding
        max_tokens = expert_counts.max().item()

        if max_tokens == 0:
            return torch.zeros(n_tokens, self.d_model, device=device, dtype=dtype)

        # Create position indices within each expert's batch - fully vectorized
        # For each assignment, compute its position within its expert's batch
        # positions[i] = i - seg_starts[sorted_expert_ids[i]]
        positions = torch.arange(total_assignments, device=device) - seg_starts[sorted_expert_ids]

        # Scatter into padded tensor: (n_experts, max_tokens, d_model)
        padded = torch.zeros(self.n_experts, max_tokens, self.d_model, device=device, dtype=dtype)
        padded[sorted_expert_ids, positions] = all_inputs

        # Single batched matmul for ALL experts - no Python loop!
        if self.activation == "swiglu":
            gate = F.silu(torch.bmm(padded, self.w1))
            value = torch.bmm(padded, self.w2)
            hidden = gate * value
            expert_out = torch.bmm(hidden, self.w3)
        else:
            hidden = F.silu(torch.bmm(padded, self.w1))
            expert_out = torch.bmm(hidden, self.w2)

        # Gather back from padded tensor
        all_outputs = expert_out[sorted_expert_ids, positions]

        # Apply routing weights
        all_outputs = all_outputs * sorted_weights.unsqueeze(1)

        # Scatter back to original positions
        output = torch.zeros(n_tokens, self.d_model, device=device, dtype=dtype)
        output.index_add_(0, sorted_token_idx, all_outputs)

        return output


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
    Token-choice router: each token selects its top-k experts.

    Uses a simple MLP to produce routing scores, then applies top-k selection.
    This is the standard MoE routing approach (Switch Transformer, etc.)

    When balanced_routing is enabled, enforces capacity constraints per expert
    to prevent collapse (each expert gets roughly equal tokens per batch).
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k
        self.balanced_routing = getattr(config, 'moe_balanced_routing', False)

        # Router network: d_model -> hidden -> n_experts
        hidden_dim = 128
        self.router = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.n_experts, bias=False),
        )

    def _balanced_topk_assignment(
        self,
        router_probs: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign top-k experts per token with global capacity constraints.

        Unlike pure top-k (which allows one expert to dominate), this ensures
        each expert gets roughly equal number of tokens across the batch.

        Uses vectorized greedy assignment with capacity constraints.

        Args:
            router_probs: (n_tokens, n_experts) routing probabilities
            top_k: number of experts per token

        Returns:
            routing_weights: (n_tokens, top_k) - weights for selected experts
            selected_experts: (n_tokens, top_k) - indices of selected experts
        """
        n_tokens, n_experts = router_probs.shape
        device = router_probs.device
        dtype = router_probs.dtype

        # Capacity per expert: total assignments / n_experts
        capacity_per_expert = max((n_tokens * top_k) // n_experts, 1)

        # Output tensors
        selected_experts = torch.zeros(n_tokens, top_k, dtype=torch.long, device=device)
        routing_weights = torch.zeros(n_tokens, top_k, device=device, dtype=dtype)

        # Work with a copy we can modify
        probs = router_probs.clone()

        # Track capacity per expert
        expert_counts = torch.zeros(n_experts, dtype=torch.long, device=device)

        for k in range(top_k):
            # Create capacity mask: experts at capacity get -inf
            at_capacity = expert_counts >= capacity_per_expert
            masked_probs = probs.clone()
            masked_probs[:, at_capacity] = float('-inf')

            # Each token picks its best available expert
            best_probs, best_experts = masked_probs.max(dim=-1)

            # Handle edge case: all experts at capacity for some tokens
            # (shouldn't happen with proper capacity, but be safe)
            needs_reset = best_probs == float('-inf')
            if needs_reset.any():
                # For these tokens, just pick from original probs
                best_experts[needs_reset] = probs[needs_reset].argmax(dim=-1)
                best_probs[needs_reset] = probs[needs_reset].max(dim=-1).values

            # Store selections
            selected_experts[:, k] = best_experts
            routing_weights[:, k] = torch.gather(router_probs, 1, best_experts.unsqueeze(-1)).squeeze(-1)

            # Update expert counts (vectorized)
            expert_counts.scatter_add_(0, best_experts, torch.ones_like(best_experts, dtype=torch.long))

            # Mask out selected experts for next k (so tokens don't pick same expert twice)
            probs.scatter_(1, best_experts.unsqueeze(-1), float('-inf'))

        # Normalize routing weights to sum to 1 per token
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        return routing_weights, selected_experts

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

        if self.balanced_routing:
            # Balanced assignment with capacity constraints
            # Flatten batch and seq for balanced assignment
            router_probs_flat = router_probs.view(-1, self.n_experts)
            routing_weights_flat, selected_experts_flat = self._balanced_topk_assignment(
                router_probs_flat, self.top_k
            )
            # Reshape back
            routing_weights = routing_weights_flat.view(batch_size, seq_len, self.top_k)
            selected_experts = selected_experts_flat.view(batch_size, seq_len, self.top_k)
        else:
            # Standard top-k selection
            routing_weights, selected_experts = torch.topk(
                router_probs, self.top_k, dim=-1
            )  # Each: (batch, seq, top_k)

            # Normalize routing weights to sum to 1
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts, router_logits


class ExpertChoiceRouter(nn.Module):
    """
    Expert-choice router: each expert selects its top-k tokens.

    From "Mixture-of-Experts with Expert Choice Routing" (Zhou et al., 2022)

    Key advantages over token-choice:
    - Perfect load balancing by construction (each expert processes same # tokens)
    - No auxiliary load balancing loss needed
    - Resistant to router collapse
    - Tokens can be processed by 0, 1, or multiple experts

    The capacity per expert is: capacity = ceil(seq_len * capacity_factor / n_experts)
    """

    def __init__(self, config: AdvancedMoEConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.capacity_factor = config.moe_capacity_factor

        # Router network: d_model -> hidden -> n_experts
        hidden_dim = 128
        self.router = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.n_experts, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing: each expert selects its top tokens.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            expert_weights: (batch, n_experts, capacity) - weights for selected tokens
            token_indices: (batch, n_experts, capacity) - which tokens each expert selected
            router_logits: (batch, seq_len, n_experts) - raw router scores (for monitoring)
            tokens_per_expert: int - capacity per expert
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Compute capacity per expert
        # Each expert selects capacity tokens, total = n_experts * capacity ≈ seq_len * capacity_factor
        capacity = int(math.ceil(seq_len * self.capacity_factor / self.n_experts))

        # Compute router logits: (batch, seq, n_experts)
        router_logits = self.router(hidden_states)

        # Transpose to get expert view: (batch, n_experts, seq)
        expert_logits = router_logits.transpose(1, 2)

        # Apply softmax over tokens (each expert distributes attention over tokens)
        expert_probs = F.softmax(expert_logits, dim=-1)  # (batch, n_experts, seq)

        # Each expert selects top-capacity tokens
        expert_weights, token_indices = torch.topk(
            expert_probs, capacity, dim=-1
        )  # Each: (batch, n_experts, capacity)

        # Normalize weights to sum to 1 per expert
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-9)

        return expert_weights, token_indices, router_logits, capacity


class GroupedExpertsExpertChoice(nn.Module):
    """
    Expert computation for expert-choice routing.

    Similar to GroupedExperts but handles the different routing format:
    - Input: token indices selected by each expert
    - Each token may be selected by 0, 1, or multiple experts
    - Outputs are combined by summing weighted contributions

    Optimizations:
    - Uses torch.compile for the expert computation path (if available)
    - Uses Triton kernel for fused scatter-add (if available)
    - Falls back to vectorized PyTorch scatter_add_ (no Python loops)
    """

    def __init__(self, n_experts: int, d_model: int, d_ff: int, activation: str = "swiglu"):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        if activation == "swiglu":
            self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
            self.w2 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
            self.w3 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        else:
            self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
            self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
            self.w3 = None

        self._init_weights()

        # Compile the expert computation if enabled
        self._expert_compute = self._expert_compute_impl
        if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
            try:
                self._expert_compute = torch.compile(
                    self._expert_compute_impl,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception:
                pass  # Fall back to non-compiled version

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for w in [self.w1, self.w2, self.w3]:
            if w is not None:
                fan_in = w.shape[1]
                gain = math.sqrt(1.0 / 3.0)
                std = gain / math.sqrt(fan_in)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(w, -bound, bound)

    def _expert_compute_impl(
        self,
        expert_inputs: torch.Tensor,  # (batch, n_experts, capacity, d_model)
        batch_size: int,
        n_experts: int,
        capacity: int,
        d_model: int,
    ) -> torch.Tensor:
        """
        Core expert computation - separated for torch.compile optimization.

        Returns: (batch, n_experts, capacity, d_model)
        """
        # Reshape for bmm: (batch * n_experts, capacity, d_model)
        expert_inputs_flat = expert_inputs.reshape(batch_size * n_experts, capacity, d_model)

        # Use einsum for cleaner batched computation (often faster than expand+reshape+bmm)
        # expert_inputs_flat: (B*E, C, D) @ w1: (E, D, F) -> need to handle batching

        # Expand weights efficiently using repeat instead of expand+reshape
        # This creates views when possible
        w1 = self.w1.repeat(batch_size, 1, 1)  # (batch * n_experts, d_model, d_ff)

        if self.activation == "swiglu":
            w2 = self.w2.repeat(batch_size, 1, 1)
            w3 = self.w3.repeat(batch_size, 1, 1)

            gate = F.silu(torch.bmm(expert_inputs_flat, w1))
            value = torch.bmm(expert_inputs_flat, w2)
            hidden = gate * value
            expert_outputs_flat = torch.bmm(hidden, w3)
        else:
            w2 = self.w2.repeat(batch_size, 1, 1)
            hidden = F.silu(torch.bmm(expert_inputs_flat, w1))
            expert_outputs_flat = torch.bmm(hidden, w2)

        return expert_outputs_flat.view(batch_size, n_experts, capacity, d_model)

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expert-choice forward pass.

        Args:
            x: (batch, seq_len, d_model) - input tokens
            expert_weights: (batch, n_experts, capacity) - weights for selected tokens
            token_indices: (batch, n_experts, capacity) - which tokens each expert selected

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        _, n_experts, capacity = token_indices.shape
        device = x.device
        dtype = x.dtype

        # Gather tokens for each expert: (batch, n_experts, capacity, d_model)
        token_idx_expanded = token_indices.unsqueeze(-1).expand(-1, -1, -1, d_model)
        x_expanded = x.unsqueeze(1).expand(-1, n_experts, -1, -1)
        expert_inputs = torch.gather(x_expanded, 2, token_idx_expanded)

        # Process through experts (compiled if available)
        expert_outputs = self._expert_compute(
            expert_inputs, batch_size, n_experts, capacity, d_model
        )

        # Apply routing weights: (batch, n_experts, capacity, 1)
        expert_weights_expanded = expert_weights.unsqueeze(-1)
        weighted_outputs = expert_outputs * expert_weights_expanded

        # Scatter-add outputs back to original token positions
        output = torch.zeros(batch_size, seq_len, d_model, device=device, dtype=dtype)

        # Use vectorized PyTorch scatter_add_ (benchmarked faster than Triton due to atomic overhead)
        # Set MOE_USE_TRITON_SCATTER=1 to force Triton kernel usage
        use_triton = TRITON_AVAILABLE and x.is_cuda and os.environ.get('MOE_USE_TRITON_SCATTER', False)
        if use_triton:
            triton_scatter_add_experts(output, weighted_outputs, token_indices)
        else:
            _pytorch_scatter_add_experts(output, weighted_outputs, token_indices)

        return output


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with configurable routing strategy.

    Routing options (via config.moe_routing):
    - "token_choice": Tokens select top-k experts (standard, requires load balancing loss)
    - "expert_choice": Experts select top-k tokens (perfect balance, collapse-resistant)

    Implementation options (via config.moe_implementation):
    - "batched": Padded batched bmm (default, fastest for small expert count)
    - "sparse": True sparse computation (no padding waste, better for many experts)
    - "expert_parallel": Distributed experts across GPUs (for multi-GPU training)
    """

    def __init__(self, config: AdvancedMoEConfig, layer_idx: int, process_group=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k
        self.capacity_factor = config.moe_capacity_factor

        # Get routing type from config (default to token_choice for backward compatibility)
        self.routing_type = getattr(config, 'moe_routing', 'token_choice')

        # Get implementation type from config (default to batched)
        self.implementation = getattr(config, 'moe_implementation', 'batched')

        # Select expert implementation based on config
        if self.implementation == 'expert_parallel' and SPARSE_MOE_AVAILABLE:
            # Expert parallelism: each GPU holds subset of experts
            # This wraps its own router and experts
            self._use_expert_parallel = True
            self.expert_parallel_layer = ExpertParallelMoE(config, layer_idx, process_group)
            self.grouped_experts = None
            self.grouped_experts_ec = None
            self.router = None
        else:
            self._use_expert_parallel = False
            self.expert_parallel_layer = None

            # Create appropriate router and experts based on routing type
            if self.routing_type == 'expert_choice':
                # Expert-choice routing
                self.router = ExpertChoiceRouter(config)
                self.grouped_experts = None
                self.grouped_experts_ec = GroupedExpertsExpertChoice(
                    n_experts=config.n_experts,
                    d_model=config.d_model,
                    d_ff=config.d_ff_expert,
                    activation=config.ffn_activation,
                )
            else:
                # Token-choice routing (default)
                self.router = Router(config)
                self.grouped_experts_ec = None

                # Select between batched and sparse experts
                if self.implementation == 'sparse' and SPARSE_MOE_AVAILABLE:
                    self.grouped_experts = SparseGroupedExpertsPyTorch(
                        n_experts=config.n_experts,
                        d_model=config.d_model,
                        d_ff=config.d_ff_expert,
                    )
                else:
                    self.grouped_experts = GroupedExperts(
                        n_experts=config.n_experts,
                        d_model=config.d_model,
                        d_ff=config.d_ff_expert,
                        activation=config.ffn_activation,
                    )

        # Keep ModuleList for compatibility but don't use it
        # (this allows loading old checkpoints)
        self.experts = None

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
        # Expert parallel path (has its own router)
        if self._use_expert_parallel:
            output = self.expert_parallel_layer(hidden_states)
            self.aux_loss = self.expert_parallel_layer.aux_loss
            return output

        # Expert-choice routing path
        if self.routing_type == 'expert_choice':
            return self._forward_expert_choice(hidden_states)

        # Token-choice routing path (default)
        return self._forward_token_choice(hidden_states)

    def _forward_token_choice(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Token-choice routing: tokens select experts."""
        batch_size, seq_len, d_model = hidden_states.shape

        # Route tokens to experts
        routing_weights, selected_experts, router_logits = self.router(hidden_states)

        # Reshape for expert processing
        hidden_states_flat = hidden_states.view(-1, d_model)
        routing_weights_flat = routing_weights.view(-1, self.top_k)
        selected_experts_flat = selected_experts.view(-1, self.top_k)

        # Process through grouped experts
        output_flat = self.grouped_experts(
            hidden_states_flat,
            selected_experts_flat,
            routing_weights_flat,
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

    def _forward_expert_choice(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expert-choice routing: experts select tokens."""
        batch_size, seq_len, d_model = hidden_states.shape

        # Route: each expert selects its tokens
        expert_weights, token_indices, router_logits, capacity = self.router(hidden_states)
        # expert_weights: (batch, n_experts, capacity)
        # token_indices: (batch, n_experts, capacity)

        # Process through expert-choice grouped experts
        output = self.grouped_experts_ec(
            hidden_states,
            expert_weights,
            token_indices,
        )

        # Expert-choice has perfect load balance by construction, so no aux loss needed
        # But we can still add z-loss for stability if desired
        if self.training and self.router_z_loss_weight > 0:
            self.aux_loss = self.router_z_loss_weight * self._compute_router_z_loss(router_logits)
        else:
            self.aux_loss = 0.0

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
