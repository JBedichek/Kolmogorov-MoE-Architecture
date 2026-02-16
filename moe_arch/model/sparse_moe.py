"""
Sparse MoE with True Grouped GEMM and Expert Parallelism.

This implementation provides:
1. True sparse computation - only processes assigned tokens (no padding waste)
2. Triton kernels for fused SwiGLU expert computation
3. Custom CUDA kernels for further acceleration (if compiled)
4. Expert parallelism across multiple GPUs using all-to-all communication

Key difference from batched approach:
- Batched: Pads all experts to max_tokens, wastes compute on padding
- Sparse: Uses segment pointers to process only actual tokens per expert

For 64 experts with uneven distribution, sparse can be 2-5x faster.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional
import math

# Try to import custom CUDA kernel
try:
    from ..kernels import grouped_swiglu_forward, is_kernel_available
    _CUDA_KERNEL_AVAILABLE = is_kernel_available()
except ImportError:
    _CUDA_KERNEL_AVAILABLE = False
    grouped_swiglu_forward = None


# =============================================================================
# Triton Grouped GEMM Kernel for True Sparse MoE
# =============================================================================

@triton.jit
def _grouped_gemm_swiglu_fwd(
    # Inputs - all tokens concatenated in expert-sorted order
    X_ptr,              # (total_tokens, d_model) - sorted by expert
    # Weights for ALL experts stacked
    W1_ptr,             # (n_experts, d_model, d_ff)
    W2_ptr,             # (n_experts, d_model, d_ff)
    W3_ptr,             # (n_experts, d_ff, d_model)
    # Output
    Y_ptr,              # (total_tokens, d_model)
    # Segment info
    seg_starts_ptr,     # (n_experts,) - start index for each expert
    seg_ends_ptr,       # (n_experts,) - end index for each expert
    # Dimensions
    total_tokens,
    d_model: tl.constexpr,
    d_ff: tl.constexpr,
    n_experts: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,  # Tokens per block
    BLOCK_K: tl.constexpr,  # Reduction dimension
    BLOCK_N: tl.constexpr,  # Output dimension per block
):
    """
    Grouped GEMM with fused SwiGLU activation.

    Each program processes a tile of tokens for ONE expert.
    Programs are 2D: (expert_id, token_tile)

    Computes: y = (silu(x @ W1) * (x @ W2)) @ W3
    """
    # Get expert and tile indices
    pid_expert = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Get segment bounds for this expert
    seg_start = tl.load(seg_starts_ptr + pid_expert)
    seg_end = tl.load(seg_ends_ptr + pid_expert)
    n_tokens_expert = seg_end - seg_start

    # Early exit if no tokens or tile is out of bounds
    tile_start = pid_m * BLOCK_M
    if tile_start >= n_tokens_expert:
        return

    actual_m = tl.minimum(BLOCK_M, n_tokens_expert - tile_start)

    # Pointers to this expert's weights
    w1_ptr = W1_ptr + pid_expert * d_model * d_ff
    w2_ptr = W2_ptr + pid_expert * d_model * d_ff
    w3_ptr = W3_ptr + pid_expert * d_ff * d_model

    # Token offsets for this tile (in global sorted array)
    offs_m = seg_start + tile_start + tl.arange(0, BLOCK_M)
    mask_m = tl.arange(0, BLOCK_M) < actual_m

    # =========================================================================
    # Stage 1: Compute gate = x @ W1 and value = x @ W2
    # =========================================================================

    # Accumulators for each output column block
    # We process d_ff in chunks of BLOCK_N

    for n_start in range(0, d_ff, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < d_ff

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        value_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Tile over d_model (reduction dimension K)
        for k_start in range(0, d_model, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < d_model

            # Load X tile: (BLOCK_M, BLOCK_K)
            x_ptrs = X_ptr + offs_m[:, None] * d_model + offs_k[None, :]
            x_mask = mask_m[:, None] & mask_k[None, :]
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            # Load W1 tile: (BLOCK_K, BLOCK_N)
            w1_ptrs = w1_ptr + offs_k[:, None] * d_ff + offs_n[None, :]
            w1_mask = mask_k[:, None] & mask_n[None, :]
            w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0).to(tl.float32)

            # Load W2 tile: (BLOCK_K, BLOCK_N)
            w2_ptrs = w2_ptr + offs_k[:, None] * d_ff + offs_n[None, :]
            w2 = tl.load(w2_ptrs, mask=w1_mask, other=0.0).to(tl.float32)

            # Accumulate
            gate_acc += tl.dot(x, w1)
            value_acc += tl.dot(x, w2)

        # Apply SiLU activation to gate: silu(x) = x * sigmoid(x)
        gate = gate_acc * tl.sigmoid(gate_acc)

        # Element-wise multiply: hidden = gate * value
        hidden = gate * value_acc  # (BLOCK_M, BLOCK_N)

        # =====================================================================
        # Stage 2: Compute hidden @ W3 for this n_start block
        # This produces a partial contribution to d_model output dims
        # =====================================================================

        # For each output dimension block
        for d_start in range(0, d_model, BLOCK_K):
            offs_d = d_start + tl.arange(0, BLOCK_K)
            mask_d = offs_d < d_model

            # Load W3 slice: (BLOCK_N, BLOCK_K)
            w3_ptrs = w3_ptr + offs_n[:, None] * d_model + offs_d[None, :]
            w3_mask = mask_n[:, None] & mask_d[None, :]
            w3 = tl.load(w3_ptrs, mask=w3_mask, other=0.0).to(tl.float32)

            # Compute partial output: (BLOCK_M, BLOCK_K)
            out_partial = tl.dot(hidden, w3)

            # Atomic add to output (multiple n_start blocks contribute)
            y_ptrs = Y_ptr + offs_m[:, None] * d_model + offs_d[None, :]
            y_mask = mask_m[:, None] & mask_d[None, :]

            # Use atomic add for accumulation across n blocks
            tl.atomic_add(y_ptrs, out_partial, mask=y_mask)


class SparseGroupedExperts(nn.Module):
    """
    True sparse grouped experts using Triton grouped GEMM.

    Unlike batched MoE which pads to max_tokens, this processes
    only the actual tokens assigned to each expert.

    Memory: O(total_assignments * d_model) not O(n_experts * max_tokens * d_model)
    Compute: O(sum of actual tokens per expert) not O(n_experts * max_tokens)
    """

    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_ff: int,
        activation: str = "swiglu"
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        # Stacked weights for all experts
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for each expert."""
        for w in [self.w1, self.w2, self.w3]:
            fan_in = w.shape[1]
            gain = math.sqrt(1.0 / 3.0)  # For SiLU
            std = gain / math.sqrt(fan_in)
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(
        self,
        x: torch.Tensor,                    # (batch*seq, d_model) or (batch, seq, d_model)
        expert_indices: torch.Tensor,       # (n_tokens, top_k)
        expert_weights: torch.Tensor,       # (n_tokens, top_k)
    ) -> torch.Tensor:
        """
        Forward pass using true sparse grouped GEMM.
        """
        # Handle 3D input
        input_shape = x.shape
        if x.dim() == 3:
            batch, seq, d = x.shape
            x = x.reshape(-1, d)

        n_tokens, d_model = x.shape
        top_k = expert_indices.shape[1]
        device = x.device
        dtype = x.dtype

        # Sort tokens by expert for coalesced memory access
        flat_experts = expert_indices.reshape(-1)
        flat_weights = expert_weights.reshape(-1)
        token_indices = torch.arange(n_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)

        # Stable sort by expert ID
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
        sorted_inputs = x[sorted_token_idx]  # (total_assignments, d_model)
        total_assignments = sorted_inputs.shape[0]

        # Allocate output (zeros for atomic add)
        sorted_outputs = torch.zeros_like(sorted_inputs)

        # Choose block sizes based on dimensions
        BLOCK_M = 32
        BLOCK_K = 64
        BLOCK_N = 64

        # Calculate grid size
        max_tokens_per_expert = expert_counts.max().item()
        n_m_blocks = (max_tokens_per_expert + BLOCK_M - 1) // BLOCK_M

        grid = (self.n_experts, n_m_blocks)

        # Ensure contiguous tensors for Triton
        sorted_inputs = sorted_inputs.contiguous()
        w1 = self.w1.contiguous()
        w2 = self.w2.contiguous()
        w3 = self.w3.contiguous()
        seg_starts = seg_starts.contiguous()
        seg_ends = seg_ends.contiguous()

        # Launch Triton kernel
        _grouped_gemm_swiglu_fwd[grid](
            sorted_inputs, w1, w2, w3, sorted_outputs,
            seg_starts, seg_ends,
            total_assignments,
            d_model, self.d_ff, self.n_experts,
            BLOCK_M, BLOCK_K, BLOCK_N,
        )

        # Apply routing weights
        sorted_outputs = sorted_outputs * sorted_weights.unsqueeze(1)

        # Scatter back to original positions
        output = torch.zeros(n_tokens, d_model, device=device, dtype=dtype)
        output.index_add_(0, sorted_token_idx, sorted_outputs)

        # Reshape if input was 3D
        if len(input_shape) == 3:
            output = output.reshape(input_shape)

        return output


# =============================================================================
# Fallback PyTorch Implementation (for when Triton fails)
# =============================================================================

class SparseGroupedExpertsPyTorch(nn.Module):
    """
    Pure PyTorch sparse grouped experts without padding waste.

    Uses sorted segments and processes each expert's tokens without padding.
    Fallback for when Triton kernel has issues.
    """

    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_ff: int,
        activation: str = "swiglu"
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))

        self._init_weights()

    def _init_weights(self):
        for w in [self.w1, self.w2, self.w3]:
            fan_in = w.shape[1]
            gain = math.sqrt(1.0 / 3.0)
            std = gain / math.sqrt(fan_in)
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Handle 3D input
        input_shape = x.shape
        if x.dim() == 3:
            batch, seq, d = x.shape
            x = x.reshape(-1, d)

        n_tokens, d_model = x.shape
        top_k = expert_indices.shape[1]
        device = x.device
        dtype = x.dtype

        # Sort tokens by expert
        flat_experts = expert_indices.reshape(-1)
        flat_weights = expert_weights.reshape(-1)
        token_indices = torch.arange(n_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)

        sorted_order = torch.argsort(flat_experts, stable=True)
        sorted_token_idx = token_indices[sorted_order]
        sorted_expert_ids = flat_experts[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        # Compute segments
        expert_counts = torch.bincount(flat_experts, minlength=self.n_experts)
        seg_ends = torch.cumsum(expert_counts, dim=0)
        seg_starts = torch.zeros_like(seg_ends)
        seg_starts[1:] = seg_ends[:-1]

        # Gather sorted inputs
        sorted_inputs = x[sorted_token_idx]
        sorted_outputs = torch.zeros_like(sorted_inputs)

        # Try custom CUDA kernel first (fastest)
        if _CUDA_KERNEL_AVAILABLE and grouped_swiglu_forward is not None:
            sorted_outputs = grouped_swiglu_forward(
                sorted_inputs, self.w1, self.w2, self.w3,
                sorted_expert_ids, seg_starts, seg_ends,
                use_cuda_kernel=True
            )
        else:
            # Fallback: PyTorch loop with minimal GPU-CPU syncs
            # Get all segment bounds in ONE sync (instead of 2*n_experts syncs!)
            seg_bounds = torch.stack([seg_starts, seg_ends], dim=1).tolist()

            # Process each expert's segment - TRUE SPARSE (no padding)
            for e, (start, end) in enumerate(seg_bounds):
                if start >= end:
                    continue

                # Process only this expert's tokens (contiguous slice)
                expert_input = sorted_inputs[start:end]  # (n_tokens_e, d_model)

                # SwiGLU: (silu(x @ W1) * (x @ W2)) @ W3
                gate = F.silu(expert_input @ self.w1[e])
                value = expert_input @ self.w2[e]
                hidden = gate * value
                expert_output = hidden @ self.w3[e]

                sorted_outputs[start:end] = expert_output

        # Apply weights and scatter back
        sorted_outputs = sorted_outputs * sorted_weights.unsqueeze(1)

        output = torch.zeros(n_tokens, d_model, device=device, dtype=dtype)
        output.index_add_(0, sorted_token_idx, sorted_outputs)

        if len(input_shape) == 3:
            output = output.reshape(input_shape)

        return output


# =============================================================================
# Expert Parallelism for Multi-GPU Training
# =============================================================================

class ExpertParallelMoE(nn.Module):
    """
    Expert-parallel MoE layer for multi-GPU training.

    Each GPU holds a subset of experts. Tokens are routed across GPUs
    using all-to-all communication.

    Flow:
    1. Each GPU computes routing for all tokens (same on all GPUs)
    2. All-to-all: Send tokens to GPUs holding their assigned experts
    3. Each GPU processes only its local experts
    4. All-to-all: Gather results back to original GPUs

    Memory savings: Each GPU stores only (n_experts / world_size) experts
    Compute scaling: Linear with number of GPUs
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        process_group=None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Get distributed info
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size(process_group)
            self.rank = torch.distributed.get_rank(process_group)
        else:
            self.world_size = 1
            self.rank = 0

        self.process_group = process_group
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k
        self.d_model = config.d_model
        self.d_ff = config.d_ff_expert

        # Divide experts across GPUs
        assert self.n_experts % self.world_size == 0, \
            f"n_experts ({self.n_experts}) must be divisible by world_size ({self.world_size})"

        self.n_local_experts = self.n_experts // self.world_size
        self.expert_start_idx = self.rank * self.n_local_experts
        self.expert_end_idx = self.expert_start_idx + self.n_local_experts

        # Router (same on all GPUs)
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        # Local experts only (saves memory!)
        self.local_experts = SparseGroupedExpertsPyTorch(
            n_experts=self.n_local_experts,
            d_model=config.d_model,
            d_ff=config.d_ff_expert,
        )

        # Aux loss tracking
        self.aux_loss = 0.0

        print(f"[Rank {self.rank}] ExpertParallelMoE: experts {self.expert_start_idx}-{self.expert_end_idx-1} "
              f"(n_local={self.n_local_experts})")

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq, d_model)
    ) -> torch.Tensor:
        """
        Expert-parallel forward pass.
        """
        batch, seq, d = x.shape
        device = x.device
        dtype = x.dtype

        # Flatten for routing
        x_flat = x.reshape(-1, d)  # (batch*seq, d_model)
        n_tokens = x_flat.shape[0]

        # Compute routing logits (same on all GPUs)
        router_logits = self.router(x_flat)  # (n_tokens, n_experts)

        # Top-k selection
        topk_weights, topk_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        topk_weights = F.softmax(topk_weights, dim=-1)  # Normalize weights

        # Compute auxiliary losses
        if self.training:
            self.aux_loss = self._compute_aux_loss(router_logits, topk_indices)

        if self.world_size == 1:
            # Single GPU: just use local experts
            local_indices = topk_indices  # All experts are local
            local_weights = topk_weights
            output = self.local_experts(x_flat, local_indices, local_weights)
            return output.reshape(batch, seq, d)

        # Multi-GPU: expert parallelism with all-to-all
        output = self._expert_parallel_forward(
            x_flat, topk_indices, topk_weights
        )

        return output.reshape(batch, seq, d)

    def _expert_parallel_forward(
        self,
        x: torch.Tensor,          # (n_tokens, d_model)
        expert_indices: torch.Tensor,   # (n_tokens, top_k)
        expert_weights: torch.Tensor,   # (n_tokens, top_k)
    ) -> torch.Tensor:
        """
        All-to-all expert parallel forward.
        """
        n_tokens = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Flatten routing info
        flat_indices = expert_indices.reshape(-1)  # (n_tokens * top_k,)
        flat_weights = expert_weights.reshape(-1)
        token_ids = torch.arange(n_tokens, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        # Determine which GPU each assignment goes to
        target_gpu = flat_indices // self.n_local_experts  # (n_tokens * top_k,)
        local_expert_id = flat_indices % self.n_local_experts

        # Count how many tokens go to each GPU
        send_counts = torch.bincount(target_gpu, minlength=self.world_size)

        # All-to-all to exchange counts
        recv_counts = torch.empty_like(send_counts)
        torch.distributed.all_to_all_single(
            recv_counts, send_counts, group=self.process_group
        )

        # Sort assignments by target GPU for efficient all-to-all
        sorted_order = torch.argsort(target_gpu, stable=True)
        sorted_token_ids = token_ids[sorted_order]
        sorted_expert_ids = local_expert_id[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        sorted_inputs = x[sorted_token_ids]

        # Prepare send/recv buffers
        total_send = sorted_inputs.shape[0]
        total_recv = recv_counts.sum().item()

        send_buffer = sorted_inputs
        recv_buffer = torch.empty(total_recv, self.d_model, device=device, dtype=dtype)

        # All-to-all for inputs
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        torch.distributed.all_to_all_single(
            recv_buffer, send_buffer,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group
        )

        # All-to-all for expert IDs and weights
        send_expert_ids = sorted_expert_ids.to(torch.int32)
        recv_expert_ids = torch.empty(total_recv, dtype=torch.int32, device=device)
        torch.distributed.all_to_all_single(
            recv_expert_ids, send_expert_ids,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group
        )

        send_weights = sorted_weights
        recv_weights = torch.empty(total_recv, dtype=dtype, device=device)
        torch.distributed.all_to_all_single(
            recv_weights, send_weights,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group
        )

        # Process local tokens through local experts
        recv_expert_ids = recv_expert_ids.long()
        local_output = self._process_local_experts(
            recv_buffer, recv_expert_ids, recv_weights
        )

        # All-to-all to send results back
        result_send_buffer = local_output
        result_recv_buffer = torch.empty(total_send, self.d_model, device=device, dtype=dtype)

        torch.distributed.all_to_all_single(
            result_recv_buffer, result_send_buffer,
            output_split_sizes=send_splits,  # Reversed!
            input_split_sizes=recv_splits,
            group=self.process_group
        )

        # Scatter results back to original positions
        output = torch.zeros(n_tokens, self.d_model, device=device, dtype=dtype)
        output.index_add_(0, sorted_token_ids, result_recv_buffer)

        return output

    def _process_local_experts(
        self,
        x: torch.Tensor,           # (n_local_tokens, d_model)
        expert_ids: torch.Tensor,  # (n_local_tokens,) - local expert indices
        weights: torch.Tensor,     # (n_local_tokens,)
    ) -> torch.Tensor:
        """Process tokens through local experts."""

        if x.shape[0] == 0:
            return torch.zeros_like(x)

        # Reshape for local experts (expects top_k dimension)
        expert_ids = expert_ids.unsqueeze(1)  # (n, 1)
        weights = weights.unsqueeze(1)  # (n, 1)

        return self.local_experts(x, expert_ids, weights)

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,  # (n_tokens, n_experts)
        selected_experts: torch.Tensor,  # (n_tokens, top_k)
    ) -> torch.Tensor:
        """Compute load balancing and z-loss."""
        n_tokens = router_logits.shape[0]

        # Load balancing loss
        router_probs = F.softmax(router_logits, dim=-1)
        expert_mask = F.one_hot(selected_experts, self.n_experts).float()
        expert_mask = expert_mask.sum(dim=1)  # (n_tokens, n_experts)

        tokens_per_expert = expert_mask.mean(dim=0)
        router_prob_per_expert = router_probs.mean(dim=0)

        load_balance_loss = self.n_experts * (tokens_per_expert * router_prob_per_expert).sum()

        # Router z-loss (for numerical stability)
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        return (
            self.config.moe_load_balance_loss_weight * load_balance_loss +
            self.config.moe_router_z_loss_weight * z_loss
        )


# =============================================================================
# Factory function to get best available implementation
# =============================================================================

def get_sparse_experts(
    n_experts: int,
    d_model: int,
    d_ff: int,
    use_triton: bool = True,
) -> nn.Module:
    """
    Get the best available sparse experts implementation.

    Tries Triton kernel first, falls back to PyTorch if issues.
    """
    if use_triton:
        try:
            # Test Triton kernel
            test_input = torch.randn(128, d_model, device='cuda', dtype=torch.bfloat16)
            test_indices = torch.randint(0, n_experts, (128, 2), device='cuda')
            test_weights = torch.softmax(torch.randn(128, 2, device='cuda'), dim=-1).bfloat16()

            sparse = SparseGroupedExperts(n_experts, d_model, d_ff).cuda().bfloat16()
            _ = sparse(test_input, test_indices, test_weights)

            print(f"Using Triton SparseGroupedExperts ({n_experts} experts)")
            return SparseGroupedExperts(n_experts, d_model, d_ff)
        except Exception as e:
            print(f"Triton kernel failed: {e}")
            print("Falling back to PyTorch implementation")

    print(f"Using PyTorch SparseGroupedExpertsPyTorch ({n_experts} experts)")
    return SparseGroupedExpertsPyTorch(n_experts, d_model, d_ff)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_sparse_vs_batched():
    """Compare sparse vs batched MoE implementations."""
    import time
    from .triton_moe import BatchedMoEOptimized

    print("=" * 70)
    print("Sparse vs Batched MoE Benchmark")
    print("=" * 70)

    # Production-like config
    n_experts = 64
    d_model = 2048
    d_ff = 1524
    n_tokens = 2048 * 4
    top_k = 2

    device = torch.device('cuda')
    dtype = torch.bfloat16

    print(f"Config: {n_experts} experts, d_model={d_model}, d_ff={d_ff}")
    print(f"Tokens: {n_tokens}, top_k={top_k}")
    print()

    # Inputs
    x = torch.randn(n_tokens, d_model, device=device, dtype=dtype)
    expert_indices = torch.randint(0, n_experts, (n_tokens, top_k), device=device)
    expert_weights = torch.softmax(torch.randn(n_tokens, top_k, device=device), dim=-1).to(dtype)

    # Implementations
    batched = BatchedMoEOptimized(n_experts, d_model, d_ff).to(device).to(dtype)
    sparse_pt = SparseGroupedExpertsPyTorch(n_experts, d_model, d_ff).to(device).to(dtype)

    # Copy weights
    sparse_pt.w1.data.copy_(batched.w1.data)
    sparse_pt.w2.data.copy_(batched.w2.data)
    sparse_pt.w3.data.copy_(batched.w3.data)

    # Warmup
    for _ in range(3):
        _ = batched(x, expert_indices, expert_weights)
        _ = sparse_pt(x, expert_indices, expert_weights)
    torch.cuda.synchronize()

    n_iters = 20

    # Benchmark batched
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        out_batched = batched(x, expert_indices, expert_weights)
    torch.cuda.synchronize()
    time_batched = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark sparse PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        out_sparse = sparse_pt(x, expert_indices, expert_weights)
    torch.cuda.synchronize()
    time_sparse = (time.perf_counter() - start) / n_iters * 1000

    # Try Triton
    try:
        sparse_tri = SparseGroupedExperts(n_experts, d_model, d_ff).to(device).to(dtype)
        sparse_tri.w1.data.copy_(batched.w1.data)
        sparse_tri.w2.data.copy_(batched.w2.data)
        sparse_tri.w3.data.copy_(batched.w3.data)

        # Warmup
        for _ in range(3):
            _ = sparse_tri(x, expert_indices, expert_weights)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            out_triton = sparse_tri(x, expert_indices, expert_weights)
        torch.cuda.synchronize()
        time_triton = (time.perf_counter() - start) / n_iters * 1000

        triton_ok = True
    except Exception as e:
        print(f"Triton failed: {e}")
        time_triton = float('inf')
        triton_ok = False

    print("Results:")
    print(f"  Batched (padded bmm):    {time_batched:.2f} ms")
    print(f"  Sparse (PyTorch):        {time_sparse:.2f} ms ({time_batched/time_sparse:.2f}x vs batched)")
    if triton_ok:
        print(f"  Sparse (Triton):         {time_triton:.2f} ms ({time_batched/time_triton:.2f}x vs batched)")

    # Verify correctness
    print()
    print("Correctness:")
    print(f"  Batched vs Sparse PT: max diff = {(out_batched - out_sparse).abs().max().item():.6f}")


if __name__ == "__main__":
    benchmark_sparse_vs_batched()
