"""
Custom CUDA kernels for MoE acceleration.

Provides:
- grouped_swiglu_forward: Fused grouped GEMM with SwiGLU for all experts

If kernels are not compiled, falls back to PyTorch implementation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# Try to import compiled kernels
_KERNELS_AVAILABLE = False
try:
    import moe_kernels
    _KERNELS_AVAILABLE = True
except ImportError:
    pass


def is_kernel_available() -> bool:
    """Check if custom CUDA kernels are available."""
    return _KERNELS_AVAILABLE


def grouped_swiglu_forward_cuda(
    sorted_inputs: torch.Tensor,     # (total_tokens, d_model) bf16
    W1: torch.Tensor,                # (n_experts, d_model, d_ff) bf16
    W2: torch.Tensor,                # (n_experts, d_model, d_ff) bf16
    W3: torch.Tensor,                # (n_experts, d_ff, d_model) bf16
    sorted_expert_ids: torch.Tensor, # (total_tokens,) int64
    seg_starts: torch.Tensor,        # (n_experts,) int64
    seg_ends: torch.Tensor,          # (n_experts,) int64
) -> torch.Tensor:
    """
    Fused grouped SwiGLU GEMM using custom CUDA kernel.

    For each expert e:
        tokens = sorted_inputs[seg_starts[e]:seg_ends[e]]
        gate = silu(tokens @ W1[e])
        value = tokens @ W2[e]
        hidden = gate * value
        output = hidden @ W3[e]

    Returns:
        sorted_outputs: (total_tokens, d_model) bf16
    """
    if not _KERNELS_AVAILABLE:
        raise RuntimeError("Custom CUDA kernels not compiled. Run: cd moe_arch/kernels && python setup.py install")

    return moe_kernels.grouped_swiglu_forward(
        sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
    )


def grouped_swiglu_forward_pytorch(
    sorted_inputs: torch.Tensor,     # (total_tokens, d_model)
    W1: torch.Tensor,                # (n_experts, d_model, d_ff)
    W2: torch.Tensor,                # (n_experts, d_model, d_ff)
    W3: torch.Tensor,                # (n_experts, d_ff, d_model)
    sorted_expert_ids: torch.Tensor, # (total_tokens,)
    seg_starts: torch.Tensor,        # (n_experts,)
    seg_ends: torch.Tensor,          # (n_experts,)
) -> torch.Tensor:
    """
    PyTorch fallback implementation of grouped SwiGLU.

    Optimized to minimize GPU-CPU syncs (single .tolist() call).
    """
    n_experts = W1.shape[0]
    d_model = W1.shape[1]
    total_tokens = sorted_inputs.shape[0]
    device = sorted_inputs.device
    dtype = sorted_inputs.dtype

    # Pre-allocate output
    sorted_outputs = torch.zeros(total_tokens, d_model, device=device, dtype=dtype)

    # Get all segment bounds in ONE sync
    seg_bounds = torch.stack([seg_starts, seg_ends], dim=1).tolist()

    # Process each expert
    for e, (start, end) in enumerate(seg_bounds):
        start, end = int(start), int(end)
        if end <= start:
            continue

        # Get this expert's tokens
        tokens = sorted_inputs[start:end]  # (n_tokens, d_model)

        # SwiGLU: silu(x @ W1) * (x @ W2) @ W3
        gate = F.silu(tokens @ W1[e])      # (n_tokens, d_ff)
        value = tokens @ W2[e]              # (n_tokens, d_ff)
        hidden = gate * value               # (n_tokens, d_ff)
        output = hidden @ W3[e]             # (n_tokens, d_model)

        sorted_outputs[start:end] = output

    return sorted_outputs


def grouped_swiglu_forward(
    sorted_inputs: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    use_cuda_kernel: bool = True,
) -> torch.Tensor:
    """
    Unified interface for grouped SwiGLU forward pass.

    Automatically uses CUDA kernel if available, falls back to PyTorch.

    Args:
        sorted_inputs: Input tokens sorted by expert (total_tokens, d_model)
        W1, W2, W3: Expert weights
        sorted_expert_ids: Expert ID for each token
        seg_starts, seg_ends: Segment boundaries for each expert
        use_cuda_kernel: Whether to use custom CUDA kernel if available

    Returns:
        sorted_outputs: Output tokens in same order as inputs
    """
    if use_cuda_kernel and _KERNELS_AVAILABLE:
        return grouped_swiglu_forward_cuda(
            sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
        )
    else:
        return grouped_swiglu_forward_pytorch(
            sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
        )


class GroupedSwiGLUFunction(torch.autograd.Function):
    """
    Autograd function for grouped SwiGLU with custom backward.

    Forward: Uses fused kernel
    Backward: Computes gradients for W1, W2, W3, and inputs
    """

    @staticmethod
    def forward(
        ctx,
        sorted_inputs: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        W3: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        seg_starts: torch.Tensor,
        seg_ends: torch.Tensor,
    ) -> torch.Tensor:
        # Save for backward
        ctx.save_for_backward(sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends)

        # Forward pass
        return grouped_swiglu_forward(
            sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends = ctx.saved_tensors

        n_experts = W1.shape[0]
        d_model = W1.shape[1]
        d_ff = W1.shape[2]
        device = sorted_inputs.device
        dtype = sorted_inputs.dtype

        # Initialize gradients
        grad_inputs = torch.zeros_like(sorted_inputs)
        grad_W1 = torch.zeros_like(W1)
        grad_W2 = torch.zeros_like(W2)
        grad_W3 = torch.zeros_like(W3)

        # Get segment bounds
        seg_bounds = torch.stack([seg_starts, seg_ends], dim=1).tolist()

        for e, (start, end) in enumerate(seg_bounds):
            start, end = int(start), int(end)
            if end <= start:
                continue

            # Get tensors for this expert
            tokens = sorted_inputs[start:end]           # (n, d_model)
            grad_out = grad_output[start:end]           # (n, d_model)

            # Recompute forward for backward (could cache but memory trade-off)
            pre_gate = tokens @ W1[e]                   # (n, d_ff)
            gate = F.silu(pre_gate)                     # (n, d_ff)
            value = tokens @ W2[e]                      # (n, d_ff)
            hidden = gate * value                       # (n, d_ff)

            # Backward through W3: output = hidden @ W3
            # grad_hidden = grad_out @ W3.T
            # grad_W3 = hidden.T @ grad_out
            grad_hidden = grad_out @ W3[e].T            # (n, d_ff)
            grad_W3[e] = hidden.T @ grad_out            # (d_ff, d_model)

            # Backward through gate * value
            grad_gate = grad_hidden * value             # (n, d_ff)
            grad_value = grad_hidden * gate             # (n, d_ff)

            # Backward through silu: silu(x) = x * sigmoid(x)
            # d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            #              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            sigmoid_pre_gate = torch.sigmoid(pre_gate)
            grad_pre_gate = grad_gate * sigmoid_pre_gate * (1 + pre_gate * (1 - sigmoid_pre_gate))

            # Backward through W1 and W2
            # pre_gate = tokens @ W1, value = tokens @ W2
            grad_W1[e] = tokens.T @ grad_pre_gate       # (d_model, d_ff)
            grad_W2[e] = tokens.T @ grad_value          # (d_model, d_ff)

            # Backward through input
            grad_inputs[start:end] = grad_pre_gate @ W1[e].T + grad_value @ W2[e].T

        return grad_inputs, grad_W1, grad_W2, grad_W3, None, None, None


def grouped_swiglu_with_grad(
    sorted_inputs: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
) -> torch.Tensor:
    """
    Grouped SwiGLU with proper gradient computation.

    Use this version for training.
    """
    return GroupedSwiGLUFunction.apply(
        sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
    )
