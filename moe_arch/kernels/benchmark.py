"""
Benchmark custom CUDA kernel vs PyTorch implementation.

Usage:
    cd moe_arch/kernels
    python setup.py build_ext --inplace
    python benchmark.py
"""

import torch
import torch.nn.functional as F
import time
import sys

# Add parent to path for imports
sys.path.insert(0, '../..')

from moe_arch.kernels import (
    is_kernel_available,
    grouped_swiglu_forward_pytorch,
    grouped_swiglu_forward,
)


def create_test_data(n_experts: int, d_model: int, d_ff: int, n_tokens: int, top_k: int = 2):
    """Create test data for benchmarking."""
    device = 'cuda'
    dtype = torch.bfloat16

    # Inputs
    x = torch.randn(n_tokens, d_model, device=device, dtype=dtype)

    # Expert assignments (random)
    expert_indices = torch.randint(0, n_experts, (n_tokens, top_k), device=device)

    # Flatten and sort by expert
    flat_experts = expert_indices.reshape(-1)
    token_indices = torch.arange(n_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)

    sorted_order = torch.argsort(flat_experts, stable=True)
    sorted_token_idx = token_indices[sorted_order]
    sorted_expert_ids = flat_experts[sorted_order]

    # Gather sorted inputs
    sorted_inputs = x[sorted_token_idx]

    # Compute segment boundaries
    expert_counts = torch.bincount(flat_experts, minlength=n_experts)
    seg_ends = torch.cumsum(expert_counts, dim=0)
    seg_starts = torch.cat([torch.zeros(1, device=device, dtype=seg_ends.dtype), seg_ends[:-1]])

    # Weights
    W1 = torch.randn(n_experts, d_model, d_ff, device=device, dtype=dtype) * 0.02
    W2 = torch.randn(n_experts, d_model, d_ff, device=device, dtype=dtype) * 0.02
    W3 = torch.randn(n_experts, d_ff, d_model, device=device, dtype=dtype) * 0.02

    return sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends


def verify_correctness(
    sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
):
    """Verify CUDA kernel produces correct results."""
    print("Verifying correctness...")

    # PyTorch reference
    ref_output = grouped_swiglu_forward_pytorch(
        sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
    )

    # CUDA kernel (if available)
    if is_kernel_available():
        cuda_output = grouped_swiglu_forward(
            sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends,
            use_cuda_kernel=True
        )

        # Compare
        max_diff = (ref_output - cuda_output).abs().max().item()
        mean_diff = (ref_output - cuda_output).abs().mean().item()

        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        if max_diff < 0.1:  # BF16 tolerance
            print("  ✓ Correctness verified!")
            return True
        else:
            print("  ✗ Results differ significantly!")
            return False
    else:
        print("  CUDA kernel not available, skipping verification")
        return True


def benchmark(
    sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends,
    n_warmup: int = 5,
    n_iters: int = 20,
):
    """Benchmark PyTorch vs CUDA kernel."""
    print("\nBenchmarking...")

    # Warmup PyTorch
    for _ in range(n_warmup):
        _ = grouped_swiglu_forward_pytorch(
            sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
        )
    torch.cuda.synchronize()

    # Time PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = grouped_swiglu_forward_pytorch(
            sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends
        )
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"  PyTorch: {pytorch_time:.2f} ms")

    # Benchmark CUDA kernel if available
    if is_kernel_available():
        # Warmup CUDA
        for _ in range(n_warmup):
            _ = grouped_swiglu_forward(
                sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends,
                use_cuda_kernel=True
            )
        torch.cuda.synchronize()

        # Time CUDA
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = grouped_swiglu_forward(
                sorted_inputs, W1, W2, W3, sorted_expert_ids, seg_starts, seg_ends,
                use_cuda_kernel=True
            )
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / n_iters * 1000

        print(f"  CUDA kernel: {cuda_time:.2f} ms")
        print(f"  Speedup: {pytorch_time / cuda_time:.2f}x")
    else:
        print("  CUDA kernel not available")


def main():
    print("=" * 60)
    print("MoE Grouped SwiGLU Kernel Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA kernel available: {is_kernel_available()}")

    # Test configurations
    configs = [
        # (n_experts, d_model, d_ff, n_tokens, top_k)
        (8, 512, 1024, 1024, 2),      # Small
        (16, 768, 2048, 2048, 2),     # Medium
        (48, 1024, 3096, 4096, 2),    # Production-like
        (64, 1536, 4096, 8192, 2),    # Large
    ]

    for n_experts, d_model, d_ff, n_tokens, top_k in configs:
        print(f"\n{'='*60}")
        print(f"Config: {n_experts} experts, d_model={d_model}, d_ff={d_ff}")
        print(f"        {n_tokens} tokens, top_k={top_k}")
        print(f"        Total assignments: {n_tokens * top_k}")
        print("=" * 60)

        # Create data
        data = create_test_data(n_experts, d_model, d_ff, n_tokens, top_k)

        # Verify correctness
        verify_correctness(*data)

        # Benchmark
        benchmark(*data)


if __name__ == "__main__":
    main()
