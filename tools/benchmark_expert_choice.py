#!/usr/bin/env python3
"""
Benchmark Expert-Choice Routing Implementations

Compares:
1. Original (for-loop scatter-add)
2. Vectorized PyTorch (scatter_add_)
3. Triton kernel (if available)
4. torch.compile optimization

Usage:
    python tools/benchmark_expert_choice.py [--device cuda] [--warmup 10] [--iters 100]
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available - will skip Triton benchmarks")


def original_scatter_add(output, weighted_outputs, token_indices):
    """Original for-loop implementation."""
    batch_size, seq_len, d_model = output.shape
    _, n_experts, capacity, _ = weighted_outputs.shape

    for expert_idx in range(n_experts):
        idx = token_indices[:, expert_idx, :].unsqueeze(-1).expand(-1, -1, d_model)
        output.scatter_add_(1, idx, weighted_outputs[:, expert_idx, :, :])

    return output


def vectorized_scatter_add(output, weighted_outputs, token_indices):
    """Vectorized PyTorch implementation (no loops)."""
    batch_size, seq_len, d_model = output.shape
    _, n_experts, capacity, _ = weighted_outputs.shape

    # Flatten expert and capacity dimensions
    weighted_flat = weighted_outputs.view(batch_size, n_experts * capacity, d_model)
    indices_flat = token_indices.view(batch_size, n_experts * capacity)
    indices_expanded = indices_flat.unsqueeze(-1).expand(-1, -1, d_model)

    output.scatter_add_(1, indices_expanded, weighted_flat)
    return output


if TRITON_AVAILABLE:
    @triton.jit
    def _scatter_add_kernel(
        output_ptr,
        weighted_out_ptr,
        indices_ptr,
        batch_size: tl.constexpr,
        seq_len: tl.constexpr,
        n_experts: tl.constexpr,
        capacity: tl.constexpr,
        d_model: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cap_idx = pid % capacity
        expert_idx = (pid // capacity) % n_experts
        batch_idx = pid // (capacity * n_experts)

        idx_offset = batch_idx * n_experts * capacity + expert_idx * capacity + cap_idx
        token_idx = tl.load(indices_ptr + idx_offset)

        for d_start in range(0, d_model, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < d_model

            weighted_offset = (
                batch_idx * n_experts * capacity * d_model +
                expert_idx * capacity * d_model +
                cap_idx * d_model +
                d_offsets
            )
            weighted_vals = tl.load(weighted_out_ptr + weighted_offset, mask=d_mask, other=0.0)
            out_offset = batch_idx * seq_len * d_model + token_idx * d_model + d_offsets
            tl.atomic_add(output_ptr + out_offset, weighted_vals, mask=d_mask)

    def triton_scatter_add(output, weighted_outputs, token_indices):
        """Triton kernel implementation."""
        batch_size, seq_len, d_model = output.shape
        _, n_experts, capacity, _ = weighted_outputs.shape

        n_programs = batch_size * n_experts * capacity
        BLOCK_D = min(128, triton.next_power_of_2(d_model))

        _scatter_add_kernel[(n_programs,)](
            output, weighted_outputs.contiguous(), token_indices.contiguous(),
            batch_size, seq_len, n_experts, capacity, d_model,
            BLOCK_D=BLOCK_D,
        )
        return output


def benchmark_fn(fn, output, weighted_outputs, token_indices, warmup=10, iters=100):
    """Benchmark a scatter-add function."""
    # Warmup
    for _ in range(warmup):
        out = torch.zeros_like(output)
        fn(out, weighted_outputs, token_indices)
        torch.cuda.synchronize() if output.is_cuda else None

    # Timed iterations
    torch.cuda.synchronize() if output.is_cuda else None
    start = time.perf_counter()
    for _ in range(iters):
        out = torch.zeros_like(output)
        fn(out, weighted_outputs, token_indices)
        torch.cuda.synchronize() if output.is_cuda else None
    end = time.perf_counter()

    return (end - start) / iters * 1000  # ms per iteration


def main():
    parser = argparse.ArgumentParser(description="Benchmark expert-choice routing")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--d-model', type=int, default=1024)
    parser.add_argument('--n-experts', type=int, default=16)
    parser.add_argument('--capacity-factor', type=float, default=1.25)
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print("=" * 60)
    print("Expert-Choice Routing Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"d_model: {args.d_model}")
    print(f"n_experts: {args.n_experts}")
    print(f"Capacity factor: {args.capacity_factor}")

    # Calculate capacity
    capacity = int(math.ceil(args.seq_len * args.capacity_factor / args.n_experts))
    print(f"Capacity per expert: {capacity}")
    print(f"Total token-expert pairs: {args.n_experts * capacity}")
    print()

    # Create test tensors
    output = torch.zeros(args.batch_size, args.seq_len, args.d_model, device=device, dtype=torch.float32)
    weighted_outputs = torch.randn(args.batch_size, args.n_experts, capacity, args.d_model, device=device, dtype=torch.float32)
    token_indices = torch.randint(0, args.seq_len, (args.batch_size, args.n_experts, capacity), device=device)

    # Benchmark implementations
    print("Benchmarking...")
    print()

    # 1. Original (for-loop)
    t_original = benchmark_fn(original_scatter_add, output, weighted_outputs, token_indices, args.warmup, args.iters)
    print(f"Original (for-loop):     {t_original:.4f} ms")

    # 2. Vectorized PyTorch
    t_vectorized = benchmark_fn(vectorized_scatter_add, output, weighted_outputs, token_indices, args.warmup, args.iters)
    speedup = t_original / t_vectorized
    print(f"Vectorized PyTorch:      {t_vectorized:.4f} ms ({speedup:.2f}x speedup)")

    # 3. Triton (if available)
    if TRITON_AVAILABLE and device == 'cuda':
        t_triton = benchmark_fn(triton_scatter_add, output, weighted_outputs, token_indices, args.warmup, args.iters)
        speedup = t_original / t_triton
        print(f"Triton kernel:           {t_triton:.4f} ms ({speedup:.2f}x speedup)")
    else:
        print("Triton kernel:           [skipped - not available on CPU]")

    # 4. torch.compile (if available)
    if hasattr(torch, 'compile') and device == 'cuda':
        try:
            compiled_fn = torch.compile(vectorized_scatter_add, mode="reduce-overhead")
            # Extra warmup for compilation
            for _ in range(5):
                out = torch.zeros_like(output)
                compiled_fn(out, weighted_outputs, token_indices)
                torch.cuda.synchronize()

            t_compiled = benchmark_fn(compiled_fn, output, weighted_outputs, token_indices, args.warmup, args.iters)
            speedup = t_original / t_compiled
            print(f"torch.compile:           {t_compiled:.4f} ms ({speedup:.2f}x speedup)")
        except Exception as e:
            print(f"torch.compile:           [failed: {e}]")
    else:
        print("torch.compile:           [skipped]")

    # Verify correctness
    print()
    print("Verifying correctness...")

    out_orig = torch.zeros_like(output)
    out_vec = torch.zeros_like(output)
    original_scatter_add(out_orig, weighted_outputs, token_indices)
    vectorized_scatter_add(out_vec, weighted_outputs, token_indices)

    if torch.allclose(out_orig, out_vec, rtol=1e-4, atol=1e-4):
        print("  Vectorized: PASS")
    else:
        diff = (out_orig - out_vec).abs().max().item()
        print(f"  Vectorized: FAIL (max diff: {diff})")

    if TRITON_AVAILABLE and device == 'cuda':
        out_triton = torch.zeros_like(output)
        triton_scatter_add(out_triton, weighted_outputs, token_indices)
        if torch.allclose(out_orig, out_triton, rtol=1e-4, atol=1e-4):
            print("  Triton:     PASS")
        else:
            diff = (out_orig - out_triton).abs().max().item()
            print(f"  Triton:     FAIL (max diff: {diff})")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
