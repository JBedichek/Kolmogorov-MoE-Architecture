/*
 * Grouped GEMM CUDA Kernel for MoE
 *
 * Fused operation: SwiGLU(x @ W1, x @ W2) @ W3
 * Where SwiGLU(gate, value) = silu(gate) * value
 *
 * Optimizations:
 * - Shared memory tiling for efficient memory access
 * - Coalesced global memory reads/writes
 * - Fused activation to avoid extra memory round-trips
 * - BFloat16 compute with FP32 accumulation
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Tile sizes - tuned for modern GPUs
// Keep within 48KB shared memory limit
#define TILE_M 32   // Tokens per tile
#define TILE_N 32   // Output dimension per tile
#define TILE_K 32   // Reduction dimension per tile
#define THREADS_PER_BLOCK 256

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// BF16 to float conversion helpers
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}


/*
 * Tiled GEMM kernel: C = A @ B
 * A: (M, K), B: (K, N), C: (M, N)
 *
 * Each thread block computes a TILE_M x TILE_N tile of C.
 */
template<int TM, int TN, int TK>
__device__ void tiled_gemm(
    const __nv_bfloat16* __restrict__ A,  // (M, K)
    const __nv_bfloat16* __restrict__ B,  // (K, N)
    float* __restrict__ C,                 // (M, N) - accumulator
    int M, int K, int N,
    int row_start, int col_start
) {
    // Shared memory for tiles
    __shared__ float smem_A[TM][TK];
    __shared__ float smem_B[TK][TN];

    // Thread position within block
    int tx = threadIdx.x % TN;
    int ty = threadIdx.x / TN;

    // Accumulators (each thread computes one element)
    float acc = 0.0f;

    // Tile over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TK) {
        // Cooperative loading of A tile
        for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) {
            int local_row = i / TK;
            int local_col = i % TK;
            int global_row = row_start + local_row;
            int global_col = k_tile + local_col;

            if (global_row < M && global_col < K) {
                smem_A[local_row][local_col] = bf16_to_float(A[global_row * K + global_col]);
            } else {
                smem_A[local_row][local_col] = 0.0f;
            }
        }

        // Cooperative loading of B tile
        for (int i = threadIdx.x; i < TK * TN; i += blockDim.x) {
            int local_row = i / TN;
            int local_col = i % TN;
            int global_row = k_tile + local_row;
            int global_col = col_start + local_col;

            if (global_row < K && global_col < N) {
                smem_B[local_row][local_col] = bf16_to_float(B[global_row * N + global_col]);
            } else {
                smem_B[local_row][local_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial dot product
        if (ty < TM && tx < TN) {
            for (int k = 0; k < TK; k++) {
                acc += smem_A[ty][k] * smem_B[k][tx];
            }
        }

        __syncthreads();
    }

    // Store result
    int global_row = row_start + ty;
    int global_col = col_start + tx;
    if (global_row < M && global_col < N && ty < TM && tx < TN) {
        C[global_row * N + global_col] = acc;
    }
}


/*
 * Fused SwiGLU Expert Kernel (currently disabled - using simple kernel instead)
 *
 * The tiled version requires careful shared memory management.
 * For now, the simple kernel is used which is more robust.
 */


/*
 * Simpler version: One thread per output element
 * Less efficient but more correct, good for verification
 */
__global__ void swiglu_expert_simple_kernel(
    const __nv_bfloat16* __restrict__ input,   // (n_tokens, d_model)
    const __nv_bfloat16* __restrict__ W1,      // (d_model, d_ff)
    const __nv_bfloat16* __restrict__ W2,      // (d_model, d_ff)
    const __nv_bfloat16* __restrict__ W3,      // (d_ff, d_model)
    __nv_bfloat16* __restrict__ output,        // (n_tokens, d_model)
    int n_tokens,
    int d_model,
    int d_ff
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / d_model;
    int out_dim = idx % d_model;

    if (token_idx >= n_tokens) return;

    // Pointer to this token's input
    const __nv_bfloat16* token_input = input + token_idx * d_model;

    // Shared memory for this token's input (cooperative load)
    extern __shared__ float shared_input[];

    // Load input to shared memory cooperatively
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        shared_input[i] = bf16_to_float(token_input[i]);
    }
    __syncthreads();

    // Compute output[token_idx, out_dim] = sum_f(silu(sum_d(input[d]*W1[d,f])) * sum_d(input[d]*W2[d,f]) * W3[f,out_dim])
    float result = 0.0f;

    for (int f = 0; f < d_ff; f++) {
        // Compute gate[f] and value[f]
        float gate = 0.0f;
        float value = 0.0f;

        for (int d = 0; d < d_model; d++) {
            float inp = shared_input[d];
            gate += inp * bf16_to_float(W1[d * d_ff + f]);
            value += inp * bf16_to_float(W2[d * d_ff + f]);
        }

        // SiLU and multiply
        float hidden = silu(gate) * value;

        // Accumulate output
        result += hidden * bf16_to_float(W3[f * d_model + out_dim]);
    }

    output[token_idx * d_model + out_dim] = float_to_bf16(result);
}


/*
 * Batched expert kernel - processes multiple experts with CUDA streams
 */
torch::Tensor grouped_swiglu_forward(
    torch::Tensor sorted_inputs,     // (total_tokens, d_model) bf16
    torch::Tensor W1,                // (n_experts, d_model, d_ff) bf16
    torch::Tensor W2,                // (n_experts, d_model, d_ff) bf16
    torch::Tensor W3,                // (n_experts, d_ff, d_model) bf16
    torch::Tensor sorted_expert_ids, // (total_tokens,) int64
    torch::Tensor seg_starts,        // (n_experts,) int64
    torch::Tensor seg_ends           // (n_experts,) int64
) {
    const int total_tokens = sorted_inputs.size(0);
    const int d_model = sorted_inputs.size(1);
    const int n_experts = W1.size(0);
    const int d_ff = W1.size(2);

    // Allocate output (initialized to zero for atomic adds)
    auto sorted_outputs = torch::zeros_like(sorted_inputs);

    // Get segment bounds on CPU for kernel launches
    auto seg_starts_cpu = seg_starts.to(torch::kCPU);
    auto seg_ends_cpu = seg_ends.to(torch::kCPU);
    auto starts = seg_starts_cpu.accessor<int64_t, 1>();
    auto ends = seg_ends_cpu.accessor<int64_t, 1>();

    // Create CUDA streams for concurrent kernel execution
    const int num_streams = min(n_experts, 8);
    cudaStream_t streams[8];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernels for each expert (using multiple streams)
    for (int e = 0; e < n_experts; e++) {
        int64_t start = starts[e];
        int64_t end = ends[e];
        int n_tokens_expert = end - start;

        if (n_tokens_expert <= 0) continue;

        // Pointers for this expert
        const __nv_bfloat16* expert_input =
            reinterpret_cast<const __nv_bfloat16*>(sorted_inputs.data_ptr()) + start * d_model;
        const __nv_bfloat16* expert_W1 =
            reinterpret_cast<const __nv_bfloat16*>(W1.data_ptr()) + e * d_model * d_ff;
        const __nv_bfloat16* expert_W2 =
            reinterpret_cast<const __nv_bfloat16*>(W2.data_ptr()) + e * d_model * d_ff;
        const __nv_bfloat16* expert_W3 =
            reinterpret_cast<const __nv_bfloat16*>(W3.data_ptr()) + e * d_ff * d_model;
        __nv_bfloat16* expert_output =
            reinterpret_cast<__nv_bfloat16*>(sorted_outputs.data_ptr()) + start * d_model;

        cudaStream_t stream = streams[e % num_streams];

        // Use simple kernel for now (more robust)
        int total_elements = n_tokens_expert * d_model;
        int block_size = 256;
        int num_blocks = (total_elements + block_size - 1) / block_size;
        int shared_mem_size = d_model * sizeof(float);

        swiglu_expert_simple_kernel<<<num_blocks, block_size, shared_mem_size, stream>>>(
            expert_input, expert_W1, expert_W2, expert_W3, expert_output,
            n_tokens_expert, d_model, d_ff
        );
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return sorted_outputs;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_swiglu_forward", &grouped_swiglu_forward,
          "Grouped SwiGLU forward pass for MoE experts");
}
