#include "householder.cuh"
#include <cstdio>
#include <cmath>
#include <algorithm>

// CONSTANTS
#define EPSILON 1e-10f

// GEMM TUNING
// 64x64 Output Tile
// 16 Accumulation Depth (BK)
// Vectorized loads (float4)
#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_BK 16 
#define GEMM_TM 8
#define GEMM_TN 4

namespace Householder {

    static cudaStream_t g_stream_panel = nullptr;
    static cudaStream_t g_stream_update = nullptr;

    void init_resources() {
        if (!g_stream_panel) {
            cudaStreamCreate(&g_stream_panel);
            cudaStreamCreate(&g_stream_update);
        }
    }

    void cleanup_resources() {
        if (g_stream_panel) {
            cudaStreamDestroy(g_stream_panel);
            cudaStreamDestroy(g_stream_update);
            g_stream_panel = nullptr;
            g_stream_update = nullptr;
        }
    }

    // ============================================================================
    // DEVICE UTILITIES
    // ============================================================================

    __inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }

    __inline__ __device__ float blockReduceSum(float val) {
        __shared__ float shared[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        val = warpReduceSum(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        if (wid == 0) val = warpReduceSum(val);
        return val;
    }

    // ============================================================================
    // VECTORIZED GEMM KERNELS
    // ============================================================================

    // VECTORIZED NN GEMM
    __global__
        void kernel_gemm_nn_vec(int M, int N, int K, float alpha,
            const float* __restrict__ A, int lda,
            const float* __restrict__ B, int ldb,
            float beta,
            float* __restrict__ C, int ldc)
    {
        // Block Calculation Indices
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        const int tid = threadIdx.x; // 0..255

        // Shared Memory
        __shared__ float As[GEMM_BK][GEMM_BM]; // Transposed storage [16][64]
        __shared__ float Bs[GEMM_BK][GEMM_BN]; // [16][64]

        // Registers
        float thread_results[GEMM_TM][GEMM_TN] = { 0.0f }; // 8x4 = 32 regs
        float reg_a[GEMM_TM]; // 8 regs
        float reg_b[GEMM_TN]; // 4 regs

        const int ty_c = tid / 16; // 0..15
        const int tx_c = tid % 16; // 0..15

        int load_a_row = (tid % 16) * 4;
        int load_a_col = tid / 16;

        int load_b_row = (tid % 4) * 4;
        int load_b_col = tid / 4;

        // Pointers to the current block in Global Memory
        const float* A_ptr = A + by * GEMM_BM; // Move to block row
        const float* B_ptr = B + bx * GEMM_BN * ldb; // Move to block col

        float4 ldg_a, ldg_b;

        for (int k = 0; k < K; k += GEMM_BK) {

            // 1. Vector Load A (A is M x K)
            if (by * GEMM_BM + load_a_row < M && k + load_a_col < K) {
                ldg_a = *reinterpret_cast<const float4*>(&A_ptr[(k + load_a_col) * lda + load_a_row]);
                As[load_a_col][load_a_row + 0] = ldg_a.x;
                As[load_a_col][load_a_row + 1] = ldg_a.y;
                As[load_a_col][load_a_row + 2] = ldg_a.z;
                As[load_a_col][load_a_row + 3] = ldg_a.w;
            }
            else {
                As[load_a_col][load_a_row + 0] = 0.0f;
                As[load_a_col][load_a_row + 1] = 0.0f;
                As[load_a_col][load_a_row + 2] = 0.0f;
                As[load_a_col][load_a_row + 3] = 0.0f;
            }

            // 2. Vector Load B (B is K x N)
            if (k + load_b_row < K && bx * GEMM_BN + load_b_col < N) {
                ldg_b = *reinterpret_cast<const float4*>(&B_ptr[load_b_col * ldb + k + load_b_row]);
                Bs[load_b_row + 0][load_b_col] = ldg_b.x;
                Bs[load_b_row + 1][load_b_col] = ldg_b.y;
                Bs[load_b_row + 2][load_b_col] = ldg_b.z;
                Bs[load_b_row + 3][load_b_col] = ldg_b.w;
            }
            else {
                Bs[load_b_row + 0][load_b_col] = 0.0f;
                Bs[load_b_row + 1][load_b_col] = 0.0f;
                Bs[load_b_row + 2][load_b_col] = 0.0f;
                Bs[load_b_row + 3][load_b_col] = 0.0f;
            }

            __syncthreads();

            // 3. Compute (4x4 tiles per thread)
#pragma unroll
            for (int bk = 0; bk < GEMM_BK; ++bk) {
#pragma unroll
                for (int i = 0; i < 4; ++i) reg_a[i] = As[bk][ty_c * 4 + i];

#pragma unroll
                for (int i = 0; i < 4; ++i) reg_b[i] = Bs[bk][tx_c * 4 + i];

#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        thread_results[i][j] += reg_a[i] * reg_b[j];
                    }
                }
            }
            __syncthreads();
        }

        // 4. Store
        int global_c_row = by * GEMM_BM + ty_c * 4;
        int global_c_col = bx * GEMM_BN + tx_c * 4;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = global_c_col + j;
            if (col < N) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int row = global_c_row + i;
                    if (row < M) {
                        float val = alpha * thread_results[i][j];
                        if (beta != 0.0f) val += beta * C[col * ldc + row];
                        C[col * ldc + row] = val;
                    }
                }
            }
        }
    }

    // VECTORIZED TN GEMM
    __global__
        void kernel_gemm_tn_vec(int M, int N, int K, float alpha,
            const float* __restrict__ A, int lda,
            const float* __restrict__ B, int ldb,
            float beta,
            float* __restrict__ C, int ldc)
    {
        // Dimensions: M=64 (fixed block), N=Variable, K=Large
        // Block: 64x64 output tile.

        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tid = threadIdx.x;

        __shared__ float As[GEMM_BK][GEMM_BM]; // [16][64]
        __shared__ float Bs[GEMM_BK][GEMM_BN]; // [16][64]

        float thread_results[4][4] = { 0.0f };
        float reg_a[4];
        float reg_b[4];

        const int ty_c = tid / 16;
        const int tx_c = tid % 16;

        int load_a_row_A = (tid % 4) * 4;
        int load_a_col_A = tid / 4;

        int load_b_row = (tid % 4) * 4;
        int load_b_col = tid / 4;

        const float* A_ptr = A;
        const float* B_ptr = B + bx * GEMM_BN * ldb;

        float4 ldg_a, ldg_b;

        for (int k = 0; k < K; k += GEMM_BK) {
            int global_A_col = by * GEMM_BM + load_a_col_A;
            int global_A_row = k + load_a_row_A;

            if (global_A_col < M && global_A_row < K) {
                ldg_a = *reinterpret_cast<const float4*>(&A_ptr[global_A_col * lda + global_A_row]);

                As[load_a_row_A + 0][load_a_col_A] = ldg_a.x;
                As[load_a_row_A + 1][load_a_col_A] = ldg_a.y;
                As[load_a_row_A + 2][load_a_col_A] = ldg_a.z;
                As[load_a_row_A + 3][load_a_col_A] = ldg_a.w;
            }
            else {
                As[load_a_row_A + 0][load_a_col_A] = 0.0f;
                As[load_a_row_A + 1][load_a_col_A] = 0.0f;
                As[load_a_row_A + 2][load_a_col_A] = 0.0f;
                As[load_a_row_A + 3][load_a_col_A] = 0.0f;
            }

            // Load B
            if (k + load_b_row < K && bx * GEMM_BN + load_b_col < N) {
                ldg_b = *reinterpret_cast<const float4*>(&B_ptr[load_b_col * ldb + k + load_b_row]);
                Bs[load_b_row + 0][load_b_col] = ldg_b.x;
                Bs[load_b_row + 1][load_b_col] = ldg_b.y;
                Bs[load_b_row + 2][load_b_col] = ldg_b.z;
                Bs[load_b_row + 3][load_b_col] = ldg_b.w;
            }
            else {
                Bs[load_b_row + 0][load_b_col] = 0.0f;
                Bs[load_b_row + 1][load_b_col] = 0.0f;
                Bs[load_b_row + 2][load_b_col] = 0.0f;
                Bs[load_b_row + 3][load_b_col] = 0.0f;
            }

            __syncthreads();

            // Compute (Identical to NN)
#pragma unroll
            for (int bk = 0; bk < GEMM_BK; ++bk) {
#pragma unroll
                for (int i = 0; i < 4; ++i) reg_a[i] = As[bk][ty_c * 4 + i];
#pragma unroll
                for (int i = 0; i < 4; ++i) reg_b[i] = Bs[bk][tx_c * 4 + i];

#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        thread_results[i][j] += reg_a[i] * reg_b[j];
                    }
                }
            }
            __syncthreads();
        }

        // Store
        int global_c_row = by * GEMM_BM + ty_c * 4;
        int global_c_col = bx * GEMM_BN + tx_c * 4;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = global_c_col + j;
            if (col < N) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int row = global_c_row + i;
                    if (row < M) {
                        float val = alpha * thread_results[i][j];
                        if (beta != 0.0f) val += beta * C[col * ldc + row];
                        C[col * ldc + row] = val;
                    }
                }
            }
        }
    }


    // ============================================================================
    // QR KERNELS
    // ============================================================================

    __global__
        void kernel_single_column_householder(float* d_A, int m, int col_idx, float* d_tau) {
        int tid = threadIdx.x;
        int len = m - col_idx;
        float* col_ptr = &d_A[col_idx * m + col_idx];

        float local_sum = 0.0f;
        for (int i = tid; i < len; i += blockDim.x) {
            float val = col_ptr[i];
            local_sum += val * val;
        }
        local_sum = blockReduceSum(local_sum);

        __shared__ float smem_tau;
        __shared__ float smem_beta;
        __shared__ float smem_r_kk;

        if (tid == 0) {
            float norm = sqrtf(local_sum);
            float x0 = col_ptr[0];

            if (norm < EPSILON) {
                d_tau[col_idx] = 0.0f;
                smem_tau = 0.0f;
                smem_beta = 0.0f;
                smem_r_kk = x0;
            }
            else {
                float alpha = (x0 >= 0.0f) ? -norm : norm;
                float v0 = x0 - alpha;
                if (fabsf(v0) < EPSILON * fabsf(norm)) {
                    d_tau[col_idx] = 0.0f;
                    smem_tau = 0.0f;
                    smem_beta = 0.0f;
                    smem_r_kk = alpha;
                }
                else {
                    float beta = 1.0f / v0;
                    float tau = 2.0f * v0 * v0 / (v0 * v0 + (local_sum - x0 * x0));
                    d_tau[col_idx] = tau;
                    smem_tau = tau;
                    smem_beta = beta;
                    smem_r_kk = alpha;
                }
            }
        }
        __syncthreads();

        float beta = smem_beta;
        float tau = smem_tau;

        if (tid == 0) col_ptr[0] = smem_r_kk;

        if (fabsf(tau) > EPSILON) {
            for (int i = tid + 1; i < len; i += blockDim.x) col_ptr[i] *= beta;
        }
    }

    __global__
        void kernel_apply_reflector_single_target(float* d_A, int m, int reflector_col, int target_col_base, float* d_tau) {
        int target_col = target_col_base + blockIdx.x;
        int tid = threadIdx.x;
        int len = m - reflector_col;

        float tau = d_tau[reflector_col];
        if (fabsf(tau) < EPSILON) return;

        float* v_ptr = &d_A[reflector_col * m + reflector_col];
        float* target_ptr = &d_A[target_col * m + reflector_col];

        float dot = 0.0f;
        if (tid == 0) dot = target_ptr[0];

        for (int i = tid + 1; i < len; i += blockDim.x) dot += v_ptr[i] * target_ptr[i];

        dot = blockReduceSum(dot);
        __shared__ float smem_dot;
        if (tid == 0) smem_dot = dot;
        __syncthreads();

        float val = smem_dot * tau;
        if (tid == 0) target_ptr[0] -= val;
        for (int i = tid + 1; i < len; i += blockDim.x) target_ptr[i] -= val * v_ptr[i];
    }

    __global__
        void kernel_compute_T_parallel(float* d_A, float* d_T, const float* d_tau,
            int m, int k, int b_size) {
        int i = blockIdx.y;
        int j = blockIdx.x;
        int tid = threadIdx.x;

        if (i > j || i >= b_size || j >= b_size) return;

        int col_i = k + i;
        int col_j = k + j;
        float tau_j = d_tau[col_j];

        if (i == j) {
            if (tid == 0) d_T[j * b_size + i] = tau_j;
            return;
        }

        if (fabsf(tau_j) < EPSILON) {
            if (tid == 0) d_T[j * b_size + i] = 0.0f;
            return;
        }

        float dot = 0.0f;
        if (tid == 0) {
            if (col_j == col_i) dot = 1.0f;
            else dot = d_A[col_i * m + col_j];
        }

        for (int abs_row = col_j + 1 + tid; abs_row < m; abs_row += blockDim.x) {
            float v_i_val = d_A[col_i * m + abs_row];
            float v_j_val = d_A[col_j * m + abs_row];
            dot += v_i_val * v_j_val;
        }

        dot = blockReduceSum(dot);
        if (tid == 0) d_T[j * b_size + i] = -tau_j * dot;
    }

    __global__
        void kernel_finish_T_column(float* d_T, int b_size, int j_col) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= j_col) return;
        float sum = 0.0f;
#pragma unroll 8
        for (int k = i; k < j_col; k++) {
            sum += d_T[k * b_size + i] * d_T[j_col * b_size + k];
        }
        d_T[j_col * b_size + i] = sum;
    }

    __global__
        void kernel_expand_V_coalesced(float* d_A, float* d_V_expanded, int m, int k, int b_size) {
        int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int m_sub = m - k;
        int total_elements = m_sub * b_size;

        if (global_idx >= total_elements) return;

        int row = global_idx % m_sub;
        int col = global_idx / m_sub;

        if (row == 0) d_V_expanded[col * m_sub] = 1.0f;
        else d_V_expanded[col * m_sub + row] = d_A[(k + col) * m + k + row];
    }

    // ============================================================================
    // HOST FUNCTIONS
    // ============================================================================

    int compute_optimal_block_size(int m, int n) {
        int max_possible = std::min(m, n);
        // GEMM_BM = 64. Must use 64.
        return std::min(64, max_possible);
    }

    void panel_qr_gpu_optimized(GPUMatrix& A, int k, int b_size, float* d_tau) {
        int m = A.m;
        for (int j = 0; j < b_size; j++) {
            int current_col = k + j;
            kernel_single_column_householder << <1, 256, 0, g_stream_panel >> > (A.d_data, m, current_col, d_tau);

            int trail_cols = b_size - 1 - j;
            if (trail_cols > 0) {
                int next_col = current_col + 1;
                kernel_apply_reflector_single_target << <trail_cols, 256, 0, g_stream_panel >> > (
                    A.d_data, m, current_col, next_col, d_tau
                    );
            }
        }
    }

    void compute_T_optimized(const GPUMatrix& A, int k, int b_size,
        const float* d_tau, float* d_T, float* d_T_temp) {
        dim3 grid_T(b_size, b_size);
        dim3 block_T(256);
        kernel_compute_T_parallel << <grid_T, block_T, 0, g_stream_panel >> > (A.d_data, d_T, d_tau, A.m, k, b_size);

        for (int j = 1; j < b_size; j++) {
            int threads = 256;
            int blocks = (j + threads - 1) / threads;
            kernel_finish_T_column << <blocks, threads, 0, g_stream_panel >> > (d_T, b_size, j);
        }
    }

    void apply_block_reflector_optimized(GPUMatrix& A, int k, int b_size,
        int trail_start, int trail_cols,
        float* d_T, float* d_W, float* d_W2,
        float* d_V_expanded) {
        int m = A.m;
        int m_sub = m - k;
        if (trail_cols <= 0) return;

        // 1. Expand V
        int total_elements = m_sub * b_size;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        kernel_expand_V_coalesced << <blocks, threads, 0, g_stream_update >> > (A.d_data, d_V_expanded, m, k, b_size);

        // 2. W = V^T * C
        dim3 block_gemm(256);
        dim3 grid_gemm_tn((trail_cols + GEMM_BN - 1) / GEMM_BN,
            (b_size + GEMM_BM - 1) / GEMM_BM);

        kernel_gemm_tn_vec << <grid_gemm_tn, block_gemm, 0, g_stream_update >> > (
            b_size, trail_cols, m_sub,
            1.0f, d_V_expanded, m_sub,
            A.d_data + trail_start * m + k, m,
            0.0f, d_W, b_size
            );

        // 3. W2 = T^T * W
        kernel_gemm_tn_vec << <grid_gemm_tn, block_gemm, 0, g_stream_update >> > (
            b_size, trail_cols, b_size,
            1.0f, d_T, b_size,
            d_W, b_size,
            0.0f, d_W2, b_size
            );

        // 4. C = C - V * W2
        dim3 grid_gemm_nn((trail_cols + GEMM_BN - 1) / GEMM_BN,
            (m_sub + GEMM_BM - 1) / GEMM_BM);

        kernel_gemm_nn_vec << <grid_gemm_nn, block_gemm, 0, g_stream_update >> > (
            m_sub, trail_cols, b_size,
            -1.0f, d_V_expanded, m_sub,
            d_W2, b_size,
            1.0f, A.d_data + trail_start * m + k, m
            );
    }

    void qr_decomposition(GPUMatrix& A, std::vector<float>& h_tau, int block_size) {
        int m = A.m;
        int n = A.n;
        int min_dim = (m < n) ? m : n;

        if (block_size <= 0) block_size = compute_optimal_block_size(m, n);
        else block_size = std::min(block_size, min_dim);

	//	std::cout << "   Block size: " << block_size << std::endl;

        h_tau.resize(min_dim);
        init_resources();
        A.CopyToDevice();

        float* d_tau, * d_T, * d_T_temp, * d_W, * d_W2, * d_V_expanded;
        checkCudaErrors(cudaMalloc(&d_tau, min_dim * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_T, block_size * block_size * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_T_temp, block_size * block_size * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_W, block_size * n * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_W2, block_size * n * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_V_expanded, m * block_size * sizeof(float)));

        for (int k = 0; k < min_dim; k += block_size) {
            int b_size = std::min(block_size, min_dim - k);

            if (k > 0) cudaStreamSynchronize(g_stream_update);

            panel_qr_gpu_optimized(A, k, b_size, d_tau);

            checkCudaErrors(cudaMemsetAsync(d_T, 0, block_size * block_size * sizeof(float), g_stream_panel));
            compute_T_optimized(A, k, b_size, d_tau, d_T, d_T_temp);

            if (k + b_size < n) {
                cudaEvent_t event;
                cudaEventCreate(&event);
                cudaEventRecord(event, g_stream_panel);
                cudaStreamWaitEvent(g_stream_update, event, 0);
                cudaEventDestroy(event);

                int trail_start = k + b_size;
                int trail_cols = n - k - b_size;
                apply_block_reflector_optimized(A, k, b_size, trail_start, trail_cols,
                    d_T, d_W, d_W2, d_V_expanded);
            }
        }

        cudaStreamSynchronize(g_stream_panel);
        cudaStreamSynchronize(g_stream_update);

        checkCudaErrors(cudaMemcpy(h_tau.data(), d_tau, min_dim * sizeof(float), cudaMemcpyDeviceToHost));
        A.CopyToHost();

        cudaFree(d_tau); cudaFree(d_T); cudaFree(d_T_temp);
        cudaFree(d_W); cudaFree(d_W2); cudaFree(d_V_expanded);
    }

    void extract_R(const GPUMatrix& A, GPUMatrix& R) {
        R.SetZero();
        for (int j = 0; j < A.n; j++) {
            for (int i = 0; i <= j && i < A.m; i++) {
                R(i, j) = A(i, j);
            }
        }
        R.CopyToDevice();
    }

    void extract_Q(const GPUMatrix& A, const std::vector<float>& h_tau, GPUMatrix& Q) {
        int m = A.m;
        int min_dim = (m < A.n) ? m : A.n;
        Q.SetZero();
        for (int i = 0; i < m; i++) Q(i, i) = 1.0f;
        Q.CopyToDevice();

        for (int j = min_dim - 1; j >= 0; j--) {
            if (fabsf(h_tau[j]) < EPSILON) continue;
            int v_len = m - j;
            std::vector<float> v(v_len);
            v[0] = 1.0f;
            for (int i = 1; i < v_len; i++) v[i] = A(j + i, j);

            std::vector<float> vTQ(m - j);
            for (int col = j; col < m; col++) {
                float dot = 0.0f;
                for (int row = j; row < m; row++) dot += v[row - j] * Q(row, col);
                vTQ[col - j] = dot;
            }

            for (int row = j; row < m; row++) {
                for (int col = j; col < m; col++) {
                    Q(row, col) -= h_tau[j] * v[row - j] * vTQ[col - j];
                }
            }
        }
        Q.CopyToDevice();
    }
}