#include "householder.cuh"
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

namespace Householder {

    // ============================================================================
    // DEVICE HELPER: Warp Reduction
    // Performs efficient parallel reduction within GPU warps (sums values across threads)
    // ============================================================================
    __inline__ __device__ float warpReduceSum(float val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }

    // ============================================================================
    // KERNEL 1: Prepare Householder Vector
    // ============================================================================
    __global__
        void kernel_prepare_hh(float* d_A, int m, int k, float* d_tau, float* d_beta)
    {
        int tid = threadIdx.x;
        int lane = tid % WARP_SIZE;
        int warpId = tid / WARP_SIZE;

        int len = m - k;
        float* col_ptr = &d_A[k * m + k]; // Pointer to A(k,k)

        // 1. Parallel Reduction (Sum of Squares)
        float local_sum = 0.0f;
        for (int i = tid; i < len; i += blockDim.x) {
            float val = col_ptr[i];
            local_sum += val * val;
        }

        local_sum = warpReduceSum(local_sum);

        __shared__ float shared_sums[32];
        if (lane == 0) shared_sums[warpId] = local_sum;
        __syncthreads();

        float norm_sq = 0.0f;
        if (tid < (blockDim.x / WARP_SIZE)) {
            norm_sq = shared_sums[lane];
        }
        if (warpId == 0) {
            norm_sq = warpReduceSum(norm_sq);
        }

        __shared__ float smem_beta;
        __shared__ float smem_tau;

        if (tid == 0) {
            float norm = sqrtf(norm_sq);
            float alpha = 0.0f;
            float beta = 0.0f;
            float tau = 0.0f;
            float x0 = col_ptr[0];

            if (norm == 0.0f) {
                alpha = 0.0f; beta = 0.0f; tau = 0.0f;
            }
            else {
                alpha = (x0 >= 0.0f) ? -norm : norm;
                float v0 = x0 - alpha;
                beta = 1.0f / v0;
                tau = (alpha - x0) / alpha;
            }

            *d_tau = tau;
            *d_beta = beta;
            col_ptr[0] = alpha; // Store R(k,k)

            smem_beta = beta;
            smem_tau = tau;
        }
        __syncthreads();

        float beta = smem_beta;
        float tau = smem_tau;

        // 2. Scale vector v and store in A
        if (std::abs(tau) > 1e-10f) {
            for (int i = tid; i < len; i += blockDim.x) {
                if (i > 0) {
                    col_ptr[i] *= beta;
                }
            }
        }
    }

    // ============================================================================
    // KERNEL 2: Compute Dot Products w = v^T * A_trailing
    // ============================================================================
    __global__
        void kernel_compute_dots(float* d_A, int m, int k, int n, float* d_w)
    {
        int j_local = blockIdx.x;
        int j_global = k + 1 + j_local;

        if (j_global >= n) return;

        int len = m - k;
        int tid = threadIdx.x;

        float* col_ptr_A = &d_A[j_global * m + k];
        float* col_ptr_v = &d_A[k * m + k];

        float dot = 0.0f;

        for (int i = tid; i < len; i += blockDim.x) {
            float val_A = col_ptr_A[i];
            float val_v;
            if (i == 0) val_v = 1.0f;
            else val_v = col_ptr_v[i];

            dot += val_v * val_A;
        }

        dot = warpReduceSum(dot);

        __shared__ float sdata[32];
        int lane = tid % WARP_SIZE;
        int warpId = tid / WARP_SIZE;

        if (lane == 0) sdata[warpId] = dot;
        __syncthreads();

        if (tid == 0) {
            float block_dot = 0.0f;
            int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
            for (int w = 0; w < num_warps; w++) block_dot += sdata[w];
            d_w[j_local] = block_dot;
        }
    }

    // ============================================================================
    // KERNEL 3: Rank-1 Update (OPTIMIZED FOR COALESCING)
    // ============================================================================
    // Grid dim x -> M (Rows)
    // Grid dim y -> N (Cols)
    // threadIdx.x -> Rows (Fastest dimension for Coalescing)
    __global__
        void kernel_update_trailing(float* d_A, int m, int k, int n, float* d_tau, float* d_w)
    {
        // Map threadIdx.x to ROWS (contiguous memory - susjedna memorija)
        int row_rel = blockIdx.x * blockDim.x + threadIdx.x;
        // Map blockIdx.y to COLS
        int col_rel = blockIdx.y;

        int n_trailing = n - (k + 1);
        int m_trailing = m - k;

        if (row_rel >= m_trailing || col_rel >= n_trailing) return;

        int row_global = k + row_rel;
        int col_global = (k + 1) + col_rel;

        float tau = *d_tau;
        float w_val = d_w[col_rel];

        float v_val;
        if (row_rel == 0) {
            v_val = 1.0f;
        }
        else {
            v_val = d_A[k * m + row_global];
        }

        d_A[col_global * m + row_global] -= tau * v_val * w_val;
    }

    // ============================================================================
    // QR DECOMPOSITION
    // ============================================================================
    void qr_decomposition(GPUMatrix& A, std::vector<float>& h_tau)
    {
        int m = A.m;
        int n = A.n;
        int min_dim = std::min(m, n);

        h_tau.resize(min_dim);

        float* d_tau_array;
        float* d_w;
        float* d_beta;

        checkCudaErrors(cudaMalloc(&d_tau_array, min_dim * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_w, n * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_beta, sizeof(float)));

        // Pre-configure update grid
        int block_dim_x = 256;

        for (int k = 0; k < min_dim; k++) {

            float* d_tau_curr = d_tau_array + k;
            int v_len = m - k;
            int n_trailing = n - (k + 1);

            // 1. Prepare
            int num_blocks_hh = (v_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_prepare_hh << <num_blocks_hh, BLOCK_SIZE >> > (A.d_data, m, k, d_tau_curr, d_beta);

            if (k == n - 1) continue;

            // 2. Dots
            kernel_compute_dots << <n_trailing, BLOCK_SIZE >> > (A.d_data, m, k, n, d_w);

            // 3. Update (Optimized)
            // Grid X handles Rows, Grid Y handles Cols
            dim3 grid_dim(
                (v_len + block_dim_x - 1) / block_dim_x,
                n_trailing
            );
            kernel_update_trailing << <grid_dim, block_dim_x >> > (A.d_data, m, k, n, d_tau_curr, d_w);
        }

        checkCudaErrors(cudaMemcpy(h_tau.data(), d_tau_array, min_dim * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(d_tau_array);
        cudaFree(d_w);
        cudaFree(d_beta);
    }

    // ============================================================================
    // EXTRACTION KERNELS
    // ============================================================================
    __global__
        void kernel_init_identity(float* d_Q, int m) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m * m) {
            int row = idx % m;
            int col = idx / m;
            d_Q[idx] = (row == col) ? 1.0f : 0.0f;
        }
    }

    __global__
        void kernel_compute_dots_Q(float* d_Q, float* d_A, int m, int k, float* d_w) {
        int col = blockIdx.x;
        if (col >= m) return;

        int tid = threadIdx.x;
        float dot = 0.0f;

        for (int row = k + tid; row < m; row += blockDim.x) {
            float q_val = d_Q[col * m + row];
            float v_val;
            if (row == k) v_val = 1.0f;
            else v_val = d_A[k * m + row];

            dot += v_val * q_val;
        }

        dot = warpReduceSum(dot);
        __shared__ float sdata[32];
        if ((tid % 32) == 0) sdata[tid / 32] = dot;
        __syncthreads();
        if (tid == 0) {
            float bdot = 0.0f;
            for (int i = 0; i < (blockDim.x / 32); i++) bdot += sdata[i];
            d_w[col] = bdot;
        }
    }

    // Optimized Update for Q (Coalesced)
    __global__
        void kernel_update_Q(float* d_Q, float* d_A, int m, int k, float tau, float* d_w) {
        // Map threadIdx.x to ROWS
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y;

        if (row < k || row >= m || col >= m) return;

        float w_val = d_w[col];
        float v_val = (row == k) ? 1.0f : d_A[k * m + row];

        d_Q[col * m + row] -= tau * v_val * w_val;
    }

    void extract_Q(const GPUMatrix& A, const std::vector<float>& h_tau, GPUMatrix& Q)
    {
        int m = A.m;
        int n = A.n;
        int min_dim = std::min(m, n);

        int blocks = (m * m + 255) / 256;
        kernel_init_identity << <blocks, 256 >> > (Q.d_data, m);

        float* d_w;
        checkCudaErrors(cudaMalloc(&d_w, m * sizeof(float)));

        int block_dim_x = 256;

        for (int k = min_dim - 1; k >= 0; k--) {
            float tau = h_tau[k];
            if (std::abs(tau) < 1e-10f) continue;

            // 1. Dots
            kernel_compute_dots_Q << <m, 256 >> > (Q.d_data, A.d_data, m, k, d_w);

            // 2. Update Q
            // Grid X handles Rows, Grid Y handles Cols
            dim3 gd((m + block_dim_x - 1) / block_dim_x, m);
            kernel_update_Q << <gd, block_dim_x >> > (Q.d_data, A.d_data, m, k, tau, d_w);
        }

        cudaFree(d_w);
    }

    void extract_R(const GPUMatrix& A, GPUMatrix& R)
    {
        R.SetZero();

        std::vector<float> temp_h(A.m * A.n);
        checkCudaErrors(cudaMemcpy(temp_h.data(), A.d_data, A.m * A.n * sizeof(float), cudaMemcpyDeviceToHost));

        for (int j = 0; j < A.n; j++) {
            for (int i = 0; i <= j && i < A.m; i++) {
                R.h_data[j * R.m + i] = temp_h[j * A.m + i];
            }
        }
        R.CopyToDevice();
    }
}