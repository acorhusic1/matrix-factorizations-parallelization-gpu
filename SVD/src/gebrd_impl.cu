// gebrd_impl.cu — Blocked bidiagonalization (merged rank-2b)
#include "gebrd.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do {                                                  \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(err)); exit(1); }                           \
} while (0)

#define CHECK_CUBLAS(call) do {                                                \
    cublasStatus_t st = (call);                                                \
    if (st != CUBLAS_STATUS_SUCCESS) {                                         \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,       \
                (int)st); exit(1); }                                           \
} while (0)

// Householder reflector (dlarfg)
static constexpr int LARFG_THREADS = 256;

// Fused double-GEMV: y -= A1*x1 + A2*x2
static constexpr int FGEMV_THREADS = 256;

__global__ void kernel_fused_gemv_colupdate(
    int m, int k,
    const double* __restrict__ A1, int lda1, const double* __restrict__ x1, int incx1,
    const double* __restrict__ A2, int lda2, const double* __restrict__ x2, int incx2,
    double* __restrict__ y, int incy)
{
    // Each block computes one row of the output
    int row = blockIdx.x;
    if (row >= m) return;
    
    const int tid = threadIdx.x;
    double sum = 0.0;
    
    // Sum over k columns: A1[row, j]*x1[j] + A2[row, j]*x2[j]
    for (int j = tid; j < k; j += FGEMV_THREADS) {
        sum += A1[row + (size_t)j * lda1] * x1[j * incx1];
        sum += A2[row + (size_t)j * lda2] * x2[j * incx2];
    }
    
    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    
    // Cross-warp reduction via shared memory
    constexpr int NWARPS = FGEMV_THREADS / 32;
    __shared__ double warp_sums[NWARPS];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    
    if (tid == 0) {
        double total = 0.0;
        for (int w = 0; w < NWARPS; w++) total += warp_sums[w];
        y[row * incy] -= total;
    }
}

// Fused dual transposed GEMV: t1 = A1^T*v, t2 = A2^T*v
__global__ void kernel_fused_dual_gemvT(
    int m, int ncols,
    const double* __restrict__ A1, int lda1,
    const double* __restrict__ A2, int lda2,
    const double* __restrict__ v,
    double* __restrict__ t1, int inct1,
    double* __restrict__ t2, int inct2)
{
    int c = blockIdx.x;
    if (c >= ncols) return;

    const int tid = threadIdx.x;
    double s1 = 0.0, s2 = 0.0;

    const double* a1col = A1 + (size_t)c * lda1;
    const double* a2col = A2 + (size_t)c * lda2;

    for (int k = tid; k < m; k += FGEMV_THREADS) {
        double vk = v[k];
        s1 += a1col[k] * vk;
        s2 += a2col[k] * vk;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        s1 += __shfl_down_sync(0xFFFFFFFF, s1, offset);
        s2 += __shfl_down_sync(0xFFFFFFFF, s2, offset);
    }

    constexpr int NWARPS = FGEMV_THREADS / 32;
    __shared__ double ws1[NWARPS], ws2[NWARPS];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) { ws1[warp] = s1; ws2[warp] = s2; }
    __syncthreads();

    if (tid == 0) {
        double tot1 = 0.0, tot2 = 0.0;
        for (int w = 0; w < NWARPS; w++) { tot1 += ws1[w]; tot2 += ws2[w]; }
        t1[c * inct1] = tot1;
        t2[c * inct2] = tot2;
    }
}

// Fused update + scale: y = (y - M1*t1 - M2^T*t2) * tau
__global__ void kernel_step3_update_scale(
    int out_len, int ncols,
    const double* __restrict__ M1, int ldm1,
    const double* __restrict__ t1, int inct1,
    const double* __restrict__ M2, int ldm2,
    const double* __restrict__ t2, int inct2,
    double* __restrict__ y, int incy,
    const double* __restrict__ d_tau)
{
    int j = blockIdx.x;
    if (j >= out_len) return;

    const int tid = threadIdx.x;
    double sum = 0.0;

    for (int c = tid; c < ncols; c += FGEMV_THREADS) {
        sum += M1[j + (size_t)c * ldm1] * t1[c * inct1];
    }

    for (int c = tid; c < ncols; c += FGEMV_THREADS) {
        sum += M2[c + (size_t)j * ldm2] * t2[c * inct2];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    constexpr int NWARPS = FGEMV_THREADS / 32;
    __shared__ double warp_sums[NWARPS];
    __shared__ double s_tau;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    if (tid == 0) s_tau = *d_tau;
    __syncthreads();

    if (tid == 0) {
        double total = 0.0;
        for (int w = 0; w < NWARPS; w++) total += warp_sums[w];
        y[j * incy] = (y[j * incy] - total) * s_tau;
    }
}

// Fused dual GEMV (one OP_T, one OP_N)
__global__ void kernel_fused_dual_gemv_TN(
    int m,          // reduction dimension (shared)
    int n1, int n2, // output dimensions for t1 and t2
    const double* __restrict__ A1, int lda1,
    const double* __restrict__ A2, int lda2,
    const double* __restrict__ v, int incv,
    double* __restrict__ t1, int inct1,
    double* __restrict__ t2, int inct2)
{
    int c = blockIdx.x;
    const int tid = threadIdx.x;

    double s1 = 0.0, s2 = 0.0;
    bool do1 = (c < n1), do2 = (c < n2);

    if (do1 && do2) {
        const double* a1col = A1 + (size_t)c * lda1;
        for (int k = tid; k < m; k += FGEMV_THREADS) {
            double vk = v[k * incv];
            s1 += a1col[k] * vk;
            s2 += A2[c + (size_t)k * lda2] * vk;
        }
    } else if (do1) {
        const double* a1col = A1 + (size_t)c * lda1;
        for (int k = tid; k < m; k += FGEMV_THREADS)
            s1 += a1col[k] * v[k * incv];
    } else if (do2) {
        for (int k = tid; k < m; k += FGEMV_THREADS)
            s2 += A2[c + (size_t)k * lda2] * v[k * incv];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        s1 += __shfl_down_sync(0xFFFFFFFF, s1, offset);
        s2 += __shfl_down_sync(0xFFFFFFFF, s2, offset);
    }

    constexpr int NWARPS = FGEMV_THREADS / 32;
    __shared__ double ws1[NWARPS], ws2[NWARPS];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) { ws1[warp] = s1; ws2[warp] = s2; }
    __syncthreads();

    if (tid == 0) {
        double tot1 = 0.0, tot2 = 0.0;
        for (int w = 0; w < NWARPS; w++) { tot1 += ws1[w]; tot2 += ws2[w]; }
        if (do1) t1[c * inct1] = tot1;
        if (do2) t2[c * inct2] = tot2;
    }
}

// Fused update + scale: y = (y - M1*t1 - M2*t2) * tau
__global__ void kernel_step6_update_scale(
    int out_len, int ncols1, int ncols2,
    const double* __restrict__ M1, int ldm1,
    const double* __restrict__ t1, int inct1,
    const double* __restrict__ M2, int ldm2,
    const double* __restrict__ t2, int inct2,
    double* __restrict__ y, int incy,
    const double* __restrict__ d_tau)
{
    int j = blockIdx.x;
    if (j >= out_len) return;

    const int tid = threadIdx.x;
    double sum = 0.0;

    for (int c = tid; c < ncols1; c += FGEMV_THREADS)
        sum += M1[j + (size_t)c * ldm1] * t1[c * inct1];

    for (int c = tid; c < ncols2; c += FGEMV_THREADS)
        sum += M2[j + (size_t)c * ldm2] * t2[c * inct2];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    constexpr int NWARPS = FGEMV_THREADS / 32;
    __shared__ double warp_sums[NWARPS];
    __shared__ double s_tau;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    if (tid == 0) s_tau = *d_tau;
    __syncthreads();

    if (tid == 0) {
        double total = 0.0;
        for (int w = 0; w < NWARPS; w++) total += warp_sums[w];
        y[j * incy] = (y[j * incy] - total) * s_tau;
    }
}

// Fused GEMV for row update
__global__ void kernel_fused_gemv_rowupdate(
    int len,       // number of output elements (np - i - 1)
    int k1,        // columns for Y term (i + 1)
    int k2,        // columns for A_top term (i), can be 0
    const double* __restrict__ Y_ptr, int ldy,
    const double* __restrict__ a_col,  int inc_a,
    const double* __restrict__ A_top,  int lda_top,
    const double* __restrict__ x_col,  int inc_x,
    double* __restrict__ out, int out_inc)
{
    int j = blockIdx.x;
    if (j >= len) return;

    const int tid = threadIdx.x;
    double sum = 0.0;

    for (int c = tid; c < k1; c += FGEMV_THREADS) {
        sum += Y_ptr[j + (size_t)c * ldy] * a_col[c * inc_a];
    }

    for (int c = tid; c < k2; c += FGEMV_THREADS) {
        sum += A_top[c + (size_t)j * lda_top] * x_col[c * inc_x];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    constexpr int NWARPS = FGEMV_THREADS / 32;
    __shared__ double warp_sums[NWARPS];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (tid == 0) {
        double total = 0.0;
        for (int w = 0; w < NWARPS; w++) total += warp_sums[w];
        out[j * out_inc] -= total;
    }
}

__global__ void kernel_larfg(int len, double* x, int incx,
                              double* tau_dst, double* beta_dst)
{
    const int tid = threadIdx.x;

    // Early-exit cases — ALL threads must take the same path (no __syncthreads)
    if (len <= 0) {
        if (tid == 0) { *tau_dst = 0.0; *beta_dst = 0.0; }
        return;
    }
    if (len == 1) {
        if (tid == 0) { *tau_dst = 0.0; *beta_dst = x[0]; x[0] = 1.0; }
        return;
    }

    // Parallel reduction for sum of squares of x[1..len-1]
    double local_ssq = 0.0;
    for (int k = 1 + tid; k < len; k += LARFG_THREADS) {
        double v = x[k * incx];
        local_ssq += v * v;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ssq += __shfl_down_sync(0xFFFFFFFF, local_ssq, offset);

    // Cross-warp reduction via shared memory
    constexpr int NWARPS = LARFG_THREADS / 32;
    __shared__ double warp_sums[NWARPS];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = local_ssq;
    __syncthreads();

    // Thread 0 reduces across warps and computes tau/beta
    __shared__ double s_scl;
    if (tid == 0) {
        double ssq = 0.0;
        for (int w = 0; w < NWARPS; w++) ssq += warp_sums[w];

        double x0 = x[0];
        if (ssq == 0.0) {
            *tau_dst  = 0.0;
            *beta_dst = x0;
            x[0]      = 1.0;
            s_scl     = 0.0;
        } else {
            double nrm  = sqrt(x0 * x0 + ssq);
            double beta = (x0 >= 0.0) ? -nrm : nrm;
            double tau  = (beta - x0) / beta;
            double scl  = 1.0 / (x0 - beta);
            *tau_dst  = tau;
            *beta_dst = beta;
            x[0]      = 1.0;
            s_scl     = scl;
        }
    }
    __syncthreads();

    // Parallel scaling of x[1..len-1]
    double scl = s_scl;
    if (scl != 0.0) {
        for (int k = 1 + tid; k < len; k += LARFG_THREADS)
            x[k * incx] *= scl;
    }
}

// Restore d/e onto A's diagonal/superdiagonal
__global__ void kernel_restore_de(int cb, int n, int ib, int lda,
                                   double* A, const double* d, const double* e)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cb) return;
    A[(ib + j) + (size_t)(ib + j) * lda] = d[ib + j];
    if (ib + j + 1 < n)
        A[(ib + j) + (size_t)(ib + j + 1) * lda] = e[ib + j];
}

// ---------------------------------------------------------------------------
size_t gebrd_workspace_size(int m, int n, int nb)
{
    size_t s = 0;
    s += (size_t)m * nb;                // X
    s += (size_t)n * nb;                // Y
    s += (size_t)m * 2 * nb;            // P for merged GEMM
    s += (size_t)n * 2 * nb;            // Q for merged GEMM
    s += 8;                             // device-side constants {one, zero, mone, ...}
    return s;
}

// ---------------------------------------------------------------------------
// labrd_panel — exact port of LAPACK dlabrd, M >= N upper bidiagonal
// All cuBLAS calls use DEVICE pointer mode — scalars are on GPU (d_const).
// kernel_larfg writes tau directly to tauq[]/taup[] and beta to d[]/e[].
// ---------------------------------------------------------------------------
static void labrd_panel(
    cublasHandle_t handle,
    int mp, int np, int nb, int lda,
    double* A,
    double* d, double* e,
    double* tauq, double* taup,
    double* X, int ldx,
    double* Y, int ldy,
    double* d_const)  // device constants: d_const[0]=1, [1]=0, [2]=-1
{
    double* d_one  = d_const;
    double* d_zero = d_const + 1;
    double* d_mone = d_const + 2;

    for (int i = 0; i < nb; i++) {
        double* Aii = A + i + (size_t)i * lda;

        // Step 1: Update column A(i:mp, i) — fused double-GEMV
        if (i > 0) {
            kernel_fused_gemv_colupdate<<<mp - i, FGEMV_THREADS>>>(
                mp - i, i,
                A + i, lda, Y + i, ldy,
                X + i, ldx, A + (size_t)i * lda, 1,
                Aii, 1);
        }

        // Step 2: Left Householder — writes tau→tauq[i], beta→d[i]
        kernel_larfg<<<1, LARFG_THREADS>>>(mp - i, Aii, 1, tauq + i, d + i);

        if (i + 1 < np) {
            // Step 3: Y(i+1:np, i) = A^T * v ...
            CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T,
                mp - i, np - i - 1, d_one,
                A + i + (size_t)(i + 1) * lda, lda,
                Aii, 1,
                d_zero, Y + (i + 1) + (size_t)i * ldy, 1));

            if (i > 0) {
                // Fused: compute t1 = A_panel^T * v, t2 = X_panel^T * v simultaneously
                kernel_fused_dual_gemvT<<<i, FGEMV_THREADS>>>(
                    mp - i, i,
                    A + i, lda,                             // A_panel
                    X + i, ldx,                             // X_panel
                    Aii,                                     // v
                    Y + (size_t)i * ldy, 1,                  // t1 → Y[0:i, i]
                    X + (size_t)i * ldx, 1);                 // t2 → X[0:i, i]

                // Fused update + scale: y = (y - Y_trail * t1 - A_top^T * t2) * tau
                kernel_step3_update_scale<<<np - i - 1, FGEMV_THREADS>>>(
                    np - i - 1, i,
                    Y + (i + 1), ldy,                        // M1 = Y_trail
                    Y + (size_t)i * ldy, 1,                  // t1
                    A + (size_t)(i + 1) * lda, lda,           // M2 = A_top (transposed access)
                    X + (size_t)i * ldx, 1,                  // t2
                    Y + (i + 1) + (size_t)i * ldy, 1,        // y
                    tauq + i);                               // tau
            } else {
                // i == 0: just scale by tau
                CHECK_CUBLAS(cublasDscal(handle, np - i - 1, tauq + i,
                    Y + (i + 1) + (size_t)i * ldy, 1));
            }

            // Step 4: Update row A(i, i+1:np)
            double* Arow = A + i + (size_t)(i + 1) * lda;

            // Step 4: Update row A(i, i+1:np) — fused row-update kernel
            kernel_fused_gemv_rowupdate<<<np - i - 1, FGEMV_THREADS>>>(
                np - i - 1,
                i + 1, i,                            // k1 = i+1, k2 = i
                Y + (i + 1), ldy,                    // Y_ptr
                A + i, lda,                          // a_col, inc
                A + (size_t)(i + 1) * lda, lda,      // A_top, lda_top
                X + i, ldx,                          // x_col, inc
                Arow, lda);                          // output, stride

            // Step 5: Right Householder — writes tau→taup[i], beta→e[i]
            kernel_larfg<<<1, LARFG_THREADS>>>(np - i - 1, Arow, (int)lda,
                                                taup + i, e + i);

            // Step 6: X(i+1:mp, i) = A*u ...
            double* ui = Arow;

            CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N,
                mp - i - 1, np - i - 1, d_one,
                A + (i + 1) + (size_t)(i + 1) * lda, lda,
                ui, lda,
                d_zero, X + (i + 1) + (size_t)i * ldx, 1));

            if (i > 0) {
                // Fused: compute t1 = Y_trail^T * u (i+1 elems) and
                //                 t2 = A_trail_top * u (i elems) simultaneously
                int nblocks_6 = (i + 1 > i) ? i + 1 : i;
                kernel_fused_dual_gemv_TN<<<nblocks_6, FGEMV_THREADS>>>(
                    np - i - 1, i + 1, i,
                    Y + (i + 1), ldy,                        // A1 (OP_T)
                    A + (size_t)(i + 1) * lda, lda,           // A2 (OP_N)
                    ui, (int)lda,                             // v, incv=lda
                    X + (size_t)i * ldx, 1,                   // t1 → X[0:i+1, i]
                    Y + (size_t)i * ldy, 1);                  // t2 → Y[0:i, i]

                // Fused update + scale: x = (x - A_left * t1 - X_panel * t2) * tau
                kernel_step6_update_scale<<<mp - i - 1, FGEMV_THREADS>>>(
                    mp - i - 1, i + 1, i,
                    A + (i + 1), lda,                         // M1 = A_left
                    X + (size_t)i * ldx, 1,                   // t1
                    X + (i + 1), ldx,                         // M2 = X_panel
                    Y + (size_t)i * ldy, 1,                   // t2
                    X + (i + 1) + (size_t)i * ldx, 1,         // x
                    taup + i);                                // tau
            } else {
                // i == 0: t1 = Y_trail^T * u, x -= A_left * t1, x *= tau
                CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T,
                    np - i - 1, i + 1, d_one,
                    Y + (i + 1), ldy,
                    ui, lda,
                    d_zero, X + (size_t)i * ldx, 1));
                CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N,
                    mp - i - 1, i + 1, d_mone,
                    A + (i + 1), lda,
                    X + (size_t)i * ldx, 1,
                    d_one, X + (i + 1) + (size_t)i * ldx, 1));
                CHECK_CUBLAS(cublasDscal(handle, mp - i - 1, taup + i,
                    X + (i + 1) + (size_t)i * ldx, 1));
            }

        } else {
            double h_zero = 0.0;
            CHECK_CUDA(cudaMemcpy(taup + i, &h_zero, sizeof(double), cudaMemcpyHostToDevice));
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel to assemble P = [V | X_trail] for merged GEMM
// V = Apanel(cb:, 0:cb),  X_trail = X(cb:, 0:cb)
// P is mt × 2*cb,  ldp = mt
// ---------------------------------------------------------------------------
__global__ void kernel_assemble_P(int mt, int cb, int lda, int ldx,
                                    const double* V, const double* X_trail,
                                    double* P, int ldp)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= mt) return;
    for (int j = 0; j < cb; j++)
        P[r + (size_t)j * ldp] = V[r + (size_t)j * lda];
    for (int j = 0; j < cb; j++)
        P[r + (size_t)(cb + j) * ldp] = X_trail[r + (size_t)j * ldx];
}

// ---------------------------------------------------------------------------
// Kernel to assemble Q = [Y_trail | U_cols] for merged GEMM
// Y_trail = Y(cb:, 0:cb),  U^T = Apanel(0:cb, cb:) has rows → need columns
// Q is nt × 2*cb,  ldq = nt
// ---------------------------------------------------------------------------
__global__ void kernel_assemble_Q(int nt, int cb, int lda, int ldy,
                                    const double* Y_trail, const double* UT,
                                    double* Q, int ldq)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nt) return;
    for (int j = 0; j < cb; j++)
        Q[r + (size_t)j * ldq] = Y_trail[r + (size_t)j * ldy];
    // U^T is cb × nt stored row-major in A; U_cols(r, j) = UT(j, r) = UT[j + r*lda]
    for (int j = 0; j < cb; j++)
        Q[r + (size_t)(cb + j) * ldq] = UT[j + (size_t)r * lda];
}

// ============================================================================
// gebrd_merged_rank2b — main entry point
// ============================================================================
void gebrd_merged_rank2b(
    cublasHandle_t handle,
    int m, int n, int lda,
    double* A,
    double* d, double* e,
    double* tauq, double* taup,
    double* work,
    int nb)
{
    if (m < n) {
        fprintf(stderr, "gebrd_merged_rank2b: m >= n required\n");
        return;
    }

    // Workspace layout
    double* X        = work;
    double* Y        = X + (size_t)m * nb;
    double* P        = Y + (size_t)n * nb;
    double* Q        = P + (size_t)m * 2 * nb;
    double* d_const  = Q + (size_t)n * 2 * nb;  // [one, zero, mone]

    // Initialize device constants
    {
        double h_const[3] = {1.0, 0.0, -1.0};
        CHECK_CUDA(cudaMemcpy(d_const, h_const, 3 * sizeof(double), cudaMemcpyHostToDevice));
    }

    double* d_one  = d_const;
    double* d_mone = d_const + 2;

    // Set device-pointer mode ONCE for the entire computation
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    for (int ib = 0; ib < n; ib += nb) {
        int cb = (ib + nb <= n) ? nb : (n - ib);
        int mp = m - ib;
        int np = n - ib;
        int ldx = mp;
        int ldy = np;

        CHECK_CUDA(cudaMemset(X, 0, (size_t)mp * cb * sizeof(double)));
        CHECK_CUDA(cudaMemset(Y, 0, (size_t)np * cb * sizeof(double)));

        double* Apanel = A + ib + (size_t)ib * lda;

        labrd_panel(handle, mp, np, cb, lda,
                    Apanel,
                    d + ib, e + ib,
                    tauq + ib, taup + ib,
                    X, ldx, Y, ldy,
                    d_const);

        // Trailing update: A_trail -= V*Y'^T + X_trail*U^T
        int mt = mp - cb;
        int nt = np - cb;

        if (mt > 0 && nt > 0 && cb > 0) {
            double* A_trail = A + (ib + cb) + (size_t)(ib + cb) * lda;

            // Merged rank-2b: P=[V|X], Q=[Y|U_cols], A_trail -= P*Q^T
            int ldp = mt;
            int ldq = nt;
            int threads = 256;

            kernel_assemble_P<<<(mt + threads - 1) / threads, threads>>>(
                mt, cb, lda, ldx,
                Apanel + cb, X + cb,
                P, ldp);

            kernel_assemble_Q<<<(nt + threads - 1) / threads, threads>>>(
                nt, cb, lda, ldy,
                Y + cb,
                Apanel + (size_t)cb * lda,
                Q, ldq);

            CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                mt, nt, 2 * cb, d_mone,
                P, ldp, Q, ldq,
                d_one, A_trail, lda));
        }

        // Restore diagonal/super-diagonal
        kernel_restore_de<<<(cb + 255) / 256, 256>>>(cb, n, ib, lda, A, d, e);
    }

    // Restore host-pointer mode for the caller
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}
