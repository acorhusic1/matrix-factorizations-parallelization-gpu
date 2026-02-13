// orgbr_impl.cu — Generate Q and P^T from gebrd output
#include "svd.cuh"
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

// ---------------------------------------------------------------------------
// Kernel: Apply Householder reflector to Q from the left (column-oriented)
//   Q(i:m, j) -= tau * v_i * dot(v_i, Q(i:m, j))   for each column j = i..m-1
//
// This is a rank-1 update: Q_sub -= tau * v * (v^T * Q_sub)
// We use cuBLAS DGEMV + DGER for this.
// ---------------------------------------------------------------------------

// Kernel to set Q = I (identity matrix)
__global__ void kernel_set_identity(int n, double* Q, int ldq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int row = idx % n;
    int col = idx / n;
    Q[row + (size_t)col * ldq] = (row == col) ? 1.0 : 0.0;
}

// Kernel to extract left reflector v_i from gebrd output into a vector
// v[0..i-1] = 0, v[i] = 1, v[i+1..m-1] = A(i+1:m, i)
__global__ void kernel_extract_v_left(int m, int i, const double* A, int lda, double* v)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m) return;
    if (k < i)
        v[k] = 0.0;
    else if (k == i)
        v[k] = 1.0;
    else
        v[k] = A[k + (size_t)i * lda];
}

// Kernel to extract right reflector u_i from gebrd output into a vector
// u[0..i] = 0, u[i+1] = 1, u[i+2..n-1] = A(i, i+2:n)
// The reflector is stored in row i, starting from column i+1
__global__ void kernel_extract_u_right(int n, int i, const double* A, int lda, double* u)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    if (k <= i)
        u[k] = 0.0;
    else if (k == i + 1)
        u[k] = 1.0;
    else
        u[k] = A[i + (size_t)k * lda];  // Row i, column k
}

// ---------------------------------------------------------------------------
// orgbr_generate_Q — backward accumulation
// ---------------------------------------------------------------------------
size_t orgbr_q_workspace_size(int m, int n)
{
    // Need: one vector of length m (for the reflector) + one vector of length m (for w = Q^T * v)
    return (size_t)m + (size_t)m;
}

void orgbr_generate_Q(
    cublasHandle_t handle,
    int m, int n, int lda,
    const double* A,
    const double* tauq,
    double* Q, int ldq,
    double* work)
{
    int minmn = (m < n) ? m : n;
    
    // Workspace
    double* v = work;          // reflector vector, length m
    double* w = work + m;      // w = Q_sub^T * v, length m

    // Set Q = I
    int total = m * m;
    kernel_set_identity<<<(total + 255) / 256, 256>>>(m, Q, ldq);

    // Host scalars for cuBLAS (use host pointer mode temporarily)
    cublasPointerMode_t origMode;
    CHECK_CUBLAS(cublasGetPointerMode(handle, &origMode));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    double one = 1.0, zero = 0.0;

    // Backward accumulation: for i = minmn-1, minmn-2, ..., 0
    for (int i = minmn - 1; i >= 0; i--) {
        int len = m - i;  // length of reflector

        // Extract v_i
        kernel_extract_v_left<<<(m + 255) / 256, 256>>>(m, i, A, lda, v);

        // Copy tau to host
        double tau_i;
        CHECK_CUDA(cudaMemcpy(&tau_i, tauq + i, sizeof(double), cudaMemcpyDeviceToHost));

        if (tau_i == 0.0) continue;

        // w = Q(i:m, i:m)^T * v(i:m)   →  w is (m-i) × 1
        // Q_sub = Q + i + i*ldq,  v_sub = v + i
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T,
            len, len, &one,
            Q + i + (size_t)i * ldq, ldq,
            v + i, 1,
            &zero, w, 1));

        // Q(i:m, i:m) -= tau * v(i:m) * w^T
        double neg_tau = -tau_i;
        CHECK_CUBLAS(cublasDger(handle,
            len, len, &neg_tau,
            v + i, 1,
            w, 1,
            Q + i + (size_t)i * ldq, ldq));
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, origMode));
}

// ---------------------------------------------------------------------------
// orgbr_generate_PT — backward accumulation
// ---------------------------------------------------------------------------
size_t orgbr_pt_workspace_size(int m, int n)
{
    return (size_t)n + (size_t)n;
}

void orgbr_generate_PT(
    cublasHandle_t handle,
    int m, int n, int lda,
    const double* A,
    const double* taup,
    double* PT, int ldpt,
    double* work)
{
    int minmn = (m < n) ? m : n;
    int nref = minmn - 1;  // number of right reflectors (n-1 for square)
    if (nref <= 0) {
        // 1×1 or trivial: P^T = I
        int total = n * n;
        kernel_set_identity<<<(total + 255) / 256, 256>>>(n, PT, ldpt);
        return;
    }

    // Workspace
    double* u = work;          // reflector vector, length n
    double* w = work + n;      // w = PT_sub^T * u, length n

    // Set PT = I
    int total = n * n;
    kernel_set_identity<<<(total + 255) / 256, 256>>>(n, PT, ldpt);

    cublasPointerMode_t origMode;
    CHECK_CUBLAS(cublasGetPointerMode(handle, &origMode));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    double one = 1.0, zero = 0.0;

    // Backward accumulation: for i = nref-1, ..., 0
    // G(i) = I - taup[i] * u_i * u_i^T
    // P^T = G(0) * G(1) * ... * G(nref-1)
    // Apply from the RIGHT: PT = PT * G(i), backward from i = nref-1 to 0
    // PT * G(i) = PT - tau * (PT * u) * u^T
    for (int i = nref - 1; i >= 0; i--) {
        int start = i + 1;  // reflector starts at position i+1
        int len = n - start;

        if (len <= 0) continue;

        // Extract u_i
        kernel_extract_u_right<<<(n + 255) / 256, 256>>>(n, i, A, lda, u);

        double tau_i;
        CHECK_CUDA(cudaMemcpy(&tau_i, taup + i, sizeof(double), cudaMemcpyDeviceToHost));

        if (tau_i == 0.0) continue;

        // w = PT(start:n, start:n) * u(start:n)   (right multiply: OP_N)
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N,
            len, len, &one,
            PT + start + (size_t)start * ldpt, ldpt,
            u + start, 1,
            &zero, w, 1));

        // PT(start:n, start:n) -= tau * w * u(start:n)^T
        double neg_tau = -tau_i;
        CHECK_CUBLAS(cublasDger(handle,
            len, len, &neg_tau,
            w, 1,
            u + start, 1,
            PT + start + (size_t)start * ldpt, ldpt));
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, origMode));
}
