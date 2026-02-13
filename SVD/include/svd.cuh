// svd.cuh — SVD public API
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>

// orgbr — generate Q and P^T from gebrd output

/// Generate m×m orthogonal Q from gebrd's left reflectors.
void orgbr_generate_Q(
    cublasHandle_t handle,
    int m, int n, int lda,
    const double* A,
    const double* tauq,
    double* Q, int ldq,
    double* work);

size_t orgbr_q_workspace_size(int m, int n);

/// Generate n×n orthogonal P^T from gebrd's right reflectors.
void orgbr_generate_PT(
    cublasHandle_t handle,
    int m, int n, int lda,
    const double* A,
    const double* taup,
    double* PT, int ldpt,
    double* work);

size_t orgbr_pt_workspace_size(int m, int n);

// D&C SVD

/// Full SVD: A = U * diag(S) * VT  (m >= n).
/// A is destroyed. S has min(m,n) singular values.
/// U is m×m, VT is n×n.
void svd_full(
    cublasHandle_t handle,
    int m, int n, int lda,
    double* A,
    double* S,
    double* U, int ldu,
    double* VT, int ldvt,
    double* work,
    int nb);

size_t svd_workspace_size(int m, int n, int nb);
