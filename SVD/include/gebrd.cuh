// gebrd.cuh — Blocked BLAS-3 bidiagonalization
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>

/// Blocked bidiagonalization: B = U1^T * A * V1 (in-place on A).
void gebrd_merged_rank2b(
    cublasHandle_t handle,
    int m, int n, int lda,
    double* A,
    double* d, double* e,
    double* tauq, double* taup,
    double* work,
    int nb);

size_t gebrd_workspace_size(int m, int n, int nb);
