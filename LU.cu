%%writefile LU.cu

// ===================== LU.cu =====================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

using real = double;

// ===================== Provjera greške =====================
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
static void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line
                  << " code=" << (int)result << " \"" << func << "\"\n"
                  << "Error string: " << cudaGetErrorString(result) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

#define checkCublasErrors(val) check_cublas((val), #val, __FILE__, __LINE__)
static void check_cublas(cublasStatus_t stat, const char* func, const char* file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error at " << file << ":" << line
                  << " status=" << (int)stat << " \"" << func << "\"\n";
        std::exit(EXIT_FAILURE);
    }
}

#define checkCusolverErrors(val) check_cusolver((val), #val, __FILE__, __LINE__)
static void check_cusolver(cusolverStatus_t stat, const char* func, const char* file, int line) {
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER Error at " << file << ":" << line
                  << " status=" << (int)stat << " \"" << func << "\"\n";
        std::exit(EXIT_FAILURE);
    }
}


// ============================================================================
//                            POMOĆNE FUNKCIJE
// ============================================================================

static std::vector<real> make_test_matrix(int n) {
    std::vector<real> A((size_t)n * (size_t)n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[(size_t)i * n + j] = (i == j) ? 1e6 : (1.0 + 0.001 * (i + j));
    return A;
}

static float time_one_run_ms(void (*lu_func)(const real*, real*, real*, int),
                             const real* dA, real* dL, real* dU, int n) {
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    lu_func(dA, dL, dU, n);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    return ms;
}

#ifndef TILE
#define TILE 16
#endif

__global__ void k_matmul_rm(const real* A, const real* B, real* C, int n) {
    __shared__ real As[TILE][TILE];
    __shared__ real Bs[TILE][TILE];

    int row = (int)blockIdx.y * TILE + (int)threadIdx.y;
    int col = (int)blockIdx.x * TILE + (int)threadIdx.x;

    real sum = 0.0;
    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + (int)threadIdx.x;
        int b_row = t * TILE + (int)threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? A[(size_t)row * n + a_col] : 0.0;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < n && col < n) ? B[(size_t)b_row * n + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[(size_t)row * n + col] = sum;
}

__device__ inline double atomicMaxDouble(double* addr, double val) {
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long old = *a, assumed;
    while (true) {
        assumed = old;
        double oldv = __longlong_as_double(assumed);
        if (oldv >= val) break;
        old = atomicCAS(a, assumed, __double_as_longlong(val));
        if (old == assumed) break;
    }
    return __longlong_as_double(old);
}

__global__ void k_max_abs_diff_linear(const real* A, const real* B, size_t total, double* dMax) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        double diff = fabs((double)A[t] - (double)B[t]);
        atomicMaxDouble(dMax, diff);
    }
}

static double max_abs_diff_gpu_rm(const real* dA_rm, const real* dL_rm, const real* dU_rm, int n) {
    size_t total = (size_t)n * (size_t)n;

    real* dLU = nullptr;
    double* dMax = nullptr;
    checkCudaErrors(cudaMalloc(&dLU, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dMax, sizeof(double)));

    double zero = 0.0;
    checkCudaErrors(cudaMemcpy(dMax, &zero, sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    k_matmul_rm<<<grid, block>>>(dL_rm, dU_rm, dLU, n);
    checkCudaErrors(cudaGetLastError());

    int threads = 256;
    int blocks  = (int)std::min<size_t>(65535, (total + threads - 1) / threads);
    k_max_abs_diff_linear<<<blocks, threads>>>(dA_rm, dLU, total, dMax);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());

    double hMax = 0.0;
    checkCudaErrors(cudaMemcpy(&hMax, dMax, sizeof(double), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dLU));
    checkCudaErrors(cudaFree(dMax));
    return hMax;
}


// ============================================================================
//                          NAIVNA IMPLEMENTACIJA
// ============================================================================

__global__ void k_zero_LU(real* L, real* U, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        L[t] = 0.0;
        U[t] = 0.0;
    }
}

__global__ void k_U_row(const real* A, const real* L, const real* U, real* Uout, int n, int k) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (int j = tid; j < n; j += (int)blockDim.x * (int)gridDim.x) {
        if (j < k) continue;
        real sum = 0.0;
        for (int p = 0; p < k; ++p) sum += L[(size_t)k * n + p] * U[(size_t)p * n + j];
        Uout[(size_t)k * n + j] = A[(size_t)k * n + j] - sum;
    }
}

__global__ void k_L_col(const real* A, const real* U, real* Lout, int n, int k) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (int i = tid; i < n; i += (int)blockDim.x * (int)gridDim.x) {
        if (i < k) continue;
        size_t in = (size_t)i * n;
        if (i == k) {
            Lout[in + k] = 1.0;
        } else {
            real sum = 0.0;
            for (int p = 0; p < k; ++p) sum += Lout[in + p] * U[(size_t)p * n + k];
            real piv = U[(size_t)k * n + k];
            Lout[in + k] = (A[in + k] - sum) / piv;
        }
    }
}

void LU_naivna_gpu(const real* dA, real* dL, real* dU, int n) {
    int threads = 256;
    int blocks0 = (int)std::min<size_t>(65535, ((size_t)n * (size_t)n + threads - 1) / threads);
    k_zero_LU<<<blocks0, threads>>>(dL, dU, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int blocks1 = (int)std::min<size_t>(65535, ((size_t)n + threads - 1) / threads);

    for (int k = 0; k < n; ++k) {
        k_U_row<<<blocks1, threads>>>(dA, dL, dU, dU, n, k);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        k_L_col<<<blocks1, threads>>>(dA, dU, dL, n, k);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
}


// ============================================================================
//                            CUBLAS BLOKOVSKA
// ============================================================================

__global__ void k_rm_to_cm(const real* Arm, real* Acm, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int row = (int)(t / (size_t)n);
        int col = (int)(t - (size_t)row * (size_t)n);
        Acm[(size_t)col * (size_t)n + (size_t)row] = Arm[(size_t)row * (size_t)n + (size_t)col];
    }
}

__global__ void k_cm_to_rm(const real* Acm, real* Arm, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int row = (int)(t / (size_t)n);
        int col = (int)(t - (size_t)row * (size_t)n);
        Arm[(size_t)row * (size_t)n + (size_t)col] = Acm[(size_t)col * (size_t)n + (size_t)row];
    }
}

__global__ void k_lu_panel_inplace_cm(real* A, int n, int kb, int bs) {
    int lda = n;
    int kend = kb + bs;

    for (int k = kb; k < kend; ++k) {
        for (int j = k; j < kend; ++j) {
            real sum = 0.0;
            for (int p = kb; p < k; ++p) {
                sum += A[(size_t)p * lda + k] * A[(size_t)j * lda + p];
            }
            A[(size_t)j * lda + k] = A[(size_t)j * lda + k] - sum;
        }

        real piv = A[(size_t)k * lda + k];

        for (int i = k + 1; i < kend; ++i) {
            real sum = 0.0;
            for (int p = kb; p < k; ++p) {
                sum += A[(size_t)p * lda + i] * A[(size_t)k * lda + p];
            }
            A[(size_t)k * lda + i] = (A[(size_t)k * lda + i] - sum) / piv;
        }
    }
}

__global__ void k_extract_LU_cm(const real* A, real* L, real* U, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int col = (int)(t / (size_t)n);
        int row = (int)(t - (size_t)col * (size_t)n);
        real a = A[t];

        if (row > col) {
            L[t] = a;
            U[t] = 0.0;
        } else if (row == col) {
            L[t] = 1.0;
            U[t] = a;
        } else {
            L[t] = 0.0;
            U[t] = a;
        }
    }
}

static void LU_cublas_blocked_cm_inplace(real* dA_cm, int n, int B, cublasHandle_t h) {
    int lda = n;

    for (int kb = 0; kb < n; kb += B) {
        int bs = std::min(B, n - kb);
        int kend = kb + bs;
        int trail = n - kend;

        k_lu_panel_inplace_cm<<<1, 1>>>(dA_cm, n, kb, bs);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (trail <= 0) continue;

        real* A11 = dA_cm + (size_t)kb   * lda + kb;
        real* A12 = dA_cm + (size_t)kend * lda + kb;
        real* A21 = dA_cm + (size_t)kb   * lda + kend;
        real* A22 = dA_cm + (size_t)kend * lda + kend;

        const real one = 1.0;
        const real minus_one = -1.0;

        checkCublasErrors(cublasDtrsm(
            h,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_UNIT,
            bs, trail,
            &one,
            A11, lda,
            A12, lda
        ));

        checkCublasErrors(cublasDtrsm(
            h,
            CUBLAS_SIDE_RIGHT,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            trail, bs,
            &one,
            A11, lda,
            A21, lda
        ));

        checkCublasErrors(cublasDgemm(
            h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            trail, trail, bs,
            &minus_one,
            A21, lda,
            A12, lda,
            &one,
            A22, lda
        ));

        checkCudaErrors(cudaDeviceSynchronize());
    }
}

void LU_cublas_blocked_rm(const real* dA_rm, real* dL_rm, real* dU_rm, int n, int B) {
    cublasHandle_t h;
    checkCublasErrors(cublasCreate(&h));

    size_t total = (size_t)n * (size_t)n;

    real *dA_cm = nullptr, *dL_cm = nullptr, *dU_cm = nullptr;
    checkCudaErrors(cudaMalloc(&dA_cm, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dL_cm, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dU_cm, sizeof(real) * total));

    int threads = 256;
    int blocks = (int)std::min<size_t>(65535, (total + threads - 1) / threads);

    k_rm_to_cm<<<blocks, threads>>>(dA_rm, dA_cm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    LU_cublas_blocked_cm_inplace(dA_cm, n, B, h);

    k_extract_LU_cm<<<blocks, threads>>>(dA_cm, dL_cm, dU_cm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    k_cm_to_rm<<<blocks, threads>>>(dL_cm, dL_rm, n);
    checkCudaErrors(cudaGetLastError());
    k_cm_to_rm<<<blocks, threads>>>(dU_cm, dU_rm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(dA_cm));
    checkCudaErrors(cudaFree(dL_cm));
    checkCudaErrors(cudaFree(dU_cm));

    checkCublasErrors(cublasDestroy(h));
}

static int gB = 128;
static void LU_cublas_blocked_rm_wrap(const real* dA, real* dL, real* dU, int n) {
    LU_cublas_blocked_rm(dA, dL, dU, n, gB);
}


// ============================================================================
//                            cuSOLVER (getrf)
// ============================================================================

void LU_cusolver_getrf_rm(const real* dA_rm, real* dL_rm, real* dU_rm, int n) {
    cusolverDnHandle_t sh;
    checkCusolverErrors(cusolverDnCreate(&sh));

    const int lda = n;
    const size_t total = (size_t)n * (size_t)n;

    real* dA_cm = nullptr;
    real* dL_cm = nullptr;
    real* dU_cm = nullptr;
    checkCudaErrors(cudaMalloc(&dA_cm, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dL_cm, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dU_cm, sizeof(real) * total));

    int threads = 256;
    int blocks  = (int)std::min<size_t>(65535, (total + threads - 1) / threads);

    k_rm_to_cm<<<blocks, threads>>>(dA_rm, dA_cm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int lwork = 0;
    checkCusolverErrors(cusolverDnDgetrf_bufferSize(sh, n, n, dA_cm, lda, &lwork));

    real* dWork = nullptr;
    int*  dInfo = nullptr;
    checkCudaErrors(cudaMalloc(&dWork, sizeof(real) * (size_t)lwork));
    checkCudaErrors(cudaMalloc(&dInfo, sizeof(int)));

    // NOTE: PivotArray = nullptr => bez pivotiranja
    checkCusolverErrors(cusolverDnDgetrf(
        sh, n, n,
        dA_cm, lda,
        dWork,
        nullptr,
        dInfo
    ));
    checkCudaErrors(cudaDeviceSynchronize());

    int hInfo = -999;
    checkCudaErrors(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (hInfo != 0) {
        std::cerr << "cusolverDnDgetrf failed, info=" << hInfo << "\n";
        std::exit(EXIT_FAILURE);
    }

    k_extract_LU_cm<<<blocks, threads>>>(dA_cm, dL_cm, dU_cm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    k_cm_to_rm<<<blocks, threads>>>(dL_cm, dL_rm, n);
    checkCudaErrors(cudaGetLastError());
    k_cm_to_rm<<<blocks, threads>>>(dU_cm, dU_rm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(dA_cm));
    checkCudaErrors(cudaFree(dL_cm));
    checkCudaErrors(cudaFree(dU_cm));
    checkCudaErrors(cudaFree(dWork));
    checkCudaErrors(cudaFree(dInfo));

    checkCusolverErrors(cusolverDnDestroy(sh));
}

static void LU_cusolver_getrf_rm_wrap(const real* dA, real* dL, real* dU, int n) {
    LU_cusolver_getrf_rm(dA, dL, dU, n);
}


// ============================================================================
//                       RUN HELPER + MAIN
// ============================================================================

static void run_one(const char* name,
                    void (*fn)(const real*, real*, real*, int),
                    const real* dA, real* dL, real* dU, int n,
                    float& out_ms, double& out_err) {
    out_ms  = time_one_run_ms(fn, dA, dL, dU, n);
    out_err = max_abs_diff_gpu_rm(dA, dL, dU, n);
    printf("%-20s time_ms = %12.3f  max|A-LU| = % .3e\n", name, out_ms, out_err);
}

int main(int argc, char** argv) {
    int B = 128;
    if (argc > 1) B = std::atoi(argv[1]);
    gB = B;

    // test sizes
    const int Ns[] = {512, 1024, 2048, 4096, 8192, 16384};
    const int numN = (int)(sizeof(Ns) / sizeof(Ns[0]));

    printf("Benchmark LU: moja (cuBLAS blocked, B=%d) vs cuSOLVER getrf\n", B);
    printf("--------------------------------------------------------------------------\n");
    printf("%8s  %14s  %14s  %14s  %12s  %12s\n",
           "n", "ms_moja", "ms_cuSOLVER", "pct_slower", "err_moja", "err_solv");
    printf("--------------------------------------------------------------------------\n");

    for (int t = 0; t < numN; ++t) {
        int n = Ns[t];
        size_t total = (size_t)n * (size_t)n;

        // Host matrix
        std::vector<real> hA = make_test_matrix(n);

        // Device buffers
        real *dA = nullptr, *dL = nullptr, *dU = nullptr;

        // Try allocate; if fails, skip this n
        cudaError_t stA = cudaMalloc(&dA, sizeof(real) * total);
        cudaError_t stL = cudaMalloc(&dL, sizeof(real) * total);
        cudaError_t stU = cudaMalloc(&dU, sizeof(real) * total);

        if (stA != cudaSuccess || stL != cudaSuccess || stU != cudaSuccess) {
            if (dA) cudaFree(dA);
            if (dL) cudaFree(dL);
            if (dU) cudaFree(dU);
            printf("%8d  %14s  %14s  %14s  %12s  %12s\n",
                   n, "SKIP", "SKIP", "SKIP", "SKIP", "SKIP");
            continue;
        }

        checkCudaErrors(cudaMemcpy(dA, hA.data(), sizeof(real) * total, cudaMemcpyHostToDevice));

        // Warmup (opcionalno ali preporučeno)
        {
            float ms_w = 0.0f; double err_w = 0.0;
            run_one("warm_moja",     LU_cublas_blocked_rm_wrap, dA, dL, dU, n, ms_w, err_w);
            run_one("warm_cusolver", LU_cusolver_getrf_rm_wrap, dA, dL, dU, n, ms_w, err_w);
        }

        // Real runs (mjerenje)
        float  ms_moja = 0.0f, ms_solv = 0.0f;
        double err_moja = 0.0, err_solv = 0.0;

        run_one("LU_cublas_blocked", LU_cublas_blocked_rm_wrap, dA, dL, dU, n, ms_moja, err_moja);
        run_one("LU_cusolver_getrf", LU_cusolver_getrf_rm_wrap, dA, dL, dU, n, ms_solv, err_solv);

        // percent slower relative to cuSOLVER
        double pct = 0.0;
        if (ms_solv > 0.0f) pct = ((double)ms_moja - (double)ms_solv) / (double)ms_solv * 100.0;

        printf("%8d  %14.3f  %14.3f  %13.2f%%  %12.3e  %12.3e\n",
               n, ms_moja, ms_solv, pct, err_moja, err_solv);

        checkCudaErrors(cudaFree(dA));
        checkCudaErrors(cudaFree(dL));
        checkCudaErrors(cudaFree(dU));
    }

    printf("--------------------------------------------------------------------------\n");
    printf("pct_slower > 0 => moja sporija; pct_slower < 0 => moja brza\n");

    return 0;
}