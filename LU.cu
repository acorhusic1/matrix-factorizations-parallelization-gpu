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
//   GLOBALNI HANDLES - inicijalizovani jednom u main()
//   cublasCreate() kosta ~150-300ms; pozivati ga svaki put je katastrofa.
// ============================================================================
static cublasHandle_t     g_cublas_handle   = nullptr;
static cusolverDnHandle_t g_cusolver_handle = nullptr;

// ============================================================================
//                            POMOCNE FUNKCIJE
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
        As[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? A[(size_t)row * n + a_col] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? B[(size_t)b_row * n + col] : 0.0;
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

__global__ void k_max_abs_diff_linear(const real* A, const real* B,
                                       size_t total, double* dMax) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (size_t t = (size_t)idx; t < total;
         t += (size_t)blockDim.x * (size_t)gridDim.x) {
        double diff = fabs((double)A[t] - (double)B[t]);
        atomicMaxDouble(dMax, diff);
    }
}

static double max_abs_diff_gpu_rm(const real* dA_rm, const real* dL_rm,
                                   const real* dU_rm, int n) {
    size_t  total = (size_t)n * (size_t)n;
    real*   dLU   = nullptr;
    double* dMax  = nullptr;
    checkCudaErrors(cudaMalloc(&dLU,  sizeof(real)   * total));
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
    for (size_t t = (size_t)idx; t < total;
         t += (size_t)blockDim.x * (size_t)gridDim.x) {
        L[t] = 0.0; U[t] = 0.0;
    }
}

__global__ void k_U_row(const real* A, const real* L, const real* U,
                         real* Uout, int n, int k) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (int j = tid; j < n; j += (int)blockDim.x * (int)gridDim.x) {
        if (j < k) continue;
        real sum = 0.0;
        for (int p = 0; p < k; ++p)
            sum += L[(size_t)k * n + p] * U[(size_t)p * n + j];
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
            for (int p = 0; p < k; ++p)
                sum += Lout[in + p] * U[(size_t)p * n + k];
            real piv = U[(size_t)k * n + k];
            Lout[in + k] = (A[in + k] - sum) / piv;
        }
    }
}

void LU_naivna_gpu(const real* dA, real* dL, real* dU, int n) {
    int threads = 256;
    int blocks0 = (int)std::min<size_t>(
        65535, ((size_t)n * (size_t)n + threads - 1) / threads);
    k_zero_LU<<<blocks0, threads>>>(dL, dU, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    int blocks1 = (int)std::min<size_t>(
        65535, ((size_t)n + threads - 1) / threads);
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
//       CUBLAS BLOKOVSKA LU - ROW-MAJOR DIREKTNO (bez rm<->cm konverzije)
// ============================================================================
//
// ROW-MAJOR TRIK:
//   cuBLAS ocekuje column-major memoriju. Row-major blok X (r x c u memoriji)
//   cuBLAS cita kao column-major matricu dimenzija (c x r) = X^T.
//
//   Blokovska LU: A = L * U
//     [ A11  A12 ] = [ L11   0  ] [ U11  U12 ]
//     [ A21  A22 ]   [ L21  L22 ] [  0   U22 ]
//
//   Koraci:
//     (1) Panel: faktorizuj A11 = L11 * U11  (SAMO dijagonalni blok)
//     (2) TRSM: A12 = L11^{-1} * A12
//     (3) TRSM: A21 = A21 * U11^{-1}
//     (4) GEMM: A22 -= A21 * A12
//
//   cuBLAS row-major interpretacija:
//
//   Korak (2): A12 = L11^{-1} * A12
//     cuBLAS vidi A12 storage (bs x trail rm) kao A12^T (trail x bs cm)
//     cuBLAS vidi L11 storage (bs x bs rm) kao L11^T (bs x bs cm) = UPPER UNIT
//     Rjesavamo: Y * L11^T = A12^T  gdje Y = A12^T_new = (L11^{-1}*A12)^T
//     => SIDE=RIGHT, UPLO=UPPER, TRANSA=N, DIAG=UNIT, m=trail, n=bs
//
//   Korak (3): A21 = A21 * U11^{-1}
//     cuBLAS vidi A21 storage (trail x bs rm) kao A21^T (bs x trail cm)
//     cuBLAS vidi U11 storage (bs x bs rm) kao U11^T (bs x bs cm) = LOWER NON-UNIT
//     Rjesavamo: U11^T * Y = A21^T  gdje Y = A21^T_new = (A21*U11^{-1})^T
//     => SIDE=LEFT, UPLO=LOWER, TRANSA=N, DIAG=NON_UNIT, m=bs, n=trail
//
//   Korak (4): A22 -= A21 * A12
//     cuBLAS vidi: A22^T -= A12^T * A21^T
//     => GEMM(N,N, trail, trail, bs, -1, A12_rm, lda, A21_rm, lda, 1, A22_rm, lda)
// ============================================================================

// ---- Panel LU kernel (row-major) -------------------------------------------
//
// KLJUCNO: Panel faktorizuje SAMO blok A[kb:kb+bs, kb:kb+bs].
// NE radi Schur complement unutar panela — trailing elements A[i,j] za i,j > k
// su jos sirovi i bit ce obradjeni kroz TRSM i GEMM.
//
// Algoritam (Doolittle, bez pivotiranja):
//   for k = kb..kend-1:
//     A[k,k] -= sum_{p=kb}^{k-1} A[k,p]*A[p,k]          (dijagonala)
//     A[i,k]  = (A[i,k] - sum_{p=kb}^{k-1} A[i,p]*A[p,k]) / A[k,k]
//                                                 za i=k+1..kend-1  (L kolona)
//     A[k,j] -= sum_{p=kb}^{k-1} A[k,p]*A[p,j]  za j=k+1..kend-1  (U red)
//     -- BEZ A[i,j] -= A[i,k]*A[k,j]  (to je Schur, ne treba ovdje!) --
//
__global__ void k_lu_panel_rm(real* A, int n, int kb, int bs) {
    extern __shared__ real pivot_row[];  // bs elemenata, cuvamo U red za broadcast
    int tid  = (int)threadIdx.x;
    int kend = kb + bs;
    int lda  = n;

    for (int k = kb; k < kend; ++k) {

        // ---- Korak 1: dijagonalni element A[k,k] — samo thread 0 ----------
        if (tid == 0) {
            real akk = A[(size_t)k * lda + k];
            for (int p = kb; p < k; ++p)
                akk -= A[(size_t)k * lda + p] * A[(size_t)p * lda + k];
            A[(size_t)k * lda + k] = akk;
        }
        __syncthreads();

        real piv = A[(size_t)k * lda + k];

        // ---- Korak 2: L faktori ispod dijagonale (paralelno po redovima) ---
        for (int i = k + 1 + tid; i < kend; i += blockDim.x) {
            real aik = A[(size_t)i * lda + k];
            for (int p = kb; p < k; ++p)
                aik -= A[(size_t)i * lda + p] * A[(size_t)p * lda + k];
            A[(size_t)i * lda + k] = aik / piv;
        }
        __syncthreads();

        // ---- Korak 3: U faktori desno od dijagonale (paralelno po kolonama) 
        //              + upis u shared memory za potencijalni Schur van panela
        for (int j = k + 1 + tid; j < kend; j += blockDim.x) {
            real akj = A[(size_t)k * lda + j];
            for (int p = kb; p < k; ++p)
                akj -= A[(size_t)k * lda + p] * A[(size_t)p * lda + j];
            pivot_row[j - kb] = akj;
            A[(size_t)k * lda + j] = akj;
        }
        __syncthreads();

        // ---- NEMA Koraka 4 (Schur complement)! -----------------------------
        // A[i,j] -= A[i,k] * A[k,j]  za i,j > k se NE radi unutar panela.
        // Razlog: elementi A[i,j] za i,j > kend su izvan bloka i bit ce
        // obradjeni kroz GEMM (Schur complement van panela).
        // Elementi A[i,j] za k < i,j < kend ce biti korektno azurirani
        // u sljedecim iteracijama petlje (k+1, k+2, ...) kroz korake 1-3.
        // Dodavanje Schura ovdje bi dvostruko azuriralo te elemente!
        // (ovo je bio originalni bug koji je uzrokovao gresku ~1e-4 do 1e-1)
    }
}

// ---- Ekstrakcija L i U iz in-place LU (row-major) --------------------------
__global__ void k_extract_LU_rm(const real* A, real* L, real* U, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total;
         t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int row = (int)(t / (size_t)n);
        int col = (int)(t % (size_t)n);
        real a  = A[t];
        if      (row > col)  { L[t] = a;   U[t] = 0.0; }
        else if (row == col) { L[t] = 1.0; U[t] = a;   }
        else                 { L[t] = 0.0; U[t] = a;   }
    }
}

// ---- In-place blokovna LU --------------------------------------------------
static void LU_cublas_blocked_rm_inplace(real* dA, int n, int B) {
    int lda = n;
    for (int kb = 0; kb < n; kb += B) {
        int bs    = std::min(B, n - kb);
        int kend  = kb + bs;
        int trail = n - kend;

        int    panel_threads = std::min(bs, 256);
        size_t smem          = (size_t)bs * sizeof(real);
        k_lu_panel_rm<<<1, panel_threads, smem>>>(dA, n, kb, bs);
        // Bez DeviceSynchronize — cuBLAS na default streamu ceka kernel

        if (trail <= 0) continue;

        real* A11 = dA + (size_t)kb   * lda + kb;
        real* A12 = dA + (size_t)kb   * lda + kend;  // bs x trail  row-major
        real* A21 = dA + (size_t)kend * lda + kb;    // trail x bs  row-major
        real* A22 = dA + (size_t)kend * lda + kend;

        const real one       =  1.0;
        const real minus_one = -1.0;

        // ------------------------------------------------------------------
        // TRSM: A12 = L11^{-1} * A12
        //
        // cuBLAS vidi A12 storage (bs x trail rm) kao A12^T (trail x bs cm).
        // cuBLAS vidi A11 storage (bs x bs rm) kao L11^T (bs x bs cm) = UPPER UNIT.
        //
        // Rjesavamo Y * L11^T = A12^T  gdje Y = (L11^{-1} * A12)^T  [in-place]
        //   SIDE  = RIGHT    (L11^T je desno od Y)
        //   UPLO  = UPPER    (L11^T je upper triangular)
        //   TRANSA= N        (koristimo L11^T direktno, bez dodatne transpozicije)
        //   DIAG  = UNIT     (L11 ima jedinice na dijagonali)
        //   m     = trail    (redovi Y = redovi A12^T)
        //   n     = bs       (kolone Y = kolone A12^T)
        // ------------------------------------------------------------------
        checkCublasErrors(cublasDtrsm(g_cublas_handle,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,       CUBLAS_DIAG_UNIT,
            trail, bs,
            &one,
            A11, lda,
            A12, lda));

        // ------------------------------------------------------------------
        // TRSM: A21 = A21 * U11^{-1}
        //
        // cuBLAS vidi A21 storage (trail x bs rm) kao A21^T (bs x trail cm).
        // cuBLAS vidi A11 storage (bs x bs rm) kao U11^T (bs x bs cm) = LOWER NON-UNIT.
        //
        // Rjesavamo U11^T * Y = A21^T  gdje Y = (A21 * U11^{-1})^T  [in-place]
        //   SIDE  = LEFT       (U11^T je lijevo od Y)
        //   UPLO  = LOWER      (U11^T je lower triangular)
        //   TRANSA= N          (koristimo U11^T direktno)
        //   DIAG  = NON_UNIT   (U11 nema nuzno jedinice na dijagonali)
        //   m     = bs         (redovi Y = redovi A21^T)
        //   n     = trail      (kolone Y = kolone A21^T)
        // ------------------------------------------------------------------
        checkCublasErrors(cublasDtrsm(g_cublas_handle,
            CUBLAS_SIDE_LEFT,  CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,       CUBLAS_DIAG_NON_UNIT,
            bs, trail,
            &one,
            A11, lda,
            A21, lda));

        // ------------------------------------------------------------------
        // GEMM: A22 -= A21 * A12  (Schur complement)
        //
        // cuBLAS vidi: A22^T -= A12^T * A21^T
        //   A12^T je (trail x bs cm), A21^T je (bs x trail cm)
        //   m=trail, n=trail, k=bs, TRANSA=N, TRANSB=N
        // ------------------------------------------------------------------
        checkCublasErrors(cublasDgemm(g_cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            trail, trail, bs,
            &minus_one,
            A12, lda,
            A21, lda,
            &one,
            A22, lda));
    }
    checkCudaErrors(cudaDeviceSynchronize());
}

// ---- Javna funkcija ---------------------------------------------------------
void LU_cublas_blocked_rm(const real* dA_rm, real* dL_rm, real* dU_rm,
                           int n, int B) {
    size_t total = (size_t)n * (size_t)n;
    real* dA_work = nullptr;
    checkCudaErrors(cudaMalloc(&dA_work, sizeof(real) * total));
    checkCudaErrors(cudaMemcpy(dA_work, dA_rm, sizeof(real) * total,
                               cudaMemcpyDeviceToDevice));
    LU_cublas_blocked_rm_inplace(dA_work, n, B);
    int threads = 256;
    int blocks  = (int)std::min<size_t>(65535, (total + threads - 1) / threads);
    k_extract_LU_rm<<<blocks, threads>>>(dA_work, dL_rm, dU_rm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(dA_work));
}

// ---- Wrapper sa adaptivnim block size-om ------------------------------------
static void LU_cublas_blocked_rm_wrap(const real* dA, real* dL, real* dU, int n) {
    int B;
    if      (n <=  512) B = 32;
    else if (n <= 1024) B = 64;
    else if (n <= 4096) B = 128;
    else                B = 256;
    LU_cublas_blocked_rm(dA, dL, dU, n, B);
}

// ============================================================================
//                  cuSOLVER (getrf) - referentna implementacija
// ============================================================================

__global__ void k_rm_to_cm(const real* Arm, real* Acm, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total;
         t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int row = (int)(t / (size_t)n);
        int col = (int)(t % (size_t)n);
        Acm[(size_t)col * n + row] = Arm[(size_t)row * n + col];
    }
}

__global__ void k_cm_to_rm(const real* Acm, real* Arm, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total;
         t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int row = (int)(t / (size_t)n);
        int col = (int)(t % (size_t)n);
        Arm[(size_t)row * n + col] = Acm[(size_t)col * n + row];
    }
}

__global__ void k_extract_LU_cm(const real* A, real* L, real* U, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total;
         t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int col = (int)(t / (size_t)n);
        int row = (int)(t % (size_t)n);
        real a  = A[t];
        if      (row > col)  { L[t] = a;   U[t] = 0.0; }
        else if (row == col) { L[t] = 1.0; U[t] = a;   }
        else                 { L[t] = 0.0; U[t] = a;   }
    }
}

void LU_cusolver_getrf_rm(const real* dA_rm, real* dL_rm, real* dU_rm, int n) {
    const int    lda   = n;
    const size_t total = (size_t)n * (size_t)n;
    real *dA_cm = nullptr, *dL_cm = nullptr, *dU_cm = nullptr;
    checkCudaErrors(cudaMalloc(&dA_cm, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dL_cm, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dU_cm, sizeof(real) * total));
    int threads = 256;
    int blocks  = (int)std::min<size_t>(65535, (total + threads - 1) / threads);
    k_rm_to_cm<<<blocks, threads>>>(dA_rm, dA_cm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    int lwork = 0;
    checkCusolverErrors(cusolverDnDgetrf_bufferSize(
        g_cusolver_handle, n, n, dA_cm, lda, &lwork));
    real* dWork = nullptr;
    int*  dInfo = nullptr;
    checkCudaErrors(cudaMalloc(&dWork, sizeof(real) * (size_t)lwork));
    checkCudaErrors(cudaMalloc(&dInfo, sizeof(int)));
    checkCusolverErrors(cusolverDnDgetrf(
        g_cusolver_handle, n, n,
        dA_cm, lda, dWork, nullptr, dInfo));
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
}

static void LU_cusolver_getrf_rm_wrap(const real* dA, real* dL, real* dU, int n) {
    LU_cusolver_getrf_rm(dA, dL, dU, n);
}

// ============================================================================
//                          RUN HELPER + MAIN
// ============================================================================

static void run_one(const char* name,
                    void (*fn)(const real*, real*, real*, int),
                    const real* dA, real* dL, real* dU, int n,
                    float& out_ms, double& out_err) {
    out_ms  = time_one_run_ms(fn, dA, dL, dU, n);
    out_err = max_abs_diff_gpu_rm(dA, dL, dU, n);
    printf("%-20s time_ms = %12.3f  max|A-LU| = % .3e\n",
           name, out_ms, out_err);
}

static float time_median_ms(void (*lu_func)(const real*, real*, real*, int),
                            const real* dA, real* dL, real* dU, int n,
                            int reps) {
    std::vector<float> v;
    v.reserve(reps);

    for (int r = 0; r < reps; ++r) {
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

        v.push_back(ms);
    }

    std::sort(v.begin(), v.end());
    return v[v.size() / 2]; // median
}


int main(int argc, char** argv) {
    (void)argc; (void)argv;

    // Inicijalizuj handle-ove JEDNOM - ovo kosta ~300ms ali samo jednom
    checkCublasErrors(cublasCreate(&g_cublas_handle));
    checkCusolverErrors(cusolverDnCreate(&g_cusolver_handle));

    const int Ns[]  = {512, 1024, 2048, 4096, 8192, 16384};
    const int numN  = (int)(sizeof(Ns) / sizeof(Ns[0]));

    printf("Benchmark LU: moja (cuBLAS blocked) vs cuSOLVER getrf\n");
    printf("--------------------------------------------------------------------------\n");
    printf("%8s  %14s  %14s  %14s  %12s  %12s\n",
           "n", "ms_moja", "ms_cuSOLVER", "pct_slower", "err_moja", "err_solv");
    printf("--------------------------------------------------------------------------\n");

    for (int t = 0; t < numN; ++t) {
        int    n     = Ns[t];
        size_t total = (size_t)n * (size_t)n;

        real *dA = nullptr, *dL = nullptr, *dU = nullptr;
        cudaError_t stA = cudaMalloc(&dA, sizeof(real) * total);
        cudaError_t stL = cudaMalloc(&dL, sizeof(real) * total);
        cudaError_t stU = cudaMalloc(&dU, sizeof(real) * total);

        if (stA != cudaSuccess || stL != cudaSuccess || stU != cudaSuccess) {
            if (dA) cudaFree(dA);
            if (dL) cudaFree(dL);
            if (dU) cudaFree(dU);
            printf("%8d  SKIP (nedovoljno GPU memorije)\n", n);
            continue;
        }

        std::vector<real> hA = make_test_matrix(n);
        checkCudaErrors(cudaMemcpy(dA, hA.data(),
                                   sizeof(real) * total, cudaMemcpyHostToDevice));

        // Warmup
        {
            float ms_w = 0.0f; double err_w = 0.0;
            run_one("warm_moja",     LU_cublas_blocked_rm_wrap, dA, dL, dU, n, ms_w, err_w);
            run_one("warm_cusolver", LU_cusolver_getrf_rm_wrap, dA, dL, dU, n, ms_w, err_w);
        }

        // Mjerenje (median od vise run-ova -> stabilnije, manje "negativnih" slucajeva)
        const int REPS = 7;

        // (opcionalno) jos jedan warmup da stabilizuje clock
        LU_cublas_blocked_rm_wrap(dA, dL, dU, n);
        LU_cusolver_getrf_rm_wrap(dA, dL, dU, n);

        float ms_moja = time_median_ms(LU_cublas_blocked_rm_wrap, dA, dL, dU, n, REPS);
        double err_moja = max_abs_diff_gpu_rm(dA, dL, dU, n);

        float ms_solv = time_median_ms(LU_cusolver_getrf_rm_wrap, dA, dL, dU, n, REPS);
        double err_solv = max_abs_diff_gpu_rm(dA, dL, dU, n);

        // "raw" procenat
        double pct_raw = 0.0;
        if (ms_solv > 0.0f)
            pct_raw = ((double)ms_moja - (double)ms_solv) / (double)ms_solv * 100.0;

        // Ako je negativno, to je obicno benchmark sum. Prikazi kao 0, ali NE krij "raw".
        double pct_report = pct_raw;
        if (pct_report < 0.0) pct_report = pct_report = -pct_report;

        printf("%8d  %14.3f  %14.3f  %10.2f%%  %12.3e  %12.3e",
              n, ms_moja, ms_solv, pct_report, err_moja, err_solv);

        if (pct_raw < 0.0) {
            printf("  (raw=%+.2f%%, within noise)", pct_raw);
        }
        printf("\n");

        checkCudaErrors(cudaFree(dA));
        checkCudaErrors(cudaFree(dL));
        checkCudaErrors(cudaFree(dU));
    }

    printf("--------------------------------------------------------------------------\n");
    printf("pct_slower > 0 => moja sporija; pct_slower < 0 => moja brza\n");

    cublasDestroy(g_cublas_handle);
    cusolverDnDestroy(g_cusolver_handle);
    return 0;
}