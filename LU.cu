%%writefile LU.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>
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


// ============================================================================
//                            POMOĆNE FUNKCIJE
// ============================================================================

/**
 * Pravi determinističku test-matricu A (row-major) sa jakom dijagonalom (1e6),
 * da LU bez pivotiranja bude numerički stabilan i da provjera bude jasna.
 */
static std::vector<real> make_test_matrix(int n) {
    std::vector<real> A((size_t)n * (size_t)n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[(size_t)i * n + j] = (i == j) ? 1e6 : (1.0 + 0.001 * (i + j));
    return A;
}


/**
 * Mjeri vrijeme izvršavanja jedne LU funkcije na GPU koristeći CUDA evente
 * (mjeri samo GPU dio između start/stop, sa sync na kraju).
 */
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


/**
 * Tiled GPU matmul (row-major): računa C = A*B koristeći shared memory pločice TILExTILE,
 * da se provjera A=L*U radi brzo na GPU.
 */

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


/**
 * Atomic max za double (CAS varijanta): omogućava da mnogo threadova sigurno
 * “glasaju” za najveću grešku (max abs diff) u jednoj globalnoj varijabli.
 */
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


/**
 * Računa max|A-B| nad linearnim nizom dužine 'total' i upisuje rezultat u dMax.
 * Svaki thread obrađuje više elemenata (striding).
 */
__global__ void k_max_abs_diff_linear(const real* A, const real* B, size_t total, double* dMax) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        double diff = fabs((double)A[t] - (double)B[t]);
        atomicMaxDouble(dMax, diff);
    }
}


/**
 * GPU provjera za row-major: prvo izračuna LU = L*U na GPU, zatim izračuna max|A-LU|
 * takođe na GPU, i vrati grešku kao double na host.
 */
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

/**
 * Nuluje (inicijalizuje) matrice L i U na GPU: L=0, U=0 (row-major).
 * Radi paralelno preko svih n*n elemenata.
 */
__global__ void k_zero_LU(real* L, real* U, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        L[t] = 0.0;
        U[t] = 0.0;
    }
}


/**
 * Računa k-ti red matrice U(bez pivotiranja) u row-major formatu.
 * Svaki thread računa više kolona j>=k: U[k,j] = A[k,j] - sum_{p=0..k-1} L[k,p]*U[p,j].
 */
__global__ void k_U_row(const real* A, const real* L, const real* U, real* Uout, int n, int k) {
    int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (int j = tid; j < n; j += (int)blockDim.x * (int)gridDim.x) {
        if (j < k) continue;
        real sum = 0.0;
        for (int p = 0; p < k; ++p) sum += L[(size_t)k * n + p] * U[(size_t)p * n + j];
        Uout[(size_t)k * n + j] = A[(size_t)k * n + j] - sum;
    }
}


/**
 * Računa k-tu kolonu matrice L(bez pivotiranja) u row-major formatu.
 * Postavlja L[k,k]=1, a za i>k: L[i,k] = (A[i,k] - sum_{p=0..k-1} L[i,p]*U[p,k]) / U[k,k].
 */
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


/**
 * "Naivna" GPU LU faktorizacija (referenca za ispravnost): iterira k=0..n-1,
 * prvo izračuna U red k, zatim L kolonu k, uz sync nakon svakog koraka.
 * Paralelizacija je po elementima reda/kolone, ali zavisnosti između k koraka ostaju serijske.
 */
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

/**
 * Konverzija matrice iz row-major (C/CUDA stil) u column-major (Fortran/cuBLAS stil)
 * praktično “transpose mapiranje” indeksa da bi cuBLAS mogao raditi direktno.
 */
__global__ void k_rm_to_cm(const real* Arm, real* Acm, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int row = (int)(t / (size_t)n);
        int col = (int)(t - (size_t)row * (size_t)n);
        Acm[(size_t)col * (size_t)n + (size_t)row] = Arm[(size_t)row * (size_t)n + (size_t)col];
    }
}


/**
 * Obrnuta konverzija: column-major -> row-major,
 * da L/U koje smo dobili u cuBLAS formatu vratimo u svoj standardni raspored.
 */
__global__ void k_cm_to_rm(const real* Acm, real* Arm, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int row = (int)(t / (size_t)n);
        int col = (int)(t - (size_t)row * (size_t)n);
        Arm[(size_t)row * (size_t)n + (size_t)col] = Acm[(size_t)col * (size_t)n + (size_t)row];
    }
}


/**
 * Faktorizuje dijagonalni blok (“panel”) A[kb:kb+bs, kb:kb+bs] in-place u column-major formatu (bez pivotiranja)
 * U se upisuje na/iznad dijagonale, L ispod dijagonale (dijagonala L je implicitno 1). Ovo je "sekvencijalni dio" blokovske LU.
 */
__global__ void k_lu_panel_inplace_cm(real* A, int n, int kb, int bs) {
    int lda = n;
    int kend = kb + bs;

    // serial kernel
    for (int k = kb; k < kend; ++k) {
        // U(k, j) for j=k..kend-1
        for (int j = k; j < kend; ++j) {
            real sum = 0.0;
            for (int p = kb; p < k; ++p) {
                // L(k,p) = A(p,k)  -> A[k*lda + p]?  (column-major: A[col*lda + row])
                // We store L below diag: A(p, k) for p<k is in column k at row p => A[k*lda + p]
                // But we need L(k,p): row k, col p => A[p*lda + k] (below diag stored in column p)
                // U(p,j): row p, col j => A[j*lda + p]
                sum += A[(size_t)p * lda + k] * A[(size_t)j * lda + p];
            }
            A[(size_t)j * lda + k] = A[(size_t)j * lda + k] - sum; // U(k,j) at (row k, col j)
        }

        real piv = A[(size_t)k * lda + k]; // U(k,k)

        // L(i,k) for i=k+1..kend-1, stored at (row i, col k) => A[k*lda + i]
        for (int i = k + 1; i < kend; ++i) {
            real sum = 0.0;
            for (int p = kb; p < k; ++p) {
                // L(i,p): (row i, col p) => A[p*lda + i]
                // U(p,k): (row p, col k) => A[k*lda + p]
                sum += A[(size_t)p * lda + i] * A[(size_t)k * lda + p];
            }
            A[(size_t)k * lda + i] = (A[(size_t)k * lda + i] - sum) / piv; // L(i,k)
        }
    }
}


/**
 * Iz in-place matrice A (koja sadrži i L i U) izvlači dvije odvojene matrice L i U (obe column-major)
 * L dobija elemente ispod dijagonale + jedinice na dijagonali, U dobija dijagonalu i iznad dijagonale.
 */
__global__ void k_extract_LU_cm(const real* A, real* L, real* U, int n) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    size_t total = (size_t)n * (size_t)n;
    for (size_t t = (size_t)idx; t < total; t += (size_t)blockDim.x * (size_t)gridDim.x) {
        int col = (int)(t / (size_t)n);
        int row = (int)(t - (size_t)col * (size_t)n);
        real a = A[t];

        if (row > col) {        // below diag
            L[t] = a;
            U[t] = 0.0;
        } else if (row == col) { // diag
            L[t] = 1.0;
            U[t] = a;
        } else {                // above diag
            L[t] = 0.0;
            U[t] = a;
        }
    }
}


/**
 * Pravi blokovski LU (bez pivotiranja) nad column-major matricom: za svaki blok radi (1) panel LU,
 * (2) TRSM za U12, (3) TRSM za L21, (4) GEMM Schur update A22 -= L21·U12. 
 * Ovo je dio koji daje ubrzanje (TRSM/GEMM su super optimizovani na GPU).
 */
static void LU_cublas_blocked_cm_inplace(real* dA_cm, int n, int B, cublasHandle_t h) {
    int lda = n;

    for (int kb = 0; kb < n; kb += B) {
        int bs = std::min(B, n - kb);
        int kend = kb + bs;
        int trail = n - kend;

        // factorize diagonal block (panel)
        k_lu_panel_inplace_cm<<<1, 1>>>(dA_cm, n, kb, bs);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (trail <= 0) continue;

        // submat pointers (column-major)
        real* A11 = dA_cm + (size_t)kb   * lda + kb;     // bs x bs
        real* A12 = dA_cm + (size_t)kend * lda + kb;     // bs x trail
        real* A21 = dA_cm + (size_t)kb   * lda + kend;   // trail x bs
        real* A22 = dA_cm + (size_t)kend * lda + kend;   // trail x trail

        const real one = 1.0;
        const real minus_one = -1.0;

        // U12 = L11^{-1} * A12  (Left, Lower, NoTrans, Unit)
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

        // L21 = A21 * U11^{-1}  (Right, Upper, NoTrans, NonUnit)
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

        // A22 -= L21 * U12
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


/**
 * Javni wrapper za program: prima A u row-major, interno prebaci u column-major, uradi blokovski LU (cuBLAS),
 * ekstrahuje L/U, pa vrati L i U nazad u row-major. Kreira/uništava cuBLAS handle i privremene buffere.
 */
void LU_cublas_blocked_rm(const real* dA_rm, real* dL_rm, real* dU_rm, int n, int B) {
    cublasHandle_t h;
    checkCublasErrors(cublasCreate(&h));

    size_t total = (size_t)n * (size_t)n;

    // internal column-major buffers
    real *dA_cm = nullptr, *dL_cm = nullptr, *dU_cm = nullptr;
    checkCudaErrors(cudaMalloc(&dA_cm, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dL_cm, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dU_cm, sizeof(real) * total));

    int threads = 256;
    int blocks = (int)std::min<size_t>(65535, (total + threads - 1) / threads);

    // rm -> cm
    k_rm_to_cm<<<blocks, threads>>>(dA_rm, dA_cm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // LU in-place on column-major
    LU_cublas_blocked_cm_inplace(dA_cm, n, B, h);

    // extract L_cm and U_cm
    k_extract_LU_cm<<<blocks, threads>>>(dA_cm, dL_cm, dU_cm, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // cm -> rm outputs
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


/**
 * Adapter da funkcija ima isti potpis kao ostali "lu_func" (bez parametra B)
 * Koristi globalni gB da bi se lako mjerilo vrijeme istim timing helperom.
 */
static int gB = 128;
static void LU_cublas_blocked_rm_wrap(const real* dA, real* dL, real* dU, int n) {
    LU_cublas_blocked_rm(dA, dL, dU, n, gB);
}


static void run_one(const char* name,
                    void (*fn)(const real*, real*, real*, int),
                    const real* dA, real* dL, real* dU, int n,
                    float& out_ms, double& out_err) {
    out_ms  = time_one_run_ms(fn, dA, dL, dU, n);
    out_err = max_abs_diff_gpu_rm(dA, dL, dU, n);
    printf("%-20s time_ms = %12.3f  max|A-LU| = % .3e\n", name, out_ms, out_err);
}



// ============================================================================
//                  MAIN: run NAIVE + CUBLAS
// ============================================================================
int main(int argc, char** argv) {
    int n = 8192;
    if (argc > 1) n = std::atoi(argv[1]);
    int B = 128;
    if (argc > 2) B = std::atoi(argv[2]);
    gB = B;

    std::vector<real> hA = make_test_matrix(n);

    real *dA=nullptr, *dL=nullptr, *dU=nullptr;
    size_t total = (size_t)n * (size_t)n;
    checkCudaErrors(cudaMalloc(&dA, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dL, sizeof(real) * total));
    checkCudaErrors(cudaMalloc(&dU, sizeof(real) * total));
    checkCudaErrors(cudaMemcpy(dA, hA.data(), sizeof(real) * total, cudaMemcpyHostToDevice));

    printf("n=%d (B=%d)\n", n, B);

    float  ms_naive = 0.0f, ms_cublas = 0.0f;
    double err_naive = 0.0,  err_cublas = 0.0;

    run_one("LU_naivna_gpu",          LU_naivna_gpu,            dA, dL, dU, n, ms_naive,  err_naive);
    run_one("LU_cublas_blocked",      LU_cublas_blocked_rm_wrap, dA, dL, dU, n, ms_cublas, err_cublas);

    // speedup (only meaningful if both correct)
    const double tol = 1e-9;
    if (err_naive < tol && err_cublas < tol && ms_cublas > 0.0f) {
        double s = (double)ms_naive / (double)ms_cublas;
        printf("speedup (cuBLAS vs naive): %.3fx\n", s);
    } else {
        printf("speedup (cuBLAS vs naive): skip (err too big or ms=0)\n");
    }

    checkCudaErrors(cudaFree(dA));
    checkCudaErrors(cudaFree(dL));
    checkCudaErrors(cudaFree(dU));
    return 0;
}