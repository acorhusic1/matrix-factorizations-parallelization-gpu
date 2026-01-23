#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

using real = double;

// -------------------- GPU kernels (LU bez pivotiranja, in-place) --------------------
// A je row-major n x n. Nakon LU:
// - U: gornji trokut (uključujući dijagonalu)
// - L: donji trokut (ispod dijagonale), dijagonala L je 1 implicitno

__global__ void k_compute_multipliers(real* A, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
    if (i < n) {
        real pivot = A[k*n + k];
        A[i*n + k] /= pivot;
    }
}

__global__ void k_update_trailing_naive(real* A, int n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
    int i = blockIdx.y * blockDim.y + threadIdx.y + (k + 1);
    if (i < n && j < n) {
        A[i*n + j] -= A[i*n + k] * A[k*n + j];
    }
}

template<int TILE>
__global__ void k_update_trailing_tiled(real* A, int n, int k) {
    __shared__ real Uk[TILE];
    __shared__ real Lik[TILE];

    int base_i = (k + 1) + blockIdx.y * TILE;
    int base_j = (k + 1) + blockIdx.x * TILE;

    int ti = threadIdx.y;
    int tj = threadIdx.x;

    int i = base_i + ti;
    int j = base_j + tj;

    // učitaj dio kolone k (Lik) i dio reda k (Uk) u shared
    if (tj == 0) {
        if (i < n) Lik[ti] = A[i*n + k];
    }
    if (ti == 0) {
        if (j < n) Uk[tj] = A[k*n + j];
    }
    __syncthreads();

    if (i < n && j < n) {
        A[i*n + j] -= Lik[ti] * Uk[tj];
    }
}

// -------------------- Verzija 1: NAIVNA LU --------------------
void lu_naive_gpu(real* dA, int n) {
    dim3 blockMul(256);
    dim3 blockUpd(16, 16);

    for (int k = 0; k < n; ++k) {
        int rows = n - (k + 1);
        if (rows > 0) {
            int gridMul = (rows + blockMul.x - 1) / blockMul.x;
            k_compute_multipliers<<<gridMul, blockMul>>>(dA, n, k);
            CUDA_CHECK(cudaGetLastError());
        }

        int m = n - (k + 1);
        if (m > 0) {
            dim3 gridUpd((m + blockUpd.x - 1) / blockUpd.x,
                         (m + blockUpd.y - 1) / blockUpd.y);
            k_update_trailing_naive<<<gridUpd, blockUpd>>>(dA, n, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

// -------------------- Verzija 2: NAPREDNA LU (tiled update) --------------------
void lu_advanced_gpu(real* dA, int n) {
    dim3 blockMul(256);
    dim3 blockUpd(16, 16); // TILE = 16

    for (int k = 0; k < n; ++k) {
        int rows = n - (k + 1);
        if (rows > 0) {
            int gridMul = (rows + blockMul.x - 1) / blockMul.x;
            k_compute_multipliers<<<gridMul, blockMul>>>(dA, n, k);
            CUDA_CHECK(cudaGetLastError());
        }

        int m = n - (k + 1);
        if (m > 0) {
            dim3 gridUpd((m + 16 - 1) / 16, (m + 16 - 1) / 16);
            k_update_trailing_tiled<16><<<gridUpd, blockUpd>>>(dA, n, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

// -------------------- CPU helpers (test kao u notebooku: A ≈ L*U) --------------------
static void extract_LU(const std::vector<real>& A, int n, std::vector<real>& L, std::vector<real>& U) {
    L.assign(n*n, 0);
    U.assign(n*n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) L[i*n + j] = A[i*n + j];
            else if (i == j) { L[i*n + j] = 1; U[i*n + j] = A[i*n + j]; }
            else U[i*n + j] = A[i*n + j];
        }
    }
}

static void matmul_cpu(const std::vector<real>& A, const std::vector<real>& B, std::vector<real>& C, int n) {
    C.assign(n*n, 0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            real aik = A[i*n + k];
            for (int j = 0; j < n; ++j)
                C[i*n + j] += aik * B[k*n + j];
        }
}

static real max_abs_diff(const std::vector<real>& A, const std::vector<real>& B) {
    real m = 0;
    for (size_t i = 0; i < A.size(); ++i)
        m = std::max(m, (real)std::fabs(A[i] - B[i]));
    return m;
}

// jednostavna stabilna matrica (da LU bez pivotiranja “ne pukne”)
static std::vector<real> make_test_matrix(int n) {
    std::vector<real> A(n*n);
    for (int i = 0; i < n; ++i) {
        real rowsum = 0;
        for (int j = 0; j < n; ++j) {
            real v = (real)((i+1)*10 + (j+1));
            A[i*n + j] = v;
            rowsum += std::fabs(v);
        }
        A[i*n + i] += rowsum; // diagonal dominant
    }
    return A;
}

// mjeri vrijeme jedne funkcije (kao u notebook: CUDA events)
static float time_one_run_ms(void (*lu_func)(real*, int), real* dA, real* dA0, int n) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // reset input
    CUDA_CHECK(cudaMemcpy(dA, dA0, sizeof(real)*n*n, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    lu_func(dA, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main(int argc, char** argv) {
    int n = 64;        // default mali da ne timeout-a na leetgpu
    int mode = 0;      // 0=naive, 1=advanced

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) mode = std::atoi(argv[2]);

    std::vector<real> hA = make_test_matrix(n);
    std::vector<real> hA_orig = hA;

    real* dA  = nullptr;
    real* dA0 = nullptr;
    CUDA_CHECK(cudaMalloc(&dA,  sizeof(real)*n*n));
    CUDA_CHECK(cudaMalloc(&dA0, sizeof(real)*n*n));

    CUDA_CHECK(cudaMemcpy(dA0, hA.data(), sizeof(real)*n*n, cudaMemcpyHostToDevice));

    void (*func)(real*, int) = nullptr;
    const char* name = nullptr;

    if (mode == 0) { func = lu_naive_gpu; name = "naive"; }
    else          { func = lu_advanced_gpu; name = "advanced_tiled"; }

    float ms = time_one_run_ms(func, dA, dA0, n);

    // kopiraj nazad i testiraj: A ≈ L*U
    CUDA_CHECK(cudaMemcpy(hA.data(), dA, sizeof(real)*n*n, cudaMemcpyDeviceToHost));

    std::vector<real> L, U, LU;
    extract_LU(hA, n, L, U);
    matmul_cpu(L, U, LU, n);

    real err = max_abs_diff(LU, hA_orig);

    printf("n=%d mode=%s time_ms=%.3f max|A-LU|=%.3e\n", n, name, ms, (double)err);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dA0));
    return 0;
}