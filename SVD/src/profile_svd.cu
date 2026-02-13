// profile_svd.cu — Nsight profiling target (no cuSOLVER)
//
// Usage:
//   profile_svd.exe [size]       (default 4096)
//
// Nsight Compute:  ncu --set full -o profile profile_svd.exe 4096
// Nsight Systems:  nsys profile -o profile profile_svd.exe 4096
#include "gebrd.cuh"
#include "svd.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CHECK_CUDA(call) do {                                               \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
        fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,         \
                cudaGetErrorString(e)); exit(1); }                          \
} while(0)

#define CHECK_CUBLAS(call) do {                                             \
    cublasStatus_t s = (call);                                              \
    if (s != CUBLAS_STATUS_SUCCESS) {                                       \
        fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int)s);\
        exit(1); }                                                          \
} while(0)

static void fill_random(double* h, int m, int n, unsigned seed = 42)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            h[i + (size_t)j * m] = dist(rng);
}

int main(int argc, char** argv)
{
    int n = 4096;
    if (argc > 1) n = atoi(argv[1]);
    if (n < 2) { fprintf(stderr, "n must be >= 2\n"); return 1; }

    int m = n;
    int nb = 128;
    int minmn = n;

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1073741824.0);
    printf("Matrix size: %d x %d\n\n", m, n);


    double vram_needed_gb = (
        (size_t)m * n +          // A
        minmn +                  // S
        (size_t)m * m +          // U
        (size_t)n * n +          // VT
        (size_t)m * n            // workspace (estimate)
    ) * sizeof(double) / 1073741824.0;
    printf("Estimated VRAM needed: %.2f GB (available: %.1f GB)\n",
           vram_needed_gb * 2.0, prop.totalGlobalMem / 1073741824.0);

    if (vram_needed_gb * 2.0 > prop.totalGlobalMem / 1073741824.0 * 0.9) {
        fprintf(stderr, "WARNING: May run out of VRAM!\n");
    }


    size_t sizeA = (size_t)m * n;
    std::vector<double> h_A(sizeA);
    fill_random(h_A.data(), m, n, 42 + m + n);

    double normA = 0;
    for (auto& v : h_A) normA += v * v;
    normA = sqrt(normA);


    double *d_A, *d_S, *d_U, *d_VT, *d_work;
    CHECK_CUDA(cudaMalloc(&d_A,  sizeA * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_S,  minmn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_U,  (size_t)m * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_VT, (size_t)n * n * sizeof(double)));

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));

    size_t ws = svd_workspace_size(m, n, nb);
    CHECK_CUDA(cudaMalloc(&d_work, ws * sizeof(double)));

    // Warmup
    printf("Running warmup...\n");
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(double), cudaMemcpyHostToDevice));
    svd_full(cublas, m, n, m, d_A, d_S, d_U, m, d_VT, n, d_work, nb);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Warmup done.\n\n");

    // Timed run
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(double), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    printf("=== Starting timed SVD (%d x %d) ===\n", m, n);
    CHECK_CUDA(cudaEventRecord(t0));

    svd_full(cublas, m, n, m, d_A, d_S, d_U, m, d_VT, n, d_work, nb);

    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, t0, t1));
    printf("=== SVD completed: %.2f ms ===\n\n", elapsed_ms);

    // Correctness check
    printf("--- Correctness check ---\n");

    std::vector<double> h_S(minmn);
    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, minmn * sizeof(double), cudaMemcpyDeviceToHost));

    // U orthogonality
    double *d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, (size_t)m * m * sizeof(double)));
    double one = 1.0, zero = 0.0;
    CHECK_CUBLAS(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        m, m, m, &one, d_U, m, d_U, m, &zero, d_temp, m));
    std::vector<double> h_UtU((size_t)m * m);
    CHECK_CUDA(cudaMemcpy(h_UtU.data(), d_temp, (size_t)m * m * sizeof(double), cudaMemcpyDeviceToHost));
    double orth_u = 0;
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
            double v = h_UtU[i + (size_t)j * m] - (i == j ? 1.0 : 0.0);
            orth_u += v * v;
        }
    orth_u = sqrt(orth_u) / sqrt((double)m);
    printf("  ||U^T*U - I||/sqrt(m)   = %.3e\n", orth_u);

    // VT orthogonality
    CHECK_CUBLAS(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, n, n, &one, d_VT, n, d_VT, n, &zero, d_temp, n));
    std::vector<double> h_VtV((size_t)n * n);
    CHECK_CUDA(cudaMemcpy(h_VtV.data(), d_temp, (size_t)n * n * sizeof(double), cudaMemcpyDeviceToHost));
    double orth_vt = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            double v = h_VtV[i + (size_t)j * n] - (i == j ? 1.0 : 0.0);
            orth_vt += v * v;
        }
    orth_vt = sqrt(orth_vt) / sqrt((double)n);
    printf("  ||VT^T*VT - I||/sqrt(n) = %.3e\n", orth_vt);

    // Reconstruction: ||A - U*S*VT|| / ||A||
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(double), cudaMemcpyHostToDevice));

    double *d_T;
    CHECK_CUDA(cudaMalloc(&d_T, sizeA * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_T, d_U, (size_t)m * minmn * sizeof(double), cudaMemcpyDeviceToDevice));
    for (int j = 0; j < minmn; j++)
        CHECK_CUBLAS(cublasDscal(cublas, m, h_S.data() + j, d_T + (size_t)j * m, 1));


    double *d_Arecon;
    CHECK_CUDA(cudaMalloc(&d_Arecon, sizeA * sizeof(double)));
    CHECK_CUBLAS(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, minmn, &one, d_T, m, d_VT, n, &zero, d_Arecon, m));

    std::vector<double> h_Arecon(sizeA);
    CHECK_CUDA(cudaMemcpy(h_Arecon.data(), d_Arecon, sizeA * sizeof(double), cudaMemcpyDeviceToHost));

    double recon_err = 0;
    for (size_t i = 0; i < sizeA; i++) {
        double diff = h_A[i] - h_Arecon[i];
        recon_err += diff * diff;
    }
    recon_err = sqrt(recon_err) / normA;
    printf("  ||A-U*S*VT||/||A||      = %.3e\n", recon_err);


    printf("\n  First 5 singular values:\n");
    int nshow = (minmn < 5) ? minmn : 5;
    for (int i = 0; i < nshow; i++)
        printf("    sigma[%d] = %.12e\n", i, h_S[i]);

    double tol = 1e-10 * sqrt((double)minmn);
    bool pass = (orth_u < tol) && (orth_vt < tol) && (recon_err < tol);

    printf("\n  Tolerance: %.3e\n", tol);
    printf("  Result: %s\n\n", pass ? "PASS" : "*** FAIL ***");


    printf("============================================================\n");
    printf("  %d x %d SVD:  %.2f ms   %s\n", m, n, elapsed_ms, pass ? "PASS" : "FAIL");
    printf("============================================================\n");


    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_S); cudaFree(d_U); cudaFree(d_VT);
    cudaFree(d_work); cudaFree(d_temp); cudaFree(d_T); cudaFree(d_Arecon);
    cublasDestroy(cublas);

    return pass ? 0 : 1;
}
