// bench_svd.cu — Benchmark my SVD vs cuSOLVER dgesvd
#include "gebrd.cuh"
#include "svd.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

// ---------------------------------------------------------------------------
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

#define CHECK_CUSOLVER(call) do {                                           \
    cusolverStatus_t s = (call);                                            \
    if (s != CUSOLVER_STATUS_SUCCESS) {                                     \
        fprintf(stderr,"cuSOLVER error %s:%d: %d\n",__FILE__,__LINE__,(int)s);\
        exit(1); }                                                          \
} while(0)

// ---------------------------------------------------------------------------
static void fill_random(double* h, int m, int n, unsigned seed = 42)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            h[i + (size_t)j * m] = dist(rng);
}

// ---------------------------------------------------------------------------
// Correctness + performance test for one (m, n, nb)
// ---------------------------------------------------------------------------
static bool test_and_bench(int m, int n, int nb, int nruns)
{
    int minmn = std::min(m, n);
    size_t sizeA = (size_t)m * n;

    printf("=== Full SVD  m=%d  n=%d  nb=%d  runs=%d ===\n", m, n, nb, nruns);

    std::vector<double> h_A(sizeA);
    fill_random(h_A.data(), m, n, 42 + m + n);

    // Compute ||A||_F
    double normA = 0;
    for (auto& v : h_A) normA += v * v;
    normA = sqrt(normA);

    // Device memory for my SVD
    double *d_A;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(double)));

    double *d_S, *d_U, *d_VT;
    CHECK_CUDA(cudaMalloc(&d_S,  minmn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_U,  (size_t)m * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_VT, (size_t)n * n * sizeof(double)));

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));

    size_t ws = svd_workspace_size(m, n, nb);
    double* d_work;
    CHECK_CUDA(cudaMalloc(&d_work, ws * sizeof(double)));

    // Device memory for cuSOLVER SVD
    cusolverDnHandle_t cusolver;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver));

    double *d_A_ref, *d_S_ref, *d_U_ref, *d_VT_ref;
    CHECK_CUDA(cudaMalloc(&d_A_ref,  sizeA * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_S_ref,  minmn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_U_ref,  (size_t)m * m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_VT_ref, (size_t)n * n * sizeof(double)));

    int lwork_ref = 0;
    CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(cusolver, m, n, &lwork_ref));
    double *d_work_ref, *d_rwork;
    int *d_info;
    CHECK_CUDA(cudaMalloc(&d_work_ref, lwork_ref * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_rwork, 5 * minmn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

    // My SVD — warmup + timed runs
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(double), cudaMemcpyHostToDevice));
    svd_full(cublas, m, n, m, d_A, d_S, d_U, m, d_VT, n, d_work, nb);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    double best_mine = 1e30;
    for (int r = 0; r < nruns; r++) {
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(t0));
        svd_full(cublas, m, n, m, d_A, d_S, d_U, m, d_VT, n, d_work, nb);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        best_mine = std::min(best_mine, (double)ms);
    }

    // Save our results to host for verification
    std::vector<double> h_S(minmn);
    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, minmn * sizeof(double), cudaMemcpyDeviceToHost));

    // cuSOLVER SVD — warmup + timed runs
    CHECK_CUDA(cudaMemcpy(d_A_ref, h_A.data(), sizeA * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUSOLVER(cusolverDnDgesvd(cusolver, 'A', 'A',
        m, n, d_A_ref, m, d_S_ref,
        d_U_ref, m, d_VT_ref, n,
        d_work_ref, lwork_ref, d_rwork, d_info));
    CHECK_CUDA(cudaDeviceSynchronize());

    double best_ref = 1e30;
    for (int r = 0; r < nruns; r++) {
        CHECK_CUDA(cudaMemcpy(d_A_ref, h_A.data(), sizeA * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(t0));
        CHECK_CUSOLVER(cusolverDnDgesvd(cusolver, 'A', 'A',
            m, n, d_A_ref, m, d_S_ref,
            d_U_ref, m, d_VT_ref, n,
            d_work_ref, lwork_ref, d_rwork, d_info));
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        best_ref = std::min(best_ref, (double)ms);
    }

    // cuSOLVER singular values
    std::vector<double> h_S_ref(minmn);
    CHECK_CUDA(cudaMemcpy(h_S_ref.data(), d_S_ref, minmn * sizeof(double), cudaMemcpyDeviceToHost));

    // Correctness checks

    // Re-run to get fresh U, VT on device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(double), cudaMemcpyHostToDevice));
    svd_full(cublas, m, n, m, d_A, d_S, d_U, m, d_VT, n, d_work, nb);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, minmn * sizeof(double), cudaMemcpyDeviceToHost));

    // U orthogonality
    double *d_temp;
    int maxmn = std::max(m, n);
    CHECK_CUDA(cudaMalloc(&d_temp, (size_t)maxmn * maxmn * sizeof(double)));

    double one = 1.0, zero = 0.0;
    // U^T * U -> d_temp (m×m)
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

    // Reconstruction ||A - U*S*VT||_F / ||A||_F
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

    // Singular value comparison
    std::vector<double> h_S_sorted = h_S;
    std::sort(h_S_sorted.begin(), h_S_sorted.end(), std::greater<double>());

    double max_sv_err = 0;
    for (int i = 0; i < minmn; i++)
        max_sv_err = std::max(max_sv_err, fabs(h_S_sorted[i] - h_S_ref[i]));
    double rel_sv_err = max_sv_err / (h_S_ref[0] + 1e-30);

    // Report
    printf("  --- Correctness ---\n");
    printf("  ||U^T*U - I||/sqrt(m)     = %.3e\n", orth_u);
    printf("  ||VT^T*VT - I||/sqrt(n)   = %.3e\n", orth_vt);
    printf("  ||A-U*S*VT||/||A||        = %.3e\n", recon_err);
    printf("  max |sigma err|           = %.3e  (rel %.3e)\n", max_sv_err, rel_sv_err);

    int nshow = std::min(minmn, 5);
    for (int i = 0; i < nshow; i++)
        printf("    sigma[%d]:  mine %.12e  cuSOLVER %.12e  diff %.2e\n",
               i, h_S_sorted[i], h_S_ref[i], fabs(h_S_sorted[i] - h_S_ref[i]));

    double tol = 1e-10 * sqrt((double)minmn);
    bool pass = (orth_u < tol) && (orth_vt < tol) && (recon_err < tol) && (rel_sv_err < tol);

    printf("  Correctness: %s\n", pass ? "PASS" : "*** FAIL ***");

    // Performance
    double ratio = best_ref / best_mine;

    printf("  --- Performance (best of %d) ---\n", nruns);
    printf("  My SVD:            %10.2f ms\n", best_mine);
    printf("  cuSOLVER dgesvd:   %10.2f ms\n", best_ref);

    if (ratio > 1.0)
        printf("  Speedup: %.2fx  (mine %.1f%% FASTER)\n", ratio, (ratio - 1.0) * 100);
    else
        printf("  Ratio:   %.2fx  (mine at %.1f%% of cuSOLVER speed)\n", ratio, ratio * 100);

    printf("\n");

    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_S); cudaFree(d_U); cudaFree(d_VT); cudaFree(d_work);
    cudaFree(d_A_ref); cudaFree(d_S_ref); cudaFree(d_U_ref); cudaFree(d_VT_ref);
    cudaFree(d_work_ref); cudaFree(d_rwork); cudaFree(d_info);
    cudaFree(d_temp); cudaFree(d_T); cudaFree(d_Arecon);
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);

    return pass;
}

// ---------------------------------------------------------------------------
int main()
{
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1073741824.0);

    printf("============================================================\n");
    printf("  SVD BENCHMARK:  My SVD  vs  cuSOLVER dgesvd\n");
    printf("============================================================\n");
    printf("  gebrd -> orgbr -> bidiag D&C -> DGEMM\n");
    printf("  cuSOLVER:  dgesvd('A','A')\n");
    printf("============================================================\n\n");

    bool all_pass = true;

    // --- Correctness sweep (small sizes, 3 runs) ---
    printf("--- Small sizes (3 runs) ---\n\n");
    all_pass &= test_and_bench(  32,   32,  16, 3);
    all_pass &= test_and_bench(  64,   64,  32, 3);
    all_pass &= test_and_bench( 128,  128,  32, 3);
    all_pass &= test_and_bench( 256,  256,  32, 3);

    if (!all_pass) {
        printf("*** CORRECTNESS FAILURES — stopping ***\n");
        return 1;
    }

    // --- Performance sweep (larger sizes) ---
    printf("============================================================\n");
    printf("--- Larger sizes (3 runs) ---\n");
    printf("============================================================\n\n");
    all_pass &= test_and_bench( 384,  384,  32, 3);
    all_pass &= test_and_bench( 512,  512,  32, 3);
    all_pass &= test_and_bench(1024, 1024,  32, 3);
    all_pass &= test_and_bench(2048, 2048,  32, 3);

    // Summary
    printf("============================================================\n");
    printf("  All tests: %s\n", all_pass ? "PASSED" : "SOME FAILED");
    printf("============================================================\n");

    return all_pass ? 0 : 1;
}

