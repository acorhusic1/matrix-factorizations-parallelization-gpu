#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include "base.h"
#include "householder.cuh" 

int main() {
    srand(time(0));
    std::cout << std::fixed << std::setprecision(6);

    // Inicijalizacija biblioteka
    cusolverDnHandle_t solver_handle;
    cublasHandle_t blas_handle;
    checkCudaErrors((cudaError_t)cusolverDnCreate(&solver_handle));
    checkCudaErrors((cudaError_t)cublasCreate(&blas_handle));
/*
    // ====================================================================
    // TEST 1: DEBUG USING 3x3 MATRIX
    // ====================================================================
    {
        const int m = 3; const int n = 3;

        GPUMatrix A_master(m, n);
		// Column-major initialization
        A_master(0, 0) = 12; A_master(1, 0) = 6;   A_master(2, 0) = -4;
        A_master(0, 1) = -51; A_master(1, 1) = 167; A_master(2, 1) = 24;
        A_master(0, 2) = 4;   A_master(1, 2) = -68; A_master(2, 2) = -41;

        A_master.Print("A (Master - Pocetna)");
        A_master.CopyToDevice();

        // --- 1. cuSOLVER REFERENCE ---
        std::cout << "\n--- [1] cuSOLVER Reference ---" << std::endl;
        GPUMatrix A_lib = A_master;
        GPUMatrix Q(m, m);
        GPUMatrix R(m, n);

        float* d_tau, * d_work; int* devInfo;
        int work_size_geqrf = 0, work_size_orgqr = 0;

        cudaMalloc(&d_tau, sizeof(float) * n);
        cudaMalloc(&devInfo, sizeof(int));

        cusolverDnSgeqrf_bufferSize(solver_handle, m, n, A_lib.d_data, m, &work_size_geqrf);
        cusolverDnSorgqr_bufferSize(solver_handle, m, m, n, A_lib.d_data, m, d_tau, &work_size_orgqr);

        int work_size = std::max(work_size_geqrf, work_size_orgqr);
        cudaMalloc(&d_work, sizeof(float) * work_size);

        // QR Factorization
        checkCudaErrors((cudaError_t)cusolverDnSgeqrf(solver_handle, m, n, A_lib.d_data, m, d_tau, d_work, work_size, devInfo));

        A_lib.CopyToHost();

        // Extracting R
        Householder::extract_R(A_lib, R);

        // Extracting Q
        checkCudaErrors((cudaError_t)cusolverDnSorgqr(solver_handle, m, m, n, A_lib.d_data, m, d_tau, d_work, work_size, devInfo));
        A_lib.CopyToHost();
        Q = A_lib;
        Q.CopyToDevice();

        // Validation: res = Q * R
        GPUMatrix res(m, n);
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors((cudaError_t)cublasSgemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, m,
            &alpha,
            Q.d_data, m,
            R.d_data, m,
            &beta,
            res.d_data, m));
        res.CopyToHost();

        bool pass = GPUMatrix::isEqual(res, A_master);
        std::cout << "cuSOLVER Status: " << (pass ? "PASS" : "FAIL") << std::endl;
        if (!pass) res.Print("Rezultat Q*R (cuSOLVER Error)");

        cudaFree(d_tau); cudaFree(d_work); cudaFree(devInfo);

        // --- 2. HOUSEHOLDER ---
        std::cout << "\n--- [2] Custom Householder Implementation ---" << std::endl;
        GPUMatrix A_cust = A_master;
        std::vector<float> h_tau;

        auto start_cust = std::chrono::high_resolution_clock::now();
        Householder::qr_decomposition(A_cust, h_tau);
        auto end_cust = std::chrono::high_resolution_clock::now();
        double cust_time = std::chrono::duration<double>(end_cust - start_cust).count();

        std::cout << "Factorization time: " << cust_time * 1000 << " ms" << std::endl;

        GPUMatrix Q_cust(m, m);
        GPUMatrix R_cust(m, n);

        Householder::extract_Q(A_cust, h_tau, Q_cust);
        Householder::extract_R(A_cust, R_cust);

        Q_cust.CopyToHost();
        R_cust.CopyToHost();

        GPUMatrix res_cust(m, n);
        checkCudaErrors((cudaError_t)cublasSgemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, m,
            &alpha,
            Q_cust.d_data, m,
            R_cust.d_data, m,
            &beta,
            res_cust.d_data, m));
        res_cust.CopyToHost();

        Q_cust.Print("Q (Custom Householder)");
        R_cust.Print("R (Custom Householder)");

        bool pass_cust = GPUMatrix::isEqual(res_cust, A_master, 1e-3);
        std::cout << std::endl << "Custom Householder Status: " << (pass_cust ? "PASS" : "FAIL") << std::endl;
        if (!pass_cust) res_cust.Print("Rezultat Q*R (Custom Householder Error)");
    }
*/

    // ====================================================================
    // TEST 2: PERFORMANCE COMPARISON (SAME MATRIX PER ITERATION)
    // ====================================================================
    {
        const int M = 2048;
        const int N = 2048;
        const int TEST_RUNS = 10;

        std::cout << "\n------------------------------------------------------------\n";
        std::cout << "  GPU QR PERFORMANCE COMPARISON \n";
        std::cout << "  Matrix Size: " << M << "x" << N << " | Runs: " << TEST_RUNS << "\n";
        std::cout << "------------------------------------------------------------\n\n";

        double total_time_cusolver = 0;
        double total_time_custom = 0;

        for (int run = 0; run < TEST_RUNS; run++) {
            std::cout << std::endl << "--- ITERATION " << run + 1 << " ---" << std::endl;

            GPUMatrix A_master = GPUMatrix::GenerateRandom(M, N);
            A_master.CopyToDevice();

            // -----------------------------------------------------
            // A) cuSOLVER
            // -----------------------------------------------------
            {
                GPUMatrix A_perf = A_master;

                float* d_tau, * d_work; int* devInfo;
                int size_geqrf, size_orgqr;

                cudaMalloc(&d_tau, sizeof(float) * N);
                cudaMalloc(&devInfo, sizeof(int));

                cusolverDnSgeqrf_bufferSize(solver_handle, M, N, A_perf.d_data, M, &size_geqrf);
                cusolverDnSorgqr_bufferSize(solver_handle, M, M, N, A_perf.d_data, M, d_tau, &size_orgqr);
                int work_size = std::max(size_geqrf, size_orgqr);

                cudaMalloc(&d_work, sizeof(float) * work_size);

                auto start = std::chrono::high_resolution_clock::now();
                cusolverDnSgeqrf(solver_handle, M, N, A_perf.d_data, M, d_tau, d_work, work_size, devInfo);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                double time = std::chrono::duration<double>(end - start).count();
                total_time_cusolver += time;
                std::cout << "   cuSOLVER: " << time * 1000 << " ms" << std::endl;

                cudaFree(d_tau); cudaFree(d_work); cudaFree(devInfo);
            }

            // -----------------------------------------------------
            // B) Custom Householder
            // -----------------------------------------------------
            {
                GPUMatrix A_perf = A_master;
                std::vector<float> h_tau;

                auto start = std::chrono::high_resolution_clock::now();
                Householder::qr_decomposition(A_perf, h_tau);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                double time = std::chrono::duration<double>(end - start).count();
                total_time_custom += time;
                std::cout << "   CUSTOM:   " << time * 1000 << " ms" << std::endl;

                float alpha = 1.0f, beta = 0.0f;
                GPUMatrix Q_perf(M, M);
                GPUMatrix R_perf(M, N);

                Householder::extract_Q(A_perf, h_tau, Q_perf);
                Householder::extract_R(A_perf, R_perf);

                Q_perf.CopyToHost();
                R_perf.CopyToHost();

                GPUMatrix res_perf(M, N);
                checkCudaErrors((cudaError_t)cublasSgemm(blas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, M,
                    &alpha,
                    Q_perf.d_data, M,
                    R_perf.d_data, M,
                    &beta,
                    res_perf.d_data, M));
                res_perf.CopyToHost();

                bool pass_perf = GPUMatrix::isEqual(res_perf, A_master, 1e-2);
                std::cout << "Custom Householder Status: " << (pass_perf ? "PASS" : "FAIL") << std::endl;

            }
        }

        // --- FINAL REPORT ---
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << " FINAL RESULTS (Average over " << TEST_RUNS << " runs)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "cuSOLVER Avg: " << (total_time_cusolver / TEST_RUNS) * 1000 << " ms" << std::endl;
        std::cout << "CUSTOM   Avg: " << (total_time_custom / TEST_RUNS) * 1000 << " ms" << std::endl;

        double relativeRuntime = (total_time_cusolver / TEST_RUNS) / (total_time_custom / TEST_RUNS);
        std::cout << "Relative runtime to cuSOLVER: " << relativeRuntime << "x" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }

    cusolverDnDestroy(solver_handle);
    cublasDestroy(blas_handle);

    return 0;
}