#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include "base.h"
#include "householder.cuh" // Householder QR implementacija

// TODO: TSQR implementacija

// ====================================================================
// MAIN TEST LOGIC
// ====================================================================

int main() {
    srand(time(0));
    std::cout << std::fixed << std::setprecision(6);

    // Inicijalizacija biblioteka
    cusolverDnHandle_t solver_handle;
    cublasHandle_t blas_handle;
    checkCudaErrors((cudaError_t)cusolverDnCreate(&solver_handle));
    checkCudaErrors((cudaError_t)cublasCreate(&blas_handle));

    // ====================================================================
    // TEST 1: DEBUG NA POZNATOJ 3x3 MATRICI (FER POREDJENJE)
    // ====================================================================
    {
        std::cout << "\n>>> POKRETANJE 3x3 DEBUG TESTA <<<\n";
        const int m = 3; const int n = 3;

        GPUMatrix A_master(m, n);
        // Column-major inicijalizacija
        A_master(0, 0) = 12; A_master(1, 0) = 6;   A_master(2, 0) = -4;
        A_master(0, 1) = -51; A_master(1, 1) = 167; A_master(2, 1) = 24;
        A_master(0, 2) = 4;   A_master(1, 2) = -68; A_master(2, 2) = -41;

        A_master.Print("A (Master - Pocetna)");
        A_master.CopyToDevice();

        // --- 1. cuSOLVER REFERENCE ---
        std::cout << "\n--- [1] cuSOLVER Reference ---" << std::endl;
        GPUMatrix A_lib = A_master;
        GPUMatrix Q(m, m); // Q je m x m
        GPUMatrix R(m, n); // R je m x n

        float* d_tau, * d_work; int* devInfo, work_size;
        cudaMalloc(&d_tau, sizeof(float) * n);
        cudaMalloc(&devInfo, sizeof(int));

        cusolverDnSgeqrf_bufferSize(solver_handle, m, n, A_lib.d_data, m, &work_size);
        cudaMalloc(&d_work, sizeof(float) * work_size);

        // QR Faktorizacija
        checkCudaErrors((cudaError_t)cusolverDnSgeqrf(solver_handle, m, n, A_lib.d_data, m, d_tau, d_work, work_size, devInfo));

        // EKSTRAKCIJA R: R se nalazi u gornjem trouglu A_lib (ukljucujuci dijagonalu)
        A_lib.CopyToHost();
        R.SetZero(); // Inicijalizuj sve na 0
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                R(i, j) = A_lib(i, j);
            }
        }
        R.CopyToDevice();

        // EKSTRAKCIJA Q: Koristimo sorgqr da pretvorimo Householder vektore u Q matricu
        checkCudaErrors((cudaError_t)cusolverDnSorgqr(solver_handle, m, m, n, A_lib.d_data, m, d_tau, d_work, work_size, devInfo));
        A_lib.CopyToHost();
        Q = A_lib; // Q je sada u A_lib
        Q.CopyToDevice();

        // VALIDACIJA: res = Q * R
        GPUMatrix res(m, n);
        float alpha = 1.0f, beta = 0.0f;
        // ld za Q je m, ld za R je m, ld za res je m
        checkCudaErrors((cudaError_t)cublasSgemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, m,
            &alpha,
            Q.d_data, m,
            R.d_data, m,
            &beta,
            res.d_data, m));
        res.CopyToHost();

        // Koristimo malu toleranciju za poredjenje floating point brojeva
        bool pass = GPUMatrix::isEqual(res, A_master);
        std::cout << "cuSOLVER Status: " << (pass ? "PASS" : "FAIL") << std::endl;
        if (!pass) res.Print("Rezultat Q*R (cuSOLVER Error)");

        cudaFree(d_tau); cudaFree(d_work); cudaFree(devInfo);

        // --- 2. HOUSEHOLDER ---
        std::cout << "\n--- [2] Custom Householder Implementation ---" << std::endl;
        GPUMatrix A_cust = A_master;

        // Dodajemo try-catch ili provjeru da vidimo gdje puca (ako puca)
        Householder::qr_decomposition(A_cust);

        A_cust.CopyToHost();
        A_cust.Print("A (Faktorizirana - Custom)");
    }

    // ====================================================================
    // TEST 2: PERFORMANCE COMPARISON (SAME MATRIX PER ITERATION)
    // ====================================================================
    {
        const int M = 5000;
        const int N = 5000;
        const int TEST_RUNS = 5;

        std::cout << "\n------------------------------------------------------------\n";
        std::cout << "  GPU QR PERFORMANCE COMPARISON (Fair Play Mode) \n";
        std::cout << "  Matrix Size: " << M << "x" << N << " | Runs: " << TEST_RUNS << "\n";
        std::cout << "------------------------------------------------------------\n\n";

        double total_time_cusolver = 0;
        double total_time_custom = 0;

        for (int run = 0; run < TEST_RUNS; run++) {
            std::cout << "--- ITERATION " << run + 1 << " ---" << std::endl;

            // 1. Generisanje MASTER matrice za ovu iteraciju
            GPUMatrix A_master = GPUMatrix::GenerateRandom(M, N);
            A_master.CopyToDevice();

            // -----------------------------------------------------
            // A) Mjerenje cuSOLVER (na kopiji 1)
            // -----------------------------------------------------
            {
                GPUMatrix A_perf = A_master; // Duboka kopija

                float* d_tau, * d_work; int* devInfo, work_size;
                cudaMalloc(&d_tau, sizeof(float) * N);
                cudaMalloc(&devInfo, sizeof(int));

                cusolverDnSgeqrf_bufferSize(solver_handle, M, N, A_perf.d_data, M, &work_size);
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
            // B) Mjerenje moje Householder (na kopiji 2)
            // -----------------------------------------------------
            {
                GPUMatrix A_perf = A_master; // Duboka kopija

                auto start = std::chrono::high_resolution_clock::now();
                Householder::qr_decomposition(A_perf);
                auto end = std::chrono::high_resolution_clock::now();

                double time = std::chrono::duration<double>(end - start).count();
                total_time_custom += time;
                std::cout << "   CUSTOM:   " << time * 1000 << " ms" << std::endl;
            }
        }

        // --- FINAL REPORT ---
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << " FINAL RESULTS (Average over " << TEST_RUNS << " runs)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "cuSOLVER Avg: " << (total_time_cusolver / TEST_RUNS) * 1000 << " ms" << std::endl;
        std::cout << "CUSTOM   Avg: " << (total_time_custom / TEST_RUNS) * 1000 << " ms" << std::endl;

        double speedup = (total_time_cusolver / TEST_RUNS) / (total_time_custom / TEST_RUNS);
        std::cout << "Speedup vs cuSOLVER: " << speedup << "x" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }

    cusolverDnDestroy(solver_handle);
    cublasDestroy(blas_handle);

    return 0;
}