#pragma once
#ifndef BASE_H
#define BASE_H

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>

// Makro za CUDA greske
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

// Static inline omogucava da funkcija postoji u svakom fajlu koji je ukljuci, a da se pritom linker nece buniti
static inline void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << result << " \"" << func << "\" \n";
        std::cerr << "Error string: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// BASE MATRIX CLASS (HOST + DEVICE BRIDGE)
// ============================================================================
class GPUMatrix {
public:
    int m, n;
    float* h_data; // Podaci na CPU
    float* d_data; // Podaci na GPU
    bool on_device;

    GPUMatrix(int rows, int cols) : m(rows), n(cols), on_device(false) {
        h_data = new float[m * n];
        std::memset(h_data, 0, m * n * sizeof(float));
        checkCudaErrors(cudaMalloc(&d_data, m * n * sizeof(float)));
    }

    // Copy constructor (duboka kopija)
    GPUMatrix(const GPUMatrix& other) : m(other.m), n(other.n), on_device(false) {
        h_data = new float[m * n];
        std::memcpy(h_data, other.h_data, m * n * sizeof(float));
        checkCudaErrors(cudaMalloc(&d_data, m * n * sizeof(float)));
        if (other.on_device) this->CopyToDevice();
    }

    ~GPUMatrix() {
        delete[] h_data;
        cudaFree(d_data);
    }

    // Dodaj ovaj operator dodjele (Assignment Operator)
    GPUMatrix& operator=(const GPUMatrix& other) {
        if (this != &other) {
            // Prvo oslobodi staru memoriju
            if (h_data) delete[] h_data;
            if (d_data) cudaFree(d_data);

            m = other.m;
            n = other.n;

            // Alociraj novu memoriju i kopiraj podatke sa Host-a
            h_data = new float[m * n];
            std::memcpy(h_data, other.h_data, m * n * sizeof(float));
            checkCudaErrors(cudaMalloc(&d_data, m * n * sizeof(float)));

            // Ako je original bio na device-u, kopiraj i tamo
            if (other.on_device) {
                this->CopyToDevice();
            }
            else {
                on_device = false;
            }
        }
        return *this;
    }

    // Column-Major indeksiranje
    float& operator()(int i, int j) { return h_data[j * m + i]; }
    const float& operator()(int i, int j) const { return h_data[j * m + i]; }

    void CopyToDevice() {
        checkCudaErrors(cudaMemcpy(d_data, h_data, m * n * sizeof(float), cudaMemcpyHostToDevice));
        on_device = true;
    }

    void CopyToHost() {
        checkCudaErrors(cudaMemcpy(h_data, d_data, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void Print(const std::string& name = "Matrix") const {
        std::cout << "\nMatrix " << name << " (" << m << "x" << n << "):" << std::endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                std::cout << std::setw(11) << std::fixed << std::setprecision(5) << (*this)(i, j) << " ";
            std::cout << std::endl;
        }
    }

    void SetZero() {
        std::memset(h_data, 0, m * n * sizeof(float));
        if (on_device) {
            checkCudaErrors(cudaMemset(d_data, 0, m * n * sizeof(float)));
        }
    }

    static GPUMatrix GenerateRandom(int rows, int cols) {
        GPUMatrix A(rows, cols);
        for (int i = 0; i < rows * cols; i++)
            A.h_data[i] = (float)std::rand() / RAND_MAX * 2.0f - 1.0f;
        return A;
    }

    static bool isEqual(const GPUMatrix& A, const GPUMatrix& B, float eps = 1e-3) {
        if (A.m != B.m || A.n != B.n) {
            std::cerr << "Matrix dimensions do not match for equality check!" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < A.m * A.n; i++) {
            if (std::fabs(A.h_data[i] - B.h_data[i]) > eps)
				return false;
        }

        return true;
    }

};

#endif