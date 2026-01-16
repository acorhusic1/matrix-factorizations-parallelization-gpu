#include "householder.cuh"
#include <cmath>
#include <cstdio>

// Podesavanja
#define PIVOT_THREADS 1024
#define MATRIX_THREADS_X 128
#define MATRIX_THREADS_Y 8

namespace Householder {

    // =================================================================================
    // KERNEL 1: Update Pivot Element
    // =================================================================================
    __global__ void updatePivotKernel(float* matrix, int pivotRow, int numCols, int ld) {
        // checkCudaErrors(cudaGetLastError());

        // Shared memorija za redukciju
        __shared__ float tempArray[PIVOT_THREADS];
        __shared__ float sumOfSquares;

        int tid = threadIdx.x;
        int dim = blockDim.x;

        tempArray[tid] = 0.0f;
        if (tid == 0) sumOfSquares = 0.0f;

        __syncthreads();

        // 1. Ucitavanje i kvadriranje elemenata u chunk-ovima

        // Racunamo startnu kolonu (preskacemo kolone lijevo od pivota jer je to R matrica)
        int colStart = pivotRow;

        float localSum = 0.0f;

        for (int j = colStart + tid; j < numCols; j += dim) {
            float val = matrix[j * ld + pivotRow]; // Column-Major pristup: A[row][col] -> data[col * ld + row]
            localSum += val * val;
        }

        tempArray[tid] = localSum;
        __syncthreads();

        // 2. Redukcija unutar shared memorije
        // Standardna tree-based redukcija
        for (int s = dim / 2; s > 0; s >>= 1) {
            if (tid < s) {
                tempArray[tid] += tempArray[tid + s];
            }
            __syncthreads();
        }

        // 3. Thread 0 racuna korijen i azurira pivot
        if (tid == 0) {
            sumOfSquares = tempArray[0];
            float pivotValue = sqrtf(sumOfSquares);

            // Householder standard: odabir znaka da se izbjegne cancellation
            float currentPivot = matrix[pivotRow * ld + pivotRow];
            if (currentPivot > 0) {
                pivotValue = -pivotValue;
            }

            // Azuriranje dijagonale (ovo zapravo mijenja A u R postepeno)
            matrix[pivotRow * ld + pivotRow] = pivotValue;

        }
    }

    // =================================================================================
    // KERNEL 2: Update Matrix
    // =================================================================================
    __global__ void updateMatrixKernel(float* matrix, int pivotRow, int numRows, int numCols, int ld, int startRow) {
        // checkCudaErrors(cudaGetLastError());

        // Y dimenzija bloka obradjuje redove, X dimenzija obradjuje kolone (chunks)
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // Red koji ovaj thread obradjuje: BlockIdxY * BlockDimY + ThreadY + 1 (relativno na pivot)
        // Proslijedi startRow za Stream logiku (startRow moze biti pivotRow + 1 ili pivotRow + 2)
        int globalRow = startRow + blockIdx.y * blockDim.y + ty;

        if (globalRow >= numRows) return;

        // Shared memorija
        __shared__ float sharedPivotValues[MATRIX_THREADS_X];
        __shared__ float sharedRowSums[MATRIX_THREADS_Y];

        float threadSum = 0.0f;
        if (tx == 0) sharedRowSums[ty] = 0.0f;

        // ============================================================
        // FAZA 1: Racunanje Dot Product-a (v^T * A)
        // Iteriranje kroz kolone u "chunks"
        // ============================================================

        // Pivot element (dijagonala) koji je izracunat u Kernelu 1
        float pivotElem = matrix[pivotRow * ld + pivotRow];

        int colStart = pivotRow; // Householder radi na donjem desnom dijelu

        for (int j = colStart + tx; j < numCols; j += blockDim.x) {
            // Ucitaj pivot red u shared
            // Napomena: u Kernel 1 smo promijenili A[pivot][pivot].
            float p_val = matrix[j * ld + pivotRow];

            if (j == pivotRow) {
                // Za potrebe dot producta sa v, v[pivot] = 1.
                sharedPivotValues[tx] = 1.0f;
            }
            else {
                sharedPivotValues[tx] = p_val; // Ostali elementi vektora v
            }
            __syncthreads();

            // Racunaj parcijalni dot product
            float mat_val = matrix[j * ld + globalRow];
            threadSum += mat_val * sharedPivotValues[tx];

            __syncthreads();
        }

        // Koristimo atomicAdd za jednostavnost i tacnost prema sharedRowSums[ty]
        atomicAdd(&sharedRowSums[ty], threadSum);
        __syncthreads();

        // ============================================================
        // FAZA 2: Azuriranje Matrice (Rank-1 update)
        // A = A - v * w^T (gdje w = beta * A^T * v)
        // ============================================================

        float tau = 2.0f / (1.0f + 0.0f); // Placeholder, u pravoj implementaciji ovo mora doci iz Kernela 1
        float scalar = tau * sharedRowSums[ty]; // Ovo je nas "beta * dot"

        for (int j = colStart + tx; j < numCols; j += blockDim.x) {
            // Ponovo ucitaj v (pivot red)
            float p_val = (j == pivotRow) ? 1.0f : matrix[j * ld + pivotRow];

            // Azuriraj element
            // A[row][col] -= scalar * v[col]
            matrix[j * ld + globalRow] -= scalar * p_val;
        }
    }

    // Helper za racunanje grida
    dim3 computeGrid(int rows, int cols) {
        dim3 block(MATRIX_THREADS_X, MATRIX_THREADS_Y);
        dim3 grid(1, (rows + block.y - 1) / block.y);
        // X grid je 1 jer petljamo kroz kolone unutar kernela (chunks)
        return grid;
    }

    // =================================================================================
    // HOST FUNCTION: Stream Orchestration
    // =================================================================================
    void qr_decomposition(GPUMatrix& A) {
        int m = A.m;
        int n = A.n;
        int ld = A.m; // Leading dimension

        // 1. Inicijalizacija Streamova i Event-a
        cudaStream_t stream1, stream2;
        cudaEvent_t event1, event2;

        checkCudaErrors(cudaStreamCreate(&stream1));
        checkCudaErrors(cudaStreamCreate(&stream2));
        checkCudaErrors(cudaEventCreate(&event1));
        checkCudaErrors(cudaEventCreate(&event2));

        // 2. Inicijalno pokretanje za prvi pivot
        // Kernel 1 na Stream 1
        updatePivotKernel << <1, PIVOT_THREADS, 0, stream1 >> > (A.d_data, 0, n, ld);
        checkCudaErrors(cudaDeviceSynchronize()); // Cekamo da se prvi pivot pripremi

        // 3. Glavna petlja
        for (int lpivot = 0; lpivot < min(m - 1, n); ++lpivot) {

            // Konfiguracija kernela za Update Matrix
            // Trebamo azurirati redove od lpivot + 1 do m
            int rowsRemaining = m - (lpivot + 1);
            if (rowsRemaining <= 0) break;

            // - Stream 1 procesira sljedeci red (lpivot + 1)
            // Ovo je "kriticni" red jer on postaje pivot u iducoj iteraciji
            dim3 gridNext = computeGrid(1, n); // Samo 1 red
            updateMatrixKernel << <gridNext, dim3(MATRIX_THREADS_X, MATRIX_THREADS_Y), 0, stream1 >> > (
                A.d_data, lpivot, m, n, ld, lpivot + 1
                );

            // - Stream 2 procesira ostale redove (lpivot + 2 do m)
            if (rowsRemaining > 1) {
                // Sacekaj da Event 1 (iz prethodne iteracije, pivot spreman) bude gotov
                // U prvoj iteraciji nema eventa, ali kasnije ima
                if (lpivot > 0) {
                    checkCudaErrors(cudaStreamWaitEvent(stream2, event1, 0)); //
                }

                dim3 gridRest = computeGrid(rowsRemaining - 1, n);
                updateMatrixKernel << <gridRest, dim3(MATRIX_THREADS_X, MATRIX_THREADS_Y), 0, stream2 >> > (
                    A.d_data, lpivot, m, n, ld, lpivot + 2
                    );

                // Zabiljezi da je Stream 2 zavrsio svoj dio posla za ovu iteraciju
                checkCudaErrors(cudaEventRecord(event2, stream2));
            }

            // Kernel 1 za sljedeci pivot
            if (lpivot + 1 < min(m, n)) {
                updatePivotKernel << <1, PIVOT_THREADS, 0, stream1 >> > (A.d_data, lpivot + 1, n, ld);
            }

            // Zabiljezi Event 1 da je novi pivot spreman
            checkCudaErrors(cudaEventRecord(event1, stream1));
        }

        checkCudaErrors(cudaDeviceSynchronize());

        // Cleanup
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaEventDestroy(event1);
        cudaEventDestroy(event2);
    }
}