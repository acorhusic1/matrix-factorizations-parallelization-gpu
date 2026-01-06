// Help IntelliSense understand CUDA keywords
#ifdef __INTELLISENSE__
#define __global__
#define __device__
#define __host__
#define __shared__
void __syncthreads();
struct { int x; } threadIdx, blockIdx, blockDim, gridDim;
#define <<< ,
#define >>> ;
#endif



#include <cuda_runtime.h>
#include <iostream>

//Macro za greske
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << result << " \"" << func << "\" \n";
        std::cerr << "Error string: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Print GPU device properties
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);


    std::cout << "Starting CUDA test..." << std::endl;

    // Launch kernel
    helloFromGPU<<<1, 1024>>>();

    // Check for launch errors
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize()); // Wait for GPU to finish

    return 0;
}