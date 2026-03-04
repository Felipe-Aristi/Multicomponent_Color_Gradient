#ifndef CUDA_CONFIG_CUH
#define CUDA_CONFIG_CUH

#include <cuda_runtime.h>
#include "../constants.cuh"

__host__ __device__ __forceinline__ int ceil_div(size_t a, size_t b)
{
    return (a + b - 1) / b;
}

struct CudaConfig
{
    dim3 block;
    dim3 grid;
};

__host__ __forceinline__ CudaConfig make_cuda_config()
{
    CudaConfig cfg{};
    cfg.block = dim3(8, 8, 4);
    cfg.grid = dim3(
        ceil_div((size_t)NX, (size_t)cfg.block.x),
        ceil_div((size_t)NY, (size_t)cfg.block.y),
        ceil_div((size_t)NZ, (size_t)cfg.block.z));
    return cfg;
}

#endif