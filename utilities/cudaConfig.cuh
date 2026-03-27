#ifndef CUDACONFIG_CUH
#define CUDACONFIG_CUH

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

__host__ __forceinline__ CudaConfig make_cudaConfig()
{
    CudaConfig cfg{};
    cfg.block = dim3(32, 32, 1);
    cfg.grid = dim3(
        ceil_div((size_t)NX, (size_t)cfg.block.x),
        ceil_div((size_t)NY, (size_t)cfg.block.y),
        ceil_div((size_t)NZ, (size_t)cfg.block.z));
    return cfg;
}

// "2D" Kernels

// Use (block.x, block.y) as (x,z), ignore y and z dims
__host__ __device__ __forceinline__ dim3 block2D_xz(const CudaConfig &cfg)
{
    return dim3(cfg.block.x, cfg.block.y, 1);
}

// Grid for XZ plane: grid.x spans NX, grid.y spans NZ
__host__ __device__ __forceinline__ dim3 grid2D_xz(const CudaConfig &cfg)
{
    const dim3 b = block2D_xz(cfg);
    return dim3(
        ceil_div((size_t)NX, (size_t)b.x),
        ceil_div((size_t)NZ, (size_t)b.y),
        1);
}

#endif