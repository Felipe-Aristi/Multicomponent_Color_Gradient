#ifndef NAN_CHECK_CUH
#define NAN_CHECK_CUH

#include <cuda_runtime.h>
#include <cstdio>

#include "constants.cuh"
#include "utilities/cudaUtilities.cuh" // your CUDA_CHECK
#include "utilities/indexing.cuh"      // idx/fidx if needed

// stores (first_bad_index + 1) into *flag, or stays 0 if all finite
__global__ void check_nan_kernel(const real_t *a, int n, int *flag)
{
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= n)
        return;

    real_t v = a[t];

    // CUDA device-side isfinite/isnan are available via cuda_runtime.h
    if (!isfinite(v))
    {
        // only first writer wins (flag was 0)
        atomicCAS(flag, 0, t + 1);
    }
}

// convenience: launch + copy flag back
inline int check_nan_D2H(const real_t *d_a, int n, const char *name)
{
    int *d_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    check_nan_kernel<<<blocks, threads>>>(d_a, n, d_flag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_flag = 0;
    CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_flag));

    if (h_flag != 0)
    {
        int bad = h_flag - 1;
        std::printf("[NaN/Inf] %s first bad linear index = %d\n", name, bad);
    }
    else
    {
        std::printf("[OK] %s all finite\n", name);
    }
    return h_flag; // 0 means OK, else bad_index+1
}

// helper to decode an index in f-array into (x,y,z,i)
inline void decode_f_index(int linear_f, int &x, int &y, int &z, int &i)
{
    i = linear_f % (int)Q;
    int cell = linear_f / (int)Q;

    x = cell % (int)NX;
    int tmp = cell / (int)NX;
    y = tmp % (int)NY;
    z = tmp / (int)NY;
}

#endif