#ifndef CUDA_UTILITIES_CUH
#define CUDA_UTILITIES_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Cuda error checking
#define CUDA_CHECK(call)                                                                    \
        do                                                                                  \
        {                                                                                   \
                cudaError_t err__ = (call);                                                 \
                if (err__ != cudaSuccess)                                                   \
                {                                                                           \
                        fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",                    \
                                cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__); \
                        std::exit(1);                                                       \
                }                                                                           \
        } while (0)

// Define if the stencil is correct and has the expected values
/*         int hcx[Q], hcy[Q];
        real_t hw[Q];

        CUDA_CHECK(cudaMemcpyFromSymbol(hcx, d_cx, sizeof(hcx)));
        CUDA_CHECK(cudaMemcpyFromSymbol(hcy, d_cy, sizeof(hcy)));
        CUDA_CHECK(cudaMemcpyFromSymbol(hw, d_w, sizeof(hw)));

        printf("cx = ");
        for (int i = 0; i < Q; i++)
                printf("%d ", hcx[i]);
        printf("\n");

        printf("cy = ");
        for (int i = 0; i < Q; i++)
                printf("%d ", hcy[i]);
        printf("\n");

        printf("w  = ");
        for (int i = 0; i < Q; i++)
                printf("%.10g ", (double)hw[i]);
        printf("\n"); */

#endif