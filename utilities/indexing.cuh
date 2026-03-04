#ifndef INDEXING_CUH
#define INDEXING_CUH

#include <cstddef>
#include <cmath>

#include "../constants.cuh"

__host__ __device__ inline int idx(int x, int y, int z)
{
    return x + NX * (y + NY * z);
}

__host__ __device__ inline int fidx(int id, int i)
{
    return (std::size_t)id * (std::size_t)Q + (std::size_t)i;
}

#endif