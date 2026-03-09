#ifndef INDEXING_CUH
#define INDEXING_CUH

#include <cstddef>
#include <cmath>

#include "../constants.cuh"

__host__ __device__ [[nodiscard]] inline std::size_t idx(const std::size_t x, const std::size_t y, const std::size_t z) noexcept
{
    return x + NX * (y + NY * z);
}

__host__ __device__ [[nodiscard]] inline std::size_t fidx(const std::size_t id, const std::size_t i) noexcept
{
    return i * Ncells + id;
}

#endif