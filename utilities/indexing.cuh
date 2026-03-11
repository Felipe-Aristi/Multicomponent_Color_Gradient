#ifndef INDEXING_CUH
#define INDEXING_CUH

#include <cstddef>
#include <cmath>

#include "../constants.cuh"
#include "types.cuh"

__host__ __device__ [[nodiscard]] constexpr inline label_t idx(const label_t x, const label_t y, const label_t z) noexcept
{
    return x + NX * (y + NY * z);
}

__host__ __device__ [[nodiscard]] constexpr inline label_t fidx(const label_t id, const label_t i) noexcept
{
    // return (std::size_t)id * (std::size_t)Q + (std::size_t)i;
    return (std::size_t)i * (std::size_t)Ncells + (std::size_t)id;
}

#endif