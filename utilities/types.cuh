#ifndef TYPES_CUH
#define TYPES_CUH

#include <cstdint>

using real_t = float;
using label_t = uint32_t;

using pop_t = float; // later: __half or custom 16-bit storage

__device__ __forceinline__ real_t load_pop(const pop_t *f, int k) noexcept
{
    return static_cast<real_t>(f[k]);
}

__device__ __forceinline__ void store_pop(pop_t *f, int k, real_t x) noexcept
{
    f[k] = static_cast<pop_t>(x);
}

#endif