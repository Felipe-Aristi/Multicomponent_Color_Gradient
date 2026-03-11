#ifndef BOUNDS_CUH
#define BOUNDS_CUH

#include "../constants.cuh"

__device__ [[nodiscard]] constexpr inline bool interior(const label_t x, const label_t y, const label_t z) noexcept
{

    return (x >= NX || y >= NY || z >= NZ ||
            x == 0 || x == NX - 1 ||
            y == 0 || y == NY - 1 ||
            z == 0 || z == NZ - 1);
}

__device__ [[nodiscard]] constexpr inline bool inlet_oulet_interior(const label_t x, const label_t z) noexcept
{
    return (x >= NX || z >= NZ ||
            x == 0 || x == NX - 1 ||
            z == 0 || z == NZ - 1);
}

//  Periodic boundary conditions
__device__ inline int wrapx(int x)
{
    if (x == 0)
    {
        return NX - 2;
    }
    if (x == NX - 1)
    {
        return 1;
    }
    return x;
}

__device__ inline int wrapy(int y)
{
    if (y == 0)
    {
        return NY - 2;
    }
    if (y == NY - 1)
    {
        return 1;
    }
    return y;
}

__device__ inline int wrapz(int z)
{
    if (z == 0)
    {
        return NZ - 2;
    }
    if (z == NZ - 1)
    {
        return 1;
    }
    return z;
}

#endif