
#ifndef MATH_UTILITIES_CUH
#define MATH_UTILITIES_CUH

#include <cstddef>
#include <cmath>
#include "../constants.cuh"
#include "../stencil.cuh"

// jet shape inlet
__device__ int isJet(int x, int z)
{
    const real_t dx = static_cast<real_t>(x) - jet_x0;
    const real_t dz = static_cast<real_t>(z) - jet_z0;

    return (dx * dx + dz * dz <= jet_radius * jet_radius) ? 1 : 0;
}

// Phase- Field
__device__ __forceinline__ real_t psi(const real_t rho_self, const real_t rho_other)
{
    return real_t(rho_self - rho_other) / real_t(rho_self + rho_other);
}

// Hermite polynomial H2
__device__ __constant__ constexpr const real_t d_Hxx[Q] = {real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0)};

__device__ __constant__ constexpr const real_t d_Hxy[Q] = {real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(1.0), real_t(1.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(-1.0), real_t(-1.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(1.0), real_t(1.0), real_t(1.0), real_t(1.0), real_t(-1.0), real_t(-1.0), real_t(-1.0), real_t(-1.0)};

__device__ __constant__ constexpr const real_t d_Hyy[Q] = {real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0)};

__device__ __constant__ constexpr const real_t d_Hyz[Q] = {real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(1.0), real_t(1.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(-1.0), real_t(-1.0), real_t(1.0), real_t(1.0), real_t(-1.0), real_t(-1.0), real_t(-1.0), real_t(-1.0), real_t(1.0), real_t(1.0)};

__device__ __constant__ constexpr const real_t d_Hzz[Q] = {real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(-1.0 / 3.0), real_t(-1.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0), real_t(2.0 / 3.0)};

__device__ __constant__ constexpr const real_t d_Hxz[Q] = {real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(1.0), real_t(1.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(0.0), real_t(-1.0), real_t(-1.0), real_t(0.0), real_t(0.0), real_t(1.0), real_t(1.0), real_t(-1.0), real_t(-1.0), real_t(1.0), real_t(1.0), real_t(-1.0), real_t(-1.0)};

// __device__ __forceinline__ void H2(int i, real_t &Hxx, real_t &Hxy, real_t &Hyy,
//                                    real_t &Hyz, real_t &Hzz, real_t &Hxz)
// {
//     const real_t cx_ = static_cast<real_t>(d_cx[i]);
//     const real_t cy_ = static_cast<real_t>(d_cy[i]);
//     const real_t cz_ = static_cast<real_t>(d_cz[i]);
//     Hxx = cx_ * cx_ - cs2;
//     Hxy = cx_ * cy_;
//     Hyy = cy_ * cy_ - cs2;
//     Hyz = cy_ * cz_;
//     Hzz = cz_ * cz_ - cs2;
//     Hxz = cx_ * cz_;
// }

#endif
