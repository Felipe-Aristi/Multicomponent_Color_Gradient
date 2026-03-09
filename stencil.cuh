#ifndef STENCIL_CUH
#define STENCIL_CUH

#include "constants.cuh"
#include "utilities/types.cuh"

inline constexpr real_t w0 = real_t(8.0 / 27.0);
inline constexpr real_t w1 = real_t(2.0 / 27.0);
inline constexpr real_t w2 = real_t(1.0 / 54.0);
inline constexpr real_t w3 = real_t(1.0 / 216.0);

inline constexpr real_t B0 = real_t(-10.0 / 27.0);
inline constexpr real_t B1 = real_t(2.0 / 27.0);
inline constexpr real_t B2 = real_t(1.0 / 54.0);
inline constexpr real_t B3 = real_t(1.0 / 216.0);

__device__ __constant__ constexpr const int d_cx[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
__device__ __constant__ constexpr const int d_cy[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
__device__ __constant__ constexpr const int d_cz[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};

__device__ __constant__ constexpr const real_t d_w[Q] = {w0, w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3, w3, w3};
__device__ __constant__ constexpr const real_t d_B[Q] = {B0, B1, B1, B1, B1, B1, B1, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B3, B3, B3, B3, B3, B3, B3, B3};

#endif