#include "stencil.cuh"
#include "utilities/types.cuh"
#include "utilities/cudaUtilities.cuh"

__device__ __constant__ int d_cx[Q];
__device__ __constant__ int d_cy[Q];
__device__ __constant__ int d_cz[Q];
__device__ __constant__ real_t d_w[Q];
__device__ __constant__ real_t d_B[Q];

inline constexpr real_t w0 = real_t(8.0 / 27.0);
inline constexpr real_t w1 = real_t(2.0 / 27.0);
inline constexpr real_t w2 = real_t(1.0 / 54.0);
inline constexpr real_t w3 = real_t(1.0 / 216.0);

inline constexpr real_t B0 = real_t(-10.0 / 27.0);
inline constexpr real_t B1 = real_t(2.0 / 27.0);
inline constexpr real_t B2 = real_t(1.0 / 54.0);
inline constexpr real_t B3 = real_t(1.0 / 216.0);

void stencil()
{
    const int h_cx[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
    const int h_cy[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
    const int h_cz[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};

    const real_t h_w[Q] = {w0, w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3, w3, w3};
    const real_t h_B[Q] = {B0, B1, B1, B1, B1, B1, B1, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B3, B3, B3, B3, B3, B3, B3, B3};

    CUDA_CHECK(cudaMemcpyToSymbol(d_cx, h_cx, sizeof(h_cx)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cy, h_cy, sizeof(h_cy)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cz, h_cz, sizeof(h_cz)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_w, h_w, sizeof(h_w)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_B, h_B, sizeof(h_B)));
}