#ifndef STENCIL_CUH
#define STENCIL_CUH

#include "constants.cuh"
#include "utilities/types.cuh"

extern __device__ __constant__ int d_cx[Q];
extern __device__ __constant__ int d_cy[Q];
extern __device__ __constant__ int d_cz[Q];
extern __device__ __constant__ real_t d_w[Q];
extern __device__ __constant__ real_t d_B[Q];

void stencil();

#endif