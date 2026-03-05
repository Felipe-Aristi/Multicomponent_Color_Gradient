#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../utilities/bounds.cuh"
#include "../utilities/mathUtilities.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../constants.cuh"
#include "../stencil.cuh"

// Jet initialization

__device__ void init_density_jet(real_t *f_r, real_t *rho_r, real_t *f_b, real_t *rho_b, int x, int y, int z)
{

    const int id = idx(x, y, z);

    rho_r[id] = rho_r0;
    rho_b[id] = real_t(0.0);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        f_r[fidx(id, i)] = d_w[i] * rho_r0;
        f_b[fidx(id, i)] = d_w[i] * real_t(0.0);
    }
}

__device__ void jet_mask(real_t *f_r, real_t *rho_r, real_t *f_b, real_t *rho_b, int x, int z)
{

    const int yB = 0;
    const int id = idx(x, yB, z);

    const int is_jet = isJet(x, z);
    rho_r[id] = (real_t(1.0) - static_cast<real_t>(is_jet)) * rho_r0;
    rho_b[id] = static_cast<real_t>(is_jet) * rho_b0;

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        f_r[fidx(id, i)] = d_w[i] * rho_r[id];
        f_b[fidx(id, i)] = d_w[i] * rho_r[id];
    }
}

#endif