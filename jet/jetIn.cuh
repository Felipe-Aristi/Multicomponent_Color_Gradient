#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../utilities/bounds.cuh"
#include "../utilities/mathUtilities.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../constants.cuh"
#include "../stencil.cuh"

// Jet initialization

__device__ void init_density_jet(real_t *fr, real_t *rhor, real_t *fb, real_t *rhob, int x, int y, int z)
{

    const int id = idx(x, y, z);

    rhor[id] = rhor0;
    rhob[id] = real_t(0.0);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        fr[fidx(id, i)] = d_w[i] * rhor0;
        fb[fidx(id, i)] = d_w[i] * real_t(0.0);
    }
}

__device__ void jet_mask(real_t *fr, real_t *rhor, real_t *fb, real_t *rhob, int x, int z)
{

    const int yB = 0;
    const int id = idx(x, yB, z);

    const int is_jet = isJet(x, z);
    rhor[id] = (real_t(1.0) - static_cast<real_t>(is_jet)) * rhor0;
    rhob[id] = static_cast<real_t>(is_jet) * rhob0;

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        fr[fidx(id, i)] = d_w[i] * rhor[id];
        fb[fidx(id, i)] = d_w[i] * rhob[id];
    }
}

#endif