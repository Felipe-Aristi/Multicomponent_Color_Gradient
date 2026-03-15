#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../utilities/bounds.cuh"
#include "../utilities/mathUtilities.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../constants.cuh"
#include "../stencil.cuh"

// Jet initialization
__device__ void init_density_jet(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                                 pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                                 const label_t x, const label_t y, const label_t z)
{

    const label_t id = idx(x, y, z);

    rhor[id] = rhor0;
    rhob[id] = real_t(0.0);

#pragma unroll 27
    for (label_t i = 0; i < Q; ++i)
    {
        fr[fidx(id, i)] = save_pop(d_w[i] * rhor0);
        fb[fidx(id, i)] = save_pop(d_w[i] * real_t(0.0));
    }
}

__device__ void jet_mask(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                         pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                         const label_t x, const label_t z)
{

    const label_t yB = 0;
    const label_t id = idx(x, yB, z);

    const label_t is_jet = isJet(x, z);
    rhor[id] = (real_t(1.0) - static_cast<real_t>(is_jet)) * rhor0;
    rhob[id] = static_cast<real_t>(is_jet) * rhob0;

#pragma unroll 27
    for (label_t i = 0; i < Q; ++i)
    {
        fr[fidx(id, i)] = save_pop(d_w[i] * rhor[id]);
        fb[fidx(id, i)] = save_pop(d_w[i] * rhob[id]);
    }
}

#endif