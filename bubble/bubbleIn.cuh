#ifndef BUBBLEIN_CUH
#define BUBBLEIN_CUH

#include "../constants.cuh"
#include "../stencil.cuh"
#include "../utilities/bounds.cuh"
#include "../utilities/types.cuh"
#include "../utilities/indexing.cuh"

__device__ void bubble_mask(real_t *rhor, real_t *rhob,
                            const int x, const int y, const int z)
{
    const int id = idx(x, y, z);

    const real_t R2 = static_cast<real_t>(bubble_radius * bubble_radius);
    const real_t dx = static_cast<real_t>(x - bubble_x0);
    const real_t dy = static_cast<real_t>(y - bubble_y0);
    const real_t dz = static_cast<real_t>(z - bubble_z0);

    const real_t is_bubble = static_cast<real_t>(((dx * dx) + (dy * dy) + (dz * dz)) <= (R2));
    rhor[id] = (real_t(1.0) - is_bubble) * rhor0;
    rhob[id] = is_bubble * rhob0;
}

__device__ void init_equilibrium(pop_t *fr, pop_t *fb,
                                 const real_t *rhor, const real_t *rhob,
                                 const int x, const int y, const int z)
{
    const int id = idx(x, y, z);
    const real_t rhor_i = rhor[id];
    const real_t rhob_i = rhob[id];

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        fr[fidx(id, i)] = save_pop(d_w[i] * (rhor_i - real_t(1.0)));
        fb[fidx(id, i)] = save_pop(d_w[i] * (rhob_i - real_t(1.0)));
    }
}

#endif
