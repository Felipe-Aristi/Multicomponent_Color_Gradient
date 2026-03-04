#ifndef BUBBLEIN_CUH
#define BUBBLEIN_CUH

#include "../constants.cuh"
#include "../stencil.cuh"
#include "../utilities/bounds.cuh"
#include "../utilities/types.cuh"
#include "../utilities/indexing.cuh"

__device__ void bubble_mask(real_t *rho_r, real_t *rho_b,
                            const int x, const int y, const int z)
{
    const int id = idx(x, y, z);

    const real_t R2 = static_cast<real_t>(bubble_radius * bubble_radius);
    const real_t dx = static_cast<real_t>(x - bubble_x0);
    const real_t dy = static_cast<real_t>(y - bubble_y0);
    const real_t dz = static_cast<real_t>(z - bubble_z0);

    const real_t is_bubble = static_cast<real_t>(((dx * dx) + (dy * dy) + (dz * dz)) <= (R2));
    rho_r[id] = (real_t(1.0) - is_bubble) * rho_r0;
    rho_b[id] = is_bubble * rho_b0;
}

__device__ void init_equilibrium(real_t *f_r, real_t *f_b,
                                 const real_t *rho_r, const real_t *rho_b,
                                 const int x, const int y, const int z)
{
    const int id = idx(x, y, z);
    const real_t rho_r_i = rho_r[id];
    const real_t rho_b_i = rho_b[id];

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        f_r[fidx(id, i)] = d_w[i] * rho_r_i;
        f_b[fidx(id, i)] = d_w[i] * rho_b_i;
    }
}

#endif