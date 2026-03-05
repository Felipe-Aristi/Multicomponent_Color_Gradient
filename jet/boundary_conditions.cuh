#ifndef BOUNDARY_CONDITIONS_CUH
#define BOUNDARY_CONDITIONS_CUH

#include "cuda_runtime.h"
#include "../constants.cuh"
#include "../utilities/bounds.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../lbm.cuh"
#include "../stencil.cuh"

// inlet boundary condition

__device__ __forceinline__ void inlet_calculation(real_t *f_r, real_t *rho_r, const real_t *Pixx_r, const real_t *Pixy_r, const real_t *Piyy_r, const real_t *Piyz_r, const real_t *Pizz_r, const real_t *Pixz_r,
                                                  real_t *f_b, real_t *rho_b, const real_t *Pixx_b, const real_t *Pixy_b, const real_t *Piyy_b, const real_t *Piyz_b, const real_t *Pizz_b, const real_t *Pixz_b,
                                                  int x, int z)
{

    const int yB = 0;
    const int yF = 1;

    const int idB = idx(x, yB, z);
    const int idF = idx(x, yF, z);

    const int is_jet = isJet(x, z);

    rho_r[idB] = (real_t(1.0) - static_cast<real_t>(is_jet)) * rho_r0;
    rho_b[idB] = static_cast<real_t>(is_jet) * rho_b0;

    if (is_jet == 0)
    {
        return;
    }

    real_t uxb = real_t(0.0);
    real_t uyb = jet_velocity;
    real_t uzb = real_t(0.0);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        if (d_cy[i] == 1)
        {
            int fluid_node = idx(x + d_cx[i], yF, z + d_cz[i]);

            real_t fieq_r = feq(i, rho_r[idB], real_t(0.0), real_t(0.0), real_t(0.0));
            real_t fineqr_r = fneqr(i, Pixx_r[idF], Pixy_r[idF], Piyy_r[idF], Piyz_r[idF], Pizz_r[idF], Pixz_r[idF]);

            real_t fieq_b = feq(i, rho_b[idB], uxb, uyb, uzb);
            real_t fineqr_b = fneqr(i, Pixx_b[idF], Pixy_b[idF], Piyy_b[idF], Piyz_b[idF], Pizz_b[idF], Pixz_b[idF]);

            f_r[fidx(fluid_node, i)] = fieq_r + (real_t(1.0) - omegar) * fineqr_r;
            f_b[fidx(fluid_node, i)] = fieq_b + (real_t(1.0) - omegab) * fineqr_b;
        }
    }
}

// outlet boundary condition

__device__ __forceinline__ void neumann_calculation(real_t *f, real_t *rho,
                                                    real_t *ux, real_t *uy, real_t *uz,
                                                    const real_t *Pixx, const real_t *Pixy, const real_t *Piyy,
                                                    const real_t *Piyz, const real_t *Pizz, const real_t *Pixz, const real_t omega, int x, int z)
{

    const int yB = NY - 1;
    const int yF = NY - 2;

    const int idB = idx(x, yB, z);
    const int idF = idx(x, yF, z);

    rho[idB] = rho[idF];
    ux[idB] = ux[idF];
    uy[idB] = uy[idF];
    uz[idB] = uz[idF];

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        if (d_cy[i] == -1)
        {
            int fluid_node = idx(x + d_cx[i], yB + d_cy[i], z + d_cz[i]);

            real_t fieq = feq(i, rho[idB], ux[idB], uy[idB], uz[idB]);
            real_t fineqr = fneqr(i, Pixx[idF], Pixy[idF], Piyy[idF], Piyz[idF], Pizz[idF], Pixz[idF]);

            f[fidx(fluid_node, i)] = fieq + (real_t(1.0) - omega) * fineqr;
        }
    }
}

#endif