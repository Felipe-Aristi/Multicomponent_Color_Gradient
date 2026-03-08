#ifndef BOUNDARY_CONDITIONS_CUH
#define BOUNDARY_CONDITIONS_CUH

#include "cuda_runtime.h"
#include "../utilities/bounds.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../constants.cuh"
#include "../lbm.cuh"
#include "../forces.cuh"
#include "../collisionOperators.cuh"
#include "../stencil.cuh"

// inlet boundary condition

__device__ __forceinline__ void inlet_calculation(real_t *fr, real_t *rhor, real_t *fb, real_t *rhob,
                                                  const real_t *Pixx, const real_t *Pixy, const real_t *Piyy, const real_t *Piyz, const real_t *Pizz, const real_t *Pixz,
                                                  int x, int z)
{

    const int yB = 0;
    const int yF = 1;

    const int idB = idx(x, yB, z);
    const int idF = idx(x, yF, z);

    const int is_jet = isJet(x, z);

    rhor[idB] = (real_t(1.0) - static_cast<real_t>(is_jet)) * rhor0;
    rhob[idB] = static_cast<real_t>(is_jet) * rhob0;

    if (is_jet == 0)
    {
        return;
    }

    const real_t rT = rhor[idB] + rhob[idB];

    real_t uxb = real_t(0.0);
    real_t uyb = jet_velocity;
    real_t uzb = real_t(0.0);

    const real_t pixx = Pixx[idF];
    const real_t pixy = Pixy[idF];
    const real_t piyy = Piyy[idF];
    const real_t piyz = Piyz[idF];
    const real_t pizz = Pizz[idF];
    const real_t pixz = Pixz[idF];

    const real_t aR = rhor[idB] * real_t(1 / rT);
    const real_t aB = rhob[idB] * real_t(1 / rT);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        if (d_cy[i] == 1)
        {
            const int xn = x + d_cx[i];
            const int zn = z + d_cz[i];

            int fluid_node = idx(xn, yF, zn);

            const real_t gieq = feq(i, rT, uxb, uyb, uzb);
            const real_t gineqr = fneqr(i, pixx, pixy, piyy, piyz, pizz, pixz);

            const real_t gi = gieq + (real_t(1.0) - omegab) * gineqr;

            fr[fidx(fluid_node, i)] = aR * gi;
            fb[fidx(fluid_node, i)] = aB * gi;
        }
    }
}

// outlet boundary condition

__device__ __forceinline__ void neumann_calculation(real_t *fir, real_t *rhor,
                                                    real_t *fib, real_t *rhob,
                                                    real_t *ux, real_t *uy, real_t *uz,
                                                    const real_t *Pixx, const real_t *Pixy, const real_t *Piyy, const real_t *Piyz, const real_t *Pizz, const real_t *Pixz,
                                                    int x, int z)
{
    const int yB = NY - 1;
    const int yF = NY - 2;

    const int idB = idx(x, yB, z);
    const int idF = idx(x, yF, z);

    rhor[idB] = rhor[idF];
    rhob[idB] = rhob[idF];
    ux[idB] = ux[idF];
    uy[idB] = uy[idF];
    uz[idB] = uz[idF];

    const real_t pixx = Pixx[idF];
    const real_t pixy = Pixy[idF];
    const real_t piyy = Piyy[idF];
    const real_t piyz = Piyz[idF];
    const real_t pizz = Pizz[idF];
    const real_t pixz = Pixz[idF];

    const real_t rhoT = rhor[idB] + rhob[idB];
    const real_t invRhoT = real_t(1.0) / (rhoT);

    const real_t aR = rhor[idB] * invRhoT;
    const real_t aB = rhob[idB] * invRhoT;

    const real_t omegaMix = omegar;

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        if (d_cy[i] == -1)
        {
            const int xn = x + d_cx[i];
            const int zn = z + d_cz[i];

            const int idDest = idx(xn, yF, zn);

            const real_t gieq = feq(i, rhoT, ux[idB], uy[idB], uz[idB]);
            const real_t gineqr = fneqr(i, pixx, pixy, piyy, piyz, pizz, pixz);

            const real_t gi = gieq + (real_t(1.0) - omegaMix) * gineqr;

            fir[fidx(idDest, i)] = aR * gi;
            fib[fidx(idDest, i)] = aB * gi;
        }
    }
}

#endif