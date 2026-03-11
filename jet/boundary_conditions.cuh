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

__device__ __forceinline__ void inlet_calculation(real_t __restrict__ *fr, real_t __restrict__ *rhor,
                                                  real_t __restrict__ *fb, real_t __restrict__ *rhob,
                                                  const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                                                  const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz,
                                                  const label_t x, const label_t z)
{

    const label_t yB = 0;
    const label_t yF = 1;

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    const label_t is_jet = isJet(x, z);

    rhor[idB] = (real_t(1.0) - static_cast<real_t>(is_jet)) * rhor0;
    rhob[idB] = static_cast<real_t>(is_jet) * rhob0;

    if (is_jet == 0)
    {
        return;
    }

    const real_t rT = rhor[idB] + rhob[idB];

    real_t uxb = static_cast<real_t>(0.0);
    real_t uyb = jet_velocity;
    real_t uzb = static_cast<real_t>(0.0);

    const real_t pixx = Pixx[idF];
    const real_t pixy = Pixy[idF];
    const real_t piyy = Piyy[idF];
    const real_t piyz = Piyz[idF];
    const real_t pizz = Pizz[idF];
    const real_t pixz = Pixz[idF];

    const real_t aR = rhor[idB] * (static_cast<real_t>(1) / rT);
    const real_t aB = rhob[idB] * (static_cast<real_t>(1) / rT);

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
        constexpr label_t i = decltype(I)::value;

        if (d_cy[i] == 1)
        {
            const label_t xn = x + d_cx[i];
            const label_t zn = z + d_cz[i];

            label_t fluid_node = idx(xn, yF, zn);

            const real_t gieq = feq(i, rT, uxb, uyb, uzb);
            const real_t gineqr = fneqr(i, pixx, pixy, piyy, piyz, pizz, pixz);

            const real_t gi = gieq + (static_cast<real_t>(1.0) - omegab) * gineqr;

            fr[fidx(fluid_node, i)] = aR * gi;
            fb[fidx(fluid_node, i)] = aB * gi;
        } });
}

// outlet boundary condition

__device__ __forceinline__ void neumann_calculation(real_t __restrict__ *fir, real_t __restrict__ *rhor,
                                                    real_t __restrict__ *fib, real_t __restrict__ *rhob,
                                                    real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                                                    const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                                                    const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz,
                                                    const label_t x, const label_t z)
{
    const label_t yB = NY - 1;
    const label_t yF = NY - 2;

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

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
    const real_t invRhoT = static_cast<real_t>(1.0) / (rhoT);

    const real_t aR = rhor[idB] * invRhoT;
    const real_t aB = rhob[idB] * invRhoT;

    const real_t omegaMix = omegar;

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
          constexpr label_t i = decltype(I)::value;

        if (d_cy[i] == -1)
        {
            const label_t xn = x + d_cx[i];
            const label_t zn = z + d_cz[i];

            const label_t idDest = idx(xn, yF, zn);

            const real_t gieq = feq(i, rhoT, ux[idB], uy[idB], uz[idB]);
            const real_t gineqr = fneqr(i, pixx, pixy, piyy, piyz, pizz, pixz);

            const real_t gi = gieq + (static_cast<real_t>(1.0) - omegaMix) * gineqr;

            fir[fidx(idDest, i)] = aR * gi;
            fib[fidx(idDest, i)] = aB * gi;

        } });
}

#endif