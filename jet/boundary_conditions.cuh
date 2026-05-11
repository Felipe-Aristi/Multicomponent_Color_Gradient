#ifndef BOUNDARY_CONDITIONS_CUH
#define BOUNDARY_CONDITIONS_CUH

#include "cuda_runtime.h"
#include "../utilities/bounds.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../utilities/mathUtilities.cuh"
#include "../utilities/constexprFor.cuh"
#include "../constants.cuh"
#include "../lbm.cuh"
#include "../forces.cuh"
#include "../collisionOperators.cuh"
#include "../stencil.cuh"
#include "../stencil_ct.cuh"

__device__ __forceinline__ void inlet_calculation(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                                                  pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                                                  const real_t __restrict__ *ux,
                                                  const real_t __restrict__ *uy,
                                                  const real_t __restrict__ *uz,
                                                  const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                                                  const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz,
                                                  const label_t x, const label_t z)
{

    const label_t yB = static_cast<label_t>(0);
    const label_t yF = static_cast<label_t>(1);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    const label_t is_jet = isJet(x, z);

    rhor[idB] = (static_cast<real_t>(1.0) - static_cast<real_t>(is_jet)) * rhor0;
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
    const real_t uxf = ux[idF];
    const real_t uyf = uy[idF];
    const real_t uzf = uz[idF];

    const real_t aR = rhor[idB] * (static_cast<real_t>(1) / rT);
    const real_t aB = rhob[idB] * (static_cast<real_t>(1) / rT);

    const real_t oms = static_cast<real_t>(1.0) - omegab;

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            if constexpr (D3Q27::cy<i>() == 1)
            {
                // Populations with cy = +1 stream from the inlet ghost plane
                // into the first fluid plane.
                const int fluid_nodei = static_cast<int>(idF) + D3Q27::offset_xz<i>();
                const label_t fluid_node = static_cast<label_t>(fluid_nodei);

                const real_t gieq = feq<i>(rT, uxb, uyb, uzb);
                const real_t gineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz, uxf, uyf, uzf);

                const real_t gi = gieq + oms * gineqr;

                fr[fidx(fluid_node, i)] = aR * gi;
                fb[fidx(fluid_node, i)] = aB * gi;
            }
        });
}

__device__ __forceinline__ void neumann_calculation(pop_t __restrict__ *fir, real_t __restrict__ *rhor,
                                                    pop_t __restrict__ *fib, real_t __restrict__ *rhob,
                                                    real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                                                    const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                                                    const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz,
                                                    const label_t x, const label_t z)
{
    const label_t yB = static_cast<label_t>(NY - 1);
    const label_t yF = static_cast<label_t>(NY - 2);

    const label_t idB = idx(x, yB, z);
    const label_t idF = idx(x, yF, z);

    const real_t rrB = rhor[idF];
    const real_t rbB = rhob[idF];
    const real_t rhoT = rrB + rbB;
    const real_t invRhoT = static_cast<real_t>(1.0) / rhoT;

    const real_t uxB = ux[idF];
    const real_t uyB = uy[idF];
    const real_t uzB = uz[idF];

    rhor[idB] = rrB;
    rhob[idB] = rbB;
    ux[idB] = uxB;
    uy[idB] = uyB;
    uz[idB] = uzB;

    const real_t pixx = Pixx[idF];
    const real_t pixy = Pixy[idF];
    const real_t piyy = Piyy[idF];
    const real_t piyz = Piyz[idF];
    const real_t pizz = Pizz[idF];
    const real_t pixz = Pixz[idF];
    const real_t uxf = ux[idF];
    const real_t uyf = uy[idF];
    const real_t uzf = uz[idF];

    const real_t aR = rrB * invRhoT;
    const real_t aB = rbB * invRhoT;

    const real_t omega = omega_sponge(yF);
    const real_t oms = static_cast<real_t>(1.0) - omega;

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            if constexpr (D3Q27::cy<i>() == -1)
            {

                const int fluid_nodei = static_cast<int>(idF) + D3Q27::offset_xz<i>();
                const label_t fluid_node = static_cast<label_t>(fluid_nodei);

                const real_t gieq = feq<i>(rhoT, uxB, uyB, uzB);
                const real_t gineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz, uxf, uyf, uzf);
                const real_t gi = gieq + oms * gineqr;

                fir[fidx(fluid_node, i)] = aR * gi;
                fib[fidx(fluid_node, i)] = aB * gi;
            }
        });
}

#endif
