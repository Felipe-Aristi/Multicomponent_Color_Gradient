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

// inlet boundary condition

__device__ __forceinline__ void inlet_calculation(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                                                  pop_t __restrict__ *fb, real_t __restrict__ *rhob,
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

    const real_t aR = rhor[idB] * (static_cast<real_t>(1) / rT);
    const real_t aB = rhob[idB] * (static_cast<real_t>(1) / rT);

    const real_t oms = static_cast<real_t>(1.0) - omegab;

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
        constexpr label_t i = decltype(I)::value;

       if constexpr (D3Q27::cy<i>() == 1)
        {
            const int fluid_nodei = static_cast<int>(idF) + D3Q27::offset_xz<i>();
            const label_t fluid_node = static_cast<label_t>(fluid_nodei);

            const real_t gieq = feq<i>(rT, uxb, uyb, uzb);
            const real_t gineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz);

            const real_t gi = gieq + oms * gineqr;

            fr[fidx(fluid_node, i)] = save_pop(aR * gi);
            fb[fidx(fluid_node, i)] = save_pop(aB * gi);
        } });
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

    // -----------------------------
    // 1) Old ghost state at outlet
    // -----------------------------
    const real_t rrB_old = rhor[idB];
    const real_t rbB_old = rhob[idB];
    const real_t rhoB_old = rrB_old + rbB_old;

    const real_t jxB_old = rhoB_old * ux[idB];
    const real_t jyB_old = rhoB_old * uy[idB];
    const real_t jzB_old = rhoB_old * uz[idB];

    // ------------------------------------
    // 2) Current state on last fluid plane
    // ------------------------------------
    const real_t rrF = rhor[idF];
    const real_t rbF = rhob[idF];
    const real_t rhoF = rrF + rbF;

    const real_t jxF = rhoF * ux[idF];
    const real_t jyF = rhoF * uy[idF];
    const real_t jzF = rhoF * uz[idF];

    // ---------------------------------------------
    // 3) Explicit convective update on ghost state
    //    CFL-like clipping: 0 <= uc <= jet_velocity
    // ---------------------------------------------
    // const real_t uc_raw = fminf(fmaxf(uy[idF], static_cast<real_t>(0.0)), jet_velocity);
    const real_t uc = jet_velocity;

    const real_t rrB_new = convectiveB(rrB_old, rrF, uc);
    const real_t rbB_new = convectiveB(rbB_old, rbF, uc);

    real_t jxB_new = convectiveB(jxB_old, jxF, uc);
    real_t jyB_new = convectiveB(jyB_old, jyF, uc);
    real_t jzB_new = convectiveB(jzB_old, jzF, uc);

    // Optional but recommended: suppress backflow at the outlet ghost plane
    // jyB_new = fmaxf(jyB_new, static_cast<real_t>(0.0));

    // -----------------------------------------
    // 4) Rebuild ghost macroscopic variables
    // -----------------------------------------
    const real_t rhoT = rrB_new + rbB_new;
    const real_t invRhoT = static_cast<real_t>(1.0) / rhoT;

    const real_t uxB = jxB_new * invRhoT;
    const real_t uyB = static_cast<real_t>(0.005);
    const real_t uzB = jzB_new * invRhoT;

    rhor[idB] = rrB_new;
    rhob[idB] = rbB_new;
    ux[idB] = uxB;
    uy[idB] = uyB;
    uz[idB] = uzB;

    const real_t pixx = Pixx[idF];
    const real_t pixy = Pixy[idF];
    const real_t piyy = Piyy[idF];
    const real_t piyz = Piyz[idF];
    const real_t pizz = Pizz[idF];
    const real_t pixz = Pixz[idF];

    const real_t aR = rrB_new * invRhoT;
    const real_t aB = rbB_new * invRhoT;

    const real_t omega = omegab; // omega_sponge(yF)
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
                const real_t gineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz);
                const real_t gi = gieq + oms * gineqr;

                fir[fidx(fluid_node, i)] = save_pop(aR * gi);
                fib[fidx(fluid_node, i)] = save_pop(aB * gi);
            }
        });
}

#endif