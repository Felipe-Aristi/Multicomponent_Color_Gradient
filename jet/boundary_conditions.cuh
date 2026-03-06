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

/* __device__ __forceinline__ void neumann_calculation(real_t *fir, real_t *rhor,
                                                    const real_t *Pixxr, const real_t *Pixyr, const real_t *Piyyr,
                                                    const real_t *Piyzr, const real_t *Pizzr, const real_t *Pixzr,
                                                    real_t *fib, real_t *rhob,
                                                    const real_t *Pixxb, const real_t *Pixyb, const real_t *Piyyb,
                                                    const real_t *Piyzb, const real_t *Pizzb, const real_t *Pixzb,
                                                    real_t *ux, real_t *uy, real_t *uz,
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

    real_t Fxr, Fyr, Fzr, Ar, absforcer;
    preOmega2(rhor, rhob, x, yF, z, taur, taub, Fxr, Fyr, Fzr, absforcer, Ar);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        if (d_cy[i] == -1)
        {
            int fluid_node = idx(x + d_cx[i], yF, z + d_cz[i]);

            const real_t Omega1R = Omega1(i, rhor[idB], ux[idB], uy[idB], uz[idB], Pixxr[idF], Pixyr[idF], Piyyr[idF], Piyzr[idF], Pizzr[idF], Pixzr[idF], omegar);
            const real_t Omega2R = Omega2(i, rhor[idB], taur, rhob[idB], taub, Fxr, Fyr, Fzr, absforcer, Ar);

            const real_t Omega1B = Omega1(i, rhob[idB], ux[idB], uy[idB], uz[idB], Pixxb[idF], Pixyb[idF], Piyyb[idF], Piyzb[idF], Pizzb[idF], Pixzb[idF], omegab);

            const real_t gi = Omega1R + Omega1B + Omega2R;

            const real_t Deltai = recolorDelta(i, rhor[idB], rhob[idB], Fxr, Fyr, Fzr, absforcer);

            fir[fidx(fluid_node, i)] = real_t(rhor[idB] / (rhor[idB] + rhob[idB])) * gi; //

            fib[fidx(fluid_node, i)] = real_t(rhob[idB] / (rhor[idB] + rhob[idB])) * gi; //
        }
    }
} */

__device__ __forceinline__ void neumann_calculation(real_t *fir, real_t *rhor,
                                                    const real_t *Pixxr, const real_t *Pixyr, const real_t *Piyyr,
                                                    const real_t *Piyzr, const real_t *Pizzr, const real_t *Pixzr,
                                                    real_t *fib, real_t *rhob,
                                                    const real_t *Pixxb, const real_t *Pixyb, const real_t *Piyyb,
                                                    const real_t *Piyzb, const real_t *Pizzb, const real_t *Pixzb,
                                                    real_t *ux, real_t *uy, real_t *uz,
                                                    int x, int z)
{
    const int yB = NY - 1;   // ghost plane (not streamed by ColliStream)
    const int yOut = NY - 2; // last fluid plane (needs incoming cy=-1)
    const int ySrc = NY - 2; // interior source plane (critical)

    const int idB = idx(x, yB, z);
    const int idSrc = idx(x, ySrc, z);

    // Copy macros (Neumann)
    rhor[idB] = rhor[idSrc];
    rhob[idB] = rhob[idSrc];
    ux[idB] = ux[idSrc];
    uy[idB] = uy[idSrc];
    uz[idB] = uz[idSrc];

    const real_t rhoT = rhor[idB] + rhob[idB];
    const real_t eps = real_t(1e-12);
    const real_t invRhoT = real_t(1.0) / (rhoT + eps);

    const real_t aR = rhor[idB] * invRhoT;
    const real_t aB = rhob[idB] * invRhoT;

    // TOTAL noneq stress for regularization
    const real_t PixxT = Pixxr[idSrc] + Pixxb[idSrc];
    const real_t PixyT = Pixyr[idSrc] + Pixyb[idSrc];
    const real_t PiyyT = Piyyr[idSrc] + Piyyb[idSrc];
    const real_t PiyzT = Piyzr[idSrc] + Piyzb[idSrc];
    const real_t PizzT = Pizzr[idSrc] + Pizzb[idSrc];
    const real_t PixzT = Pixzr[idSrc] + Pixzb[idSrc];

    // Choose ONE relaxation for the mixture (simple choice: same tau)
    const real_t omegaMix = omegar; // or 0.5*(omegar+omegab)

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        if (d_cy[i] != -1)
            continue;

        const int xn = x + d_cx[i];
        const int zn = z + d_cz[i];
        if (xn <= 0 || xn >= NX - 1 || zn <= 0 || zn >= NZ - 1)
            continue;

        const int idDest = idx(xn, yOut, zn);

        // Regularized reconstruction of g_i
        const real_t geq = feq(i, rhoT, ux[idB], uy[idB], uz[idB]);
        const real_t gneq = fneqr(i, PixxT, PixyT, PiyyT, PiyzT, PizzT, PixzT);
        const real_t gi = geq + (real_t(1.0) - omegaMix) * gneq;

        // NO Omega2 and NO recoloring at outlet (prevents "wall")
        fir[fidx(idDest, i)] = aR * gi;
        fib[fidx(idDest, i)] = aB * gi;
    }
}

#endif