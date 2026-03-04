#ifndef COLLISIONOPERATORS_CUH
#define COLLISIONOPERATORS_CUH

#include "constants.cuh"
#include "forces.cuh"
#include "lbm.cuh"
#include "stencil.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

//---------- BGK --------
__device__ __forceinline__ real_t Omega1(int i, const real_t rho, const real_t ux, const real_t uy, const real_t uz,
                                         const real_t Pixx, const real_t Pixy, const real_t Piyy,
                                         const real_t Piyz, const real_t Pizz, const real_t Pixz, const real_t omega)
{
    const real_t fieq = feq(i, rho, ux, uy, uz);
    const real_t fineqr = fneqr(i, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz);

    const real_t BGK = fieq + (real_t(1.0) - omega) * fineqr;

    return BGK;
}

//-------- Perturbation collision operator --------
__device__ __forceinline__ real_t Omega2(int i, const real_t rho_self, const real_t tau_self, const real_t rho_other, const real_t tau_other,
                                         const real_t Fx, const real_t Fy, const real_t Fz, const real_t absforce, const real_t A)
{
    const real_t cos2rulei = cos2rule(i, Fx, Fy, Fz, absforce);

    const real_t Omega2 = real_t(A / 2) * absforce * (d_w[i] * cos2rulei - d_B[i]);

    return Omega2;
}

//------- Recoloring Collision operator --------

__device__ __forceinline__ real_t inv_cnorm(int i)
{
    int cx = d_cx[i], cy = d_cy[i], cz = d_cz[i];
    int s = cx * cx + cy * cy + cz * cz; // 0,1,2,3
    if (s == 0)
        return real_t(0);
    if (s == 1)
        return real_t(1);
    if (s == 2)
        return rsqrtf(real_t(2));
    return rsqrtf(real_t(3));
}

__device__ __forceinline__ real_t cosphi(int i, real_t Fx, real_t Fy, real_t Fz, const real_t abs)
{
    if (i == 0)
    {
        return real_t(0);
    }

    if (abs <= real_t(0.0001))
    {
        return real_t(0);
    }

    real_t invF = real_t(1 / abs);

    real_t dot = Fx * real_t(d_cx[i]) + Fy * real_t(d_cy[i]) + Fz * real_t(d_cz[i]);

    return dot * invF * inv_cnorm(i);
}

__device__ __forceinline__ real_t recolorDelta(int i, const real_t rhor, const real_t rhob, const real_t Fx, const real_t Fy, const real_t Fz, const real_t abs)
{
    if (i == 0)
    {
        return real_t(0.0);
    }

    const real_t gieq = d_w[i] * (rhor + rhob);

    const real_t Cos = cosphi(i, Fx, Fy, Fz, abs);
    const real_t rho = rhor + rhob;

    const real_t coeff = beta_recolor * real_t((rhor * rhob) / (rho * rho));

    return coeff * Cos * gieq;
}

#endif