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
        return static_cast<real_t>(0);
    if (s == 1)
        return static_cast<real_t>(1);
    if (s == 2)
        return rsqrtf(static_cast<real_t>(2));
    return rsqrtf(static_cast<real_t>(3));
}

__device__ __forceinline__ real_t cosphi(int i, real_t Fx, real_t Fy, real_t Fz, const real_t abs)
{
    if (i == 0)
    {
        return real_t(0);
    }

    if (abs <= static_cast<real_t>(0.0001))
    {
        return static_cast<real_t>(0);
    }

    real_t invF = real_t(1 / abs);

    real_t dot = Fx * static_cast<real_t>(d_cx[i]) + Fy * static_cast<real_t>(d_cy[i]) + Fz * static_cast<real_t>(d_cz[i]);

    return dot * invF * inv_cnorm(i);
}

__device__ __forceinline__ real_t recolorDelta(int i, const real_t rhor, const real_t rhob, const real_t Fx, const real_t Fy, const real_t Fz, const real_t abs)
{
    if (i == 0)
    {
        return static_cast<real_t>(0.0);
    }

    const real_t gieq = d_w[i] * (rhor + rhob);

    const real_t Cos = cosphi(i, Fx, Fy, Fz, abs);
    const real_t rho = rhor + rhob;

    const real_t coeff = beta_recolor * real_t((rhor * rhob) / (rho * rho));

    return coeff * Cos * gieq;
}

#endif