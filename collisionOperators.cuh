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
__device__ __forceinline__ real_t Omega2(const label_t i, const real_t rho_self, const real_t tau_self, const real_t rho_other, const real_t tau_other,
                                         const real_t Fx, const real_t Fy, const real_t Fz, const real_t absforce, const real_t A) noexcept
{
    const real_t cos2rulei = cos2rule(i, Fx, Fy, Fz, absforce);

    const real_t Omega2 = real_t(A / 2) * absforce * (d_w[i] * cos2rulei - d_B[i]);

    return Omega2;
}

//------- Recoloring Collision operator --------

__device__ __forceinline__ real_t cosphi(const label_t i, real_t Fx, real_t Fy, real_t Fz, const real_t abs) noexcept
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

    return dot * invF * d_invcnorm[i];
}

__device__ __forceinline__ real_t recolorDelta(const label_t i, const real_t rhor, const real_t rhob, const real_t Fx, const real_t Fy, const real_t Fz, const real_t abs) noexcept
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