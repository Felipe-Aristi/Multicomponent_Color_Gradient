#ifndef COLLISIONOPERATORS_CUH
#define COLLISIONOPERATORS_CUH

#include "constants.cuh"
#include "forces.cuh"
#include "lbm.cuh"
#include "stencil.cuh"
#include "stencil_ct.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

//-------- Perturbation collision operator --------
template <label_t I>
__device__ __forceinline__ real_t Omega2(const real_t Fx, const real_t Fy, const real_t Fz, const real_t absforce, const real_t A) noexcept
{

    const real_t cos2rulei = cos2rule<I>(Fx, Fy, Fz, absforce);

    const real_t Omega2 = real_t(A / 2) * absforce * (D3Q27::w<I>() * cos2rulei - D3Q27::B<I>());

    return Omega2;
}

//------- Recoloring Collision operator --------

template <label_t I>
__device__ __forceinline__ real_t cosphi(const real_t Fx, const real_t Fy, const real_t Fz, const real_t absF) noexcept
{
    if constexpr (I == 0)
    {
        return real_t(0);
    }

    if (absF <= real_t(1.0e-4))
    {
        return real_t(0);
    }

    constexpr real_t cx = static_cast<real_t>(D3Q27::cx<I>());
    constexpr real_t cy = static_cast<real_t>(D3Q27::cy<I>());
    constexpr real_t cz = static_cast<real_t>(D3Q27::cz<I>());

    const real_t invF = real_t(1.0) / absF;
    const real_t dot = Fx * cx + Fy * cy + Fz * cz;

    return dot * invF * D3Q27::invcnorm<I>();
}

template <label_t I>
__device__ __forceinline__ real_t recolorDelta(const real_t rhor, const real_t rhob,
                                               const real_t Fx, const real_t Fy, const real_t Fz,
                                               const real_t absF) noexcept
{
    if constexpr (I == 0)
    {
        return real_t(0);
    }

    const real_t rho = rhor + rhob;
    const real_t gieq = D3Q27::w<I>() * rho;
    const real_t Cos = cosphi<I>(Fx, Fy, Fz, absF);
    const real_t coeff = beta_recolor * ((rhor * rhob) / (rho * rho));

    return coeff * Cos * gieq;
}

#endif