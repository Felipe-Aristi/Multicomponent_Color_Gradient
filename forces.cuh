#ifndef FORCES_CUH
#define FORCES_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

__device__ __forceinline__ void force(const real_t *rho_self, const real_t *rho_other,
                                      const size_t x, const size_t y, const size_t z,
                                      real_t &Fx_component, real_t &Fy_component, real_t &Fz_component)
{
    real_t sx = static_cast<real_t>(0.0);
    real_t sy = static_cast<real_t>(0.0);
    real_t sz = static_cast<real_t>(0.0);

#pragma unroll 27
    for (label_t i = 1; i < Q; ++i)
    {

        const real_t cx = static_cast<real_t>(d_cx[i]);
        const real_t cy = static_cast<real_t>(d_cy[i]);
        const real_t cz = static_cast<real_t>(d_cz[i]);
        const real_t wi = static_cast<real_t>(d_w[i]);

        label_t idn = idx(x + d_cx[i], y + d_cy[i], z + d_cz[i]);

        real_t psi_n = psi(rho_self[idn], rho_other[idn]);

        sx += wi * psi_n * cx;
        sy += wi * psi_n * cy;
        sz += wi * psi_n * cz;
    }

    Fx_component = static_cast<real_t>(1) / cs2 * sx;
    Fy_component = static_cast<real_t>(1) / cs2 * sy;
    Fz_component = static_cast<real_t>(1) / cs2 * sz;
}

__device__ __forceinline__ real_t absforce_calcul(const real_t Fx, const real_t Fy, const real_t Fz) noexcept
{
    return sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
}

__device__ __forceinline__ real_t cos2rule(const label_t i, const real_t Fx, const real_t Fy, const real_t Fz, const real_t abs) noexcept
{
    if (i == 0)
    {
        return static_cast<real_t>(0);
    }

    if (abs <= static_cast<real_t>(0.0001))
    {
        return static_cast<real_t>(0);
    }

    real_t Fici = Fx * static_cast<real_t>(d_cx[i]) + Fy * static_cast<real_t>(d_cy[i]) + Fz * static_cast<real_t>(d_cz[i]);
    real_t Fici2 = Fici * Fici;

    real_t abs2 = abs * abs;

    return static_cast<real_t>(Fici2 / abs2);
}

__device__ __forceinline__ real_t tau_interface(const real_t rho_self, const real_t tau_self, const real_t rho_other, const real_t tau_other) noexcept
{
    real_t psix = psi(rho_self, rho_other);
    real_t tau = real_t((static_cast<real_t>(1) + psix) / static_cast<real_t>(2)) * tau_self +
                 real_t((static_cast<real_t>(1) - psix) / static_cast<real_t>(2)) * tau_other;
    return tau;
}

__device__ __forceinline__ real_t A_calculation(const real_t tau) noexcept
{
    return real_t(static_cast<real_t>(1) / (static_cast<real_t>(4) * cs4 * tau)) * sigma;
}

__device__ __forceinline__ void preOmega2(const real_t *rho_self,
                                          const real_t *rho_other,
                                          int x, int y, int z,
                                          real_t tau_self0, real_t tau_other0,
                                          real_t &Fx, real_t &Fy, real_t &Fz,
                                          real_t &absF, real_t &A)
{
    const int id = idx(x, y, z);

    force(rho_self, rho_other, x, y, z, Fx, Fy, Fz);
    real_t tau_eff = tau_interface(rho_self[id], tau_self0, rho_other[id], tau_other0);
    A = A_calculation(tau_eff);
    absF = absforce_calcul(Fx, Fy, Fz);
}

#endif