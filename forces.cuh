#ifndef FORCES_CUH
#define FORCES_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

__device__ __forceinline__ void force(const real_t *rho_self, const real_t *rho_other,
                                      const int x, const int y, const int z,
                                      real_t &Fx_component, real_t &Fy_component, real_t &Fz_component)
{
    real_t sx = real_t(0.0);
    real_t sy = real_t(0.0);
    real_t sz = real_t(0.0);

#pragma unroll 27
    for (int i = 1; i < Q; ++i)
    {

        int idn = idx(x + d_cx[i], y + d_cy[i], z + d_cz[i]);

        /*         const int xn = wrapx(x + d_cx[i]);
                const int yn = wrapy(y + d_cy[i]);
                const int zn = wrapz(z + d_cz[i]);
                const int idn = idx(xn, yn, zn); */

        real_t psi_n = psi(rho_self[idn], rho_other[idn]);

        sx += d_w[i] * psi_n * d_cx[i];
        sy += d_w[i] * psi_n * d_cy[i];
        sz += d_w[i] * psi_n * d_cz[i];
    }

    Fx_component = real_t(1 / cs2) * sx;
    Fy_component = real_t(1 / cs2) * sy;
    Fz_component = real_t(1 / cs2) * sz;
}

__device__ __forceinline__ real_t absforce_calcul(const real_t Fx, const real_t Fy, const real_t Fz)
{
    return sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
}

__device__ __forceinline__ real_t cos2rule(int i, const real_t Fx, const real_t Fy, const real_t Fz, const real_t abs)
{
    if (i == 0)
    {
        return real_t(0);
    }

    if (abs <= real_t(0.0001))
    {
        return real_t(0);
    }

    real_t Fici = (Fx * d_cx[i] + Fy * d_cy[i] + Fz * d_cz[i]);
    real_t Fici2 = Fici * Fici;

    real_t abs2 = abs * abs;

    return real_t(Fici2 / abs2);
}

__device__ __forceinline__ real_t tau_interface(const real_t rho_self, const real_t tau_self, const real_t rho_other, const real_t tau_other)
{
    real_t psix = psi(rho_self, rho_other);
    real_t tau = real_t((1 + psix) / (2)) * tau_self + real_t((1 - psix) / (2)) * tau_other;
    return tau;
}

__device__ __forceinline__ real_t A_calculation(const real_t tau)
{
    return real_t(1 / (4 * cs4 * tau)) * sigma;
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