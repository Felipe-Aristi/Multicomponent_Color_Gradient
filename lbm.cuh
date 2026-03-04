#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

//----------- Distribution functions and regularization -----------

__device__ __forceinline__ real_t feq(const int i, const real_t rho, const real_t ux, const real_t uy, const real_t uz)
{
    real_t cu = ux * d_cx[i] + uy * d_cy[i] + uz * d_cz[i];
    real_t usq = ux * ux + uy * uy + uz * uz;
    real_t A2eq = (cu / cs2) + (cu * cu) / (2 * cs4) - (usq / (2 * cs2));

    return d_w[i] * rho * (real_t(1.0) + A2eq);
}

__device__ __forceinline__ void pineq(int id, const real_t *f, const real_t *rho,
                                      const real_t *ux, const real_t *uy, const real_t *uz,
                                      real_t &pixx, real_t &pixy, real_t &piyy,
                                      real_t &piyz, real_t &pizz, real_t &pixz)
{
    pixx = real_t(0.0);
    pixy = real_t(0.0);
    piyy = real_t(0.0);
    piyz = real_t(0.0);
    pizz = real_t(0.0);
    pixz = real_t(0.0);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        const real_t fieq = feq(i, rho[id], ux[id], uy[id], uz[id]);
        const real_t fi_neq = f[fidx(id, i)] - fieq;

        real_t Hxx, Hxy, Hyy, Hyz, Hzz, Hxz;
        H2(i, Hxx, Hxy, Hyy, Hyz, Hzz, Hxz);

        pixx += fi_neq * Hxx;
        pixy += fi_neq * Hxy;
        piyy += fi_neq * Hyy;
        piyz += fi_neq * Hyz;
        pizz += fi_neq * Hzz;
        pixz += fi_neq * Hxz;
    }
}

__device__ __forceinline__ real_t fneqr(const int i, const real_t Pixx, const real_t Pixy, const real_t Piyy,
                                        const real_t Piyz, const real_t Pizz, const real_t Pixz)
{
    real_t Hxx, Hxy, Hyy, Hyz, Hzz, Hxz;
    H2(i, Hxx, Hxy, Hyy, Hyz, Hzz, Hxz);
    const real_t a2neq = d_w[i] * (Pixx * Hxx + 2.0 * Pixy * Hxy + Piyy * Hyy + 2.0 * Piyz * Hyz + Pizz * Hzz + 2.0 * Pixz * Hxz) / (2.0 * cs4);
    return a2neq;
}

//-------- Main steps ------------

__device__ __forceinline__ void density_calculation(const real_t *f, real_t *rho,
                                                    const int x, const int y, const int z)
{
    const int id = idx(x, y, z);

    real_t sum = real_t(0.0);

#pragma unroll 27

    for (int i = 0; i < Q; ++i)
    {
        sum += f[fidx(id, i)];
    }

    rho[id] = sum;
}

__device__ __forceinline__ void velocity_calculation(const real_t *f_r, const real_t *rho_r,
                                                     const real_t *f_b, const real_t *rho_b,
                                                     real_t *ux, real_t *uy, real_t *uz,
                                                     const int x, const int y, const int z)
{
    const int id = idx(x, y, z);

    real_t jx = real_t(0.0);
    real_t jy = real_t(0.0);
    real_t jz = real_t(0.0);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        real_t fi = f_r[fidx(id, i)] + f_b[fidx(id, i)];

        jx += fi * d_cx[i];
        jy += fi * d_cy[i];
        jz += fi * d_cz[i];
    }

    const real_t rho_total = rho_r[id] + rho_b[id];

    ux[id] = (jx) / rho_total;
    uy[id] = (jy) / rho_total;
    uz[id] = (jz) / rho_total;
}

__device__ __forceinline__ void PIneq_calculation(const real_t *f, const real_t *rho,
                                                  const real_t *ux, const real_t *uy, const real_t *uz,
                                                  real_t *Pixx, real_t *Pixy, real_t *Piyy,
                                                  real_t *Piyz, real_t *Pizz, real_t *Pixz,
                                                  const int x, const int y, const int z)
{
    const int id = idx(x, y, z);

    real_t pixx, pixy, piyy, piyz, pizz, pixz;
    pineq(id, f, rho, ux, uy, uz, pixx, pixy, piyy, piyz, pizz, pixz);

    Pixx[id] = pixx;
    Pixy[id] = pixy;
    Piyy[id] = piyy;
    Piyz[id] = piyz;
    Pizz[id] = pizz;
    Pixz[id] = pixz;
}

#endif