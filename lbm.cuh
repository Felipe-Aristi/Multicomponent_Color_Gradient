#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

//----------- Distribution functions -----------

__device__ __forceinline__ real_t feq(const int i, const real_t rho, const real_t ux, const real_t uy, const real_t uz)
{
    const real_t cx = static_cast<real_t>(d_cx[i]);
    const real_t cy = static_cast<real_t>(d_cy[i]);
    const real_t cz = static_cast<real_t>(d_cz[i]);
    const real_t wi = static_cast<real_t>(d_w[i]);

    real_t cu = ux * cx + uy * cy + uz * cz;
    real_t usq = ux * ux + uy * uy + uz * uz;
    real_t A2eq = (cu / cs2) + (cu * cu) / (2 * cs4) - (usq / (2 * cs2));

    return wi * rho * (static_cast<real_t>(1.0) + A2eq);
}

__device__ __forceinline__ real_t fneqr(const int i, const real_t Pixx, const real_t Pixy, const real_t Piyy,
                                        const real_t Piyz, const real_t Pizz, const real_t Pixz)
{
    real_t Hxx, Hxy, Hyy, Hyz, Hzz, Hxz;
    H2(i, Hxx, Hxy, Hyy, Hyz, Hzz, Hxz);

    const real_t wi = static_cast<real_t>(d_w[i]);

    const real_t a2neq = wi * (Pixx * Hxx + 2.0 * Pixy * Hxy + Piyy * Hyy + 2.0 * Piyz * Hyz + Pizz * Hzz + 2.0 * Pixz * Hxz) / (2.0 * cs4);
    return a2neq;
}

//--------------------

__device__ __forceinline__ void Mfields_calculation(const real_t *fr, real_t *rhor,
                                                    const real_t *fb, real_t *rhob,
                                                    real_t *ux, real_t *uy, real_t *uz,
                                                    real_t *Pixx, real_t *Pixy, real_t *Piyy,
                                                    real_t *Piyz, real_t *Pizz, real_t *Pixz,
                                                    const int x, const int y, const int z)
{

    const int id = idx(x, y, z);

    real_t sumr = static_cast<real_t>(0.0);
    real_t sumb = static_cast<real_t>(0.0);

    real_t jx = static_cast<real_t>(0.0);
    real_t jy = static_cast<real_t>(0.0);
    real_t jz = static_cast<real_t>(0.0);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {

        const real_t fr_i = fr[fidx(id, i)];
        const real_t fb_i = fb[fidx(id, i)];

        const real_t gi = fr_i + fb_i;

        const real_t cx = static_cast<real_t>(d_cx[i]);
        const real_t cy = static_cast<real_t>(d_cy[i]);
        const real_t cz = static_cast<real_t>(d_cz[i]);

        sumr += fr_i;
        sumb += fb_i;

        jx += gi * cx;
        jy += gi * cy;
        jz += gi * cz;
    }

    rhor[id] = sumr;
    rhob[id] = sumb;

    const real_t rhogi = sumr + sumb;

    const real_t vx = (jx) / rhogi;
    const real_t vy = (jy) / rhogi;
    const real_t vz = (jz) / rhogi;

    ux[id] = vx;
    uy[id] = vy;
    uz[id] = vz;

    real_t pixx = static_cast<real_t>(0.0);
    real_t pixy = static_cast<real_t>(0.0);
    real_t piyy = static_cast<real_t>(0.0);
    real_t piyz = static_cast<real_t>(0.0);
    real_t pizz = static_cast<real_t>(0.0);
    real_t pixz = static_cast<real_t>(0.0);

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {
        const real_t fir = fr[fidx(id, i)];
        const real_t fib = fb[fidx(id, i)];

        const real_t gieq = feq(i, rhogi, vx, vy, vz);

        const real_t gi_neq = (fir + fib) - gieq;

        real_t Hxx, Hxy, Hyy, Hyz, Hzz, Hxz;
        H2(i, Hxx, Hxy, Hyy, Hyz, Hzz, Hxz);

        pixx += gi_neq * Hxx;
        pixy += gi_neq * Hxy;
        piyy += gi_neq * Hyy;
        piyz += gi_neq * Hyz;
        pizz += gi_neq * Hzz;
        pixz += gi_neq * Hxz;
    }

    Pixx[id] = pixx;
    Pixy[id] = pixy;
    Piyy[id] = piyy;
    Piyz[id] = piyz;
    Pizz[id] = pizz;
    Pixz[id] = pixz;
}

#endif