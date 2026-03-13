#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "stencil_ct.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

//----------- Distribution functions -----------

template <label_t I>
__device__ __forceinline__ real_t feq(const real_t rho,
                                      const real_t ux,
                                      const real_t uy,
                                      const real_t uz) noexcept
{
    constexpr real_t cx = static_cast<real_t>(D3Q27::cx<I>());
    constexpr real_t cy = static_cast<real_t>(D3Q27::cy<I>());
    constexpr real_t cz = static_cast<real_t>(D3Q27::cz<I>());
    constexpr real_t wi = D3Q27::w<I>();

    const real_t cu = ux * cx + uy * cy + uz * cz;
    const real_t usq = ux * ux + uy * uy + uz * uz;
    const real_t A2eq = (cu / cs2) + (cu * cu) / (real_t(2.0) * cs4) - (usq / (real_t(2.0) * cs2));

    return wi * rho * (real_t(1.0) + A2eq);
}

template <label_t I>
__device__ __forceinline__ real_t fneqr(const real_t Pixx,
                                        const real_t Pixy,
                                        const real_t Piyy,
                                        const real_t Piyz,
                                        const real_t Pizz,
                                        const real_t Pixz) noexcept
{
    constexpr real_t Hxx = D3Q27::Hxx<I>();
    constexpr real_t Hxy = D3Q27::Hxy<I>();
    constexpr real_t Hyy = D3Q27::Hyy<I>();
    constexpr real_t Hyz = D3Q27::Hyz<I>();
    constexpr real_t Hzz = D3Q27::Hzz<I>();
    constexpr real_t Hxz = D3Q27::Hxz<I>();

    constexpr real_t wi = D3Q27::w<I>();

    const real_t a2neq = wi * (Pixx * Hxx + real_t(2.0) * Pixy * Hxy + Piyy * Hyy + real_t(2.0) * Piyz * Hyz + Pizz * Hzz + real_t(2.0) * Pixz * Hxz) / (real_t(2.0) * cs4);

    return a2neq;
}

//--------------------

__device__ __forceinline__ void Mfields_calculation(const real_t __restrict__ *fr, real_t __restrict__ *rhor,
                                                    const real_t __restrict__ *fb, real_t __restrict__ *rhob,
                                                    real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                                                    real_t __restrict__ *Pixx, real_t __restrict__ *Pixy, real_t __restrict__ *Piyy,
                                                    real_t __restrict__ *Piyz, real_t __restrict__ *Pizz, real_t __restrict__ *Pixz,
                                                    const label_t x, const label_t y, const label_t z)
{

    const int id = idx(x, y, z);

    real_t sumr = static_cast<real_t>(0.0);
    real_t sumb = static_cast<real_t>(0.0);

    real_t jx = static_cast<real_t>(0.0);
    real_t jy = static_cast<real_t>(0.0);
    real_t jz = static_cast<real_t>(0.0);

    real_t Axx = static_cast<real_t>(0.0);
    real_t Axy = static_cast<real_t>(0.0);
    real_t Ayy = static_cast<real_t>(0.0);
    real_t Ayz = static_cast<real_t>(0.0);
    real_t Azz = static_cast<real_t>(0.0);
    real_t Axz = static_cast<real_t>(0.0);

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            const real_t fr_i = fr[fidx(id, i)];
            const real_t fb_i = fb[fidx(id, i)];

            sumr += fr_i;
            sumb += fb_i;

            const real_t gi = fr_i + fb_i;

            constexpr real_t cx = static_cast<real_t>(D3Q27::cx<I>());
            constexpr real_t cy = static_cast<real_t>(D3Q27::cy<I>());
            constexpr real_t cz = static_cast<real_t>(D3Q27::cz<I>());

            jx += gi * cx;
            jy += gi * cy;
            jz += gi * cz;

            constexpr real_t Hxx = D3Q27::Hxx<I>();
            constexpr real_t Hxy = D3Q27::Hxy<I>();
            constexpr real_t Hyy = D3Q27::Hyy<I>();
            constexpr real_t Hyz = D3Q27::Hyz<I>();
            constexpr real_t Hzz = D3Q27::Hzz<I>();
            constexpr real_t Hxz = D3Q27::Hxz<I>();

            Axx += gi * Hxx;
            Axy += gi * Hxy;
            Ayy += gi * Hyy;
            Ayz += gi * Hyz;
            Azz += gi * Hzz;
            Axz += gi * Hxz;
        });

    rhor[id] = sumr;
    rhob[id] = sumb;

    const real_t rhogi = sumr + sumb;

    // const real_t uxr = fr[1] - fr[2] + fr[7] - fr[8] + fr[9] - fr[10] + fr[13] - fr[14] + fr[15] - fr[16] + fr[19] - fr[20] + fr[21] - fr[22] + fr[23] - fr[24] + fr[26] - fr[25];
    // const real_t uyr = fr[3] - fr[4] + fr[7] - fr[8] + fr[11] - fr[12] + fr[14] - fr[13] + fr[17] - fr[18] + fr[19] - fr[20] + fr[21] - fr[22] + fr[24] - fr[23] + fr[25] - fr[26];
    // const real_t uzr = fr[5] - fr[6] + fr[9] - fr[10] + fr[11] - fr[12] + fr[16] - fr[15] + fr[18] - fr[17] + fr[19] - fr[20] + fr[22] - fr[21] + fr[23] - fr[24] + fr[25] - fr[26];

    // const real_t uxb = fb[1] - fb[2] + fb[7] - fb[8] + fb[9] - fb[10] + fb[13] - fb[14] + fb[15] - fb[16] + fb[19] - fb[20] + fb[21] - fb[22] + fb[23] - fb[24] + fb[26] - fb[25];
    // const real_t uyb = fb[3] - fb[4] + fb[7] - fb[8] + fb[11] - fb[12] + fb[14] - fb[13] + fb[17] - fb[18] + fb[19] - fb[20] + fb[21] - fb[22] + fb[24] - fb[23] + fb[25] - fb[26];
    // const real_t uzb = fb[5] - fb[6] + fb[9] - fb[10] + fb[11] - fb[12] + fb[16] - fb[15] + fb[18] - fb[17] + fb[19] - fb[20] + fb[22] - fb[21] + fb[23] - fb[24] + fb[25] - fb[26];

    // const real_t jx = uxr + uxb;
    // const real_t jy = uyr + uyb;
    // const real_t jz = uzr + uzb;

    const real_t vx = (jx) / rhogi;
    const real_t vy = (jy) / rhogi;
    const real_t vz = (jz) / rhogi;

    ux[id] = vx;
    uy[id] = vy;
    uz[id] = vz;

    Pixx[id] = Axx - rhogi * vx * vx;
    Pixy[id] = Axy - rhogi * vx * vy;
    Piyy[id] = Ayy - rhogi * vy * vy;
    Piyz[id] = Ayz - rhogi * vy * vz;
    Pizz[id] = Azz - rhogi * vz * vz;
    Pixz[id] = Axz - rhogi * vx * vz;
}

#endif