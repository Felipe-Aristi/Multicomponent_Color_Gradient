#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "stencil_ct.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"
#include "utilities/constexprFor.cuh"

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

    const real_t cu2 = cu * cu;

    const real_t A2eq = (cu * cu) * inv_2cs4 - usq * inv_2cs2 + cu * inv_cs2;
    const real_t A3eq = cu * (cu2 * inv_6cs6 - usq * inv_2cs4);

    return wi * rho * (real_t(1.0) + A2eq + A3eq);
}

template <label_t I>
__device__ __forceinline__ real_t fneqr(const real_t Pixx,
                                        const real_t Pixy,
                                        const real_t Piyy,
                                        const real_t Piyz,
                                        const real_t Pizz,
                                        const real_t Pixz,
                                        const real_t ux,
                                        const real_t uy,
                                        const real_t uz) noexcept
{
    constexpr real_t Hxx = D3Q27::Hxx<I>();
    constexpr real_t Hxy = D3Q27::Hxy<I>();
    constexpr real_t Hyy = D3Q27::Hyy<I>();
    constexpr real_t Hyz = D3Q27::Hyz<I>();
    constexpr real_t Hzz = D3Q27::Hzz<I>();
    constexpr real_t Hxz = D3Q27::Hxz<I>();

    constexpr real_t Hxxy = D3Q27::Hxxy<I>();
    constexpr real_t Hxxz = D3Q27::Hxxz<I>();
    constexpr real_t Hxyy = D3Q27::Hxyy<I>();
    constexpr real_t Hxzz = D3Q27::Hxzz<I>();
    constexpr real_t Hyyz = D3Q27::Hyyz<I>();
    constexpr real_t Hyzz = D3Q27::Hyzz<I>();
    constexpr real_t Hxyz = D3Q27::Hxyz<I>();

    constexpr real_t wi = D3Q27::w<I>();

    const real_t A2neq = (Pixx * Hxx + real_t(2.0) * Pixy * Hxy + Piyy * Hyy + real_t(2.0) * Piyz * Hyz + Pizz * Hzz + real_t(2.0) * Pixz * Hxz) * inv_2cs4;

    const real_t a3xxy = Pixx * uy + real_t(2.0) * Pixy * ux;
    const real_t a3xxz = Pixx * uz + real_t(2.0) * Pixz * ux;
    const real_t a3xyy = Piyy * ux + real_t(2.0) * Pixy * uy;
    const real_t a3xzz = Pizz * ux + real_t(2.0) * Pixz * uz;
    const real_t a3yyz = Piyy * uz + real_t(2.0) * Piyz * uy;
    const real_t a3yzz = Pizz * uy + real_t(2.0) * Piyz * uz;
    const real_t a3xyz = Pixy * uz + Pixz * uy + Piyz * ux;

    const real_t A3neq = (a3xxy * Hxxy + a3xxz * Hxxz + a3xyy * Hxyy + a3xzz * Hxzz + a3yyz * Hyyz + a3yzz * Hyzz + real_t(2.0) * a3xyz * Hxyz) * inv_2cs6;

    return wi * (A2neq + A3neq);
}

//------------- Macroscopic fields calculation ------------

__device__ __forceinline__ void Mfields_calculation(const pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                                                    const pop_t __restrict__ *fb, real_t __restrict__ *rhob,
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

            constexpr real_t Hxx = D3Q27::Hxx<i>();
            constexpr real_t Hxy = D3Q27::Hxy<i>();
            constexpr real_t Hyy = D3Q27::Hyy<i>();
            constexpr real_t Hyz = D3Q27::Hyz<i>();
            constexpr real_t Hzz = D3Q27::Hzz<i>();
            constexpr real_t Hxz = D3Q27::Hxz<i>();

            Axx += gi * Hxx;
            Axy += gi * Hxy;
            Ayy += gi * Hyy;
            Ayz += gi * Hyz;
            Azz += gi * Hzz;
            Axz += gi * Hxz;
        });

    const real_t rhorr = sumr;
    const real_t rhobb = sumb;

    rhor[id] = rhorr;
    rhob[id] = rhobb;

    const real_t rhogi = rhorr + rhobb;
    const real_t invrhogi = static_cast<real_t>(1.0) / rhogi;

    const real_t vx = (jx)*invrhogi;
    const real_t vy = (jy)*invrhogi;
    const real_t vz = (jz)*invrhogi;

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

//------------- collision and streaming calculation ------------

__device__ __forceinline__ void preOmega2(const real_t __restrict__ *rho_self,
                                          const real_t __restrict__ *rho_other,
                                          const label_t id,
                                          real_t tau_self0, real_t tau_other0,
                                          real_t &Fx, real_t &Fy, real_t &Fz,
                                          real_t &absF, real_t &A);

template <label_t I>
__device__ __forceinline__ real_t Omega2(const real_t Fx, const real_t Fy, const real_t Fz, const real_t absforce, const real_t A) noexcept;

template <label_t I>
__device__ __forceinline__ real_t recolorDelta(const real_t rhor, const real_t rhob,
                                               const real_t Fx, const real_t Fy, const real_t Fz,
                                               const real_t absF) noexcept;

__device__ __forceinline__ void ColliStream_calculations(pop_t __restrict__ *fir, const real_t __restrict__ *rhor,
                                                         pop_t __restrict__ *fib, const real_t __restrict__ *rhob,
                                                         const real_t __restrict__ *ux, const real_t __restrict__ *uy, const real_t __restrict__ *uz,
                                                         const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                                                         const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz,
                                                         const label_t x, const label_t y, const label_t z)
{
    const label_t id = idx(x, y, z);

    const real_t rr = rhor[id];
    const real_t rb = rhob[id];
    const real_t rT = rr + rb;
    const real_t invrT = static_cast<real_t>(1) / static_cast<real_t>(rT);

    const real_t aR = real_t(rr * invrT);
    const real_t aB = real_t(rb * invrT);

    const real_t vx = ux[id];
    const real_t vy = uy[id];
    const real_t vz = uz[id];
    const real_t pixx = Pixx[id];
    const real_t pixy = Pixy[id];
    const real_t piyy = Piyy[id];
    const real_t piyz = Piyz[id];
    const real_t pizz = Pizz[id];
    const real_t pixz = Pixz[id];

    const real_t omega = omega_sponge(y);
    const real_t oms = (static_cast<real_t>(1.0) - omega);

    const real_t I = real_t(4.0) * rr * rb * invrT * invrT;

    if (I <= real_t(1e-4))
    {
        constexpr_for<0, Q>(
            [&] __device__(auto I)
            {
                constexpr label_t i = decltype(I)::value;

                const real_t fieq = feq<i>(rT, vx, vy, vz);
                const real_t gineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz, vx, vy, vz);

                const real_t Omega1 = fieq + oms * gineqr;

                const real_t gi = Omega1;

                // const int xn = static_cast<int>(x) + D3Q27::cx<i>();
                // const int zn = static_cast<int>(z) + D3Q27::cz<i>();

                // // periodic boundary condition
                const int xn = wrapx(static_cast<int>(x) + D3Q27::cx<i>());
                const int zn = wrapz(static_cast<int>(z) + D3Q27::cz<i>());

                const int yn = static_cast<int>(y) + D3Q27::cy<i>();

                const label_t idn = idx(static_cast<label_t>(xn),
                                        static_cast<label_t>(yn),
                                        static_cast<label_t>(zn));

                fir[fidx(idn, i)] = aR * gi;
                fib[fidx(idn, i)] = aB * gi;
            });
        return;
    }

    real_t Fxr, Fyr, Fzr, Ar, absforcer;
    preOmega2(rhor, rhob, id, taur, taub, Fxr, Fyr, Fzr, absforcer, Ar);

    // const real_t Fxb = -Fxr;
    // const real_t Fyb = -Fyr;
    // const real_t Fzb = -Fzr;
    // const real_t Ab = Ar;

    // real_t Fxb, Fyb, Fzb, Ab, absforceb;
    // preOmega2(rhob, rhor, id, taub, taur, Fxb, Fyb, Fzb, absforceb, Ab);

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            const real_t fieq = feq<i>(rT, vx, vy, vz);
            const real_t gineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz, vx, vy, vz);

            const real_t Omega1 = fieq + oms * gineqr;

            const real_t Omega2R = Omega2<i>(Fxr, Fyr, Fzr, absforcer, Ar);
            // const real_t Omega2B = Omega2<i>(Fxb, Fyb, Fzb, absforcer, Ab);

            const real_t gi = Omega1 + Omega2R; //+ Omega2B

            const real_t Deltai = recolorDelta<i>(rr, rb, Fxr, Fyr, Fzr, absforcer);

            // const int xn = static_cast<int>(x) + D3Q27::cx<i>();
            // const int zn = static_cast<int>(z) + D3Q27::cz<i>();

            // periodic boundary condition
            const int xn = wrapx(static_cast<int>(x) + D3Q27::cx<i>());
            const int zn = wrapz(static_cast<int>(z) + D3Q27::cz<i>());

            const int yn = static_cast<int>(y) + D3Q27::cy<i>();

            const label_t idn = idx(static_cast<label_t>(xn),
                                    static_cast<label_t>(yn),
                                    static_cast<label_t>(zn));

            fir[fidx(idn, i)] = aR * gi + Deltai;
            fib[fidx(idn, i)] = aB * gi - Deltai;
        });
}

#endif
