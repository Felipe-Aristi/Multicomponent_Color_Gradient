#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "utilities/bounds.cuh"
#include "utilities/indexing.cuh"
#include "utilities/types.cuh"
#include "utilities/mathUtilities.cuh"

//----------- Distribution functions -----------

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

namespace device
{
    template <typename T, T v>
    struct integralConstant
    {
        static constexpr const T value = v;
        using value_type = T;
        using type = integralConstant;
        __device__ __host__ [[nodiscard]] inline consteval operator value_type() const noexcept { return value; }
        __device__ __host__ [[nodiscard]] inline consteval value_type operator()() const noexcept { return value; }
    };

    template <const std::size_t Start, const std::size_t End, typename F>
    __device__ inline constexpr void constexpr_for(F &&f)
    {
        if constexpr (Start < End)
        {
            f(integralConstant<std::size_t, Start>());
            if constexpr (Start + 1 < End)
            {
                device::constexpr_for<Start + 1, End>(std::forward<F>(f));
            }
        }
    }
}

template <const int coeff>
__device__ [[nodiscard]] inline constexpr real_t process_element(const real_t f) noexcept
{
    if constexpr (coeff == 1)
    {
        return f;
    }

    if constexpr (coeff == 0)
    {
        return 0;
    }

    if constexpr (coeff == -1)
    {
        return -f;
    }
}

template <const std::size_t i>
__device__ [[nodiscard]] inline consteval int cx() noexcept
{
    constexpr const int h_cx[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};

    return h_cx[i];
}

template <const std::size_t i>
__device__ [[nodiscard]] inline consteval int cy() noexcept
{
    constexpr const int h_cy[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};

    return h_cy[i];
}

template <const std::size_t i>
__device__ [[nodiscard]] inline consteval int cz() noexcept
{
    constexpr const int h_cz[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};

    return h_cz[i];
}

inline constexpr real_t w0 = real_t(8.0 / 27.0);
inline constexpr real_t w1 = real_t(2.0 / 27.0);
inline constexpr real_t w2 = real_t(1.0 / 54.0);
inline constexpr real_t w3 = real_t(1.0 / 216.0);

inline constexpr real_t B0 = real_t(-10.0 / 27.0);
inline constexpr real_t B1 = real_t(2.0 / 27.0);
inline constexpr real_t B2 = real_t(1.0 / 54.0);
inline constexpr real_t B3 = real_t(1.0 / 216.0);

template <const std::size_t i>
__device__ [[nodiscard]] inline consteval real_t w() noexcept
{
    constexpr const real_t h_w[Q] = {w0, w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3, w3, w3};

    return h_w[i];
}

template <const std::size_t i>
__device__ [[nodiscard]] inline consteval real_t B() noexcept
{
    constexpr const real_t h_B[Q] = {B0, B1, B1, B1, B1, B1, B1, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B2, B3, B3, B3, B3, B3, B3, B3, B3};

    return h_B[i];
}

template <const std::size_t i>
__device__ [[nodiscard]] inline constexpr real_t velocity_sum(const real_t ux, const real_t uy, const real_t uz) noexcept
{
    return (process_element<cx<i>()>(ux)) + (process_element<cy<i>()>(uy)) + (process_element<cz<i>()>(uz));
}

__device__ __forceinline__ real_t feq(const std::size_t i, const real_t rho, const real_t ux, const real_t uy, const real_t uz) noexcept
{
    const real_t cx = static_cast<real_t>(d_cx[i]);
    const real_t cy = static_cast<real_t>(d_cy[i]);
    const real_t cz = static_cast<real_t>(d_cz[i]);
    const real_t wi = static_cast<real_t>(d_w[i]);

    const real_t cu = (cx * ux) + (cy * uy) * (cz * uz);
    const real_t usq = ux * ux + uy * uy + uz * uz;
    const real_t A2eq = (cu / cs2) + (cu * cu) / (static_cast<real_t>(2) * cs4) - (usq / (static_cast<real_t>(2) * cs2));

    return wi * rho * (static_cast<real_t>(1) + A2eq);
}

template <const std::size_t i>
__device__ __forceinline__ real_t feq(const real_t rho, const real_t ux, const real_t uy, const real_t uz) noexcept
{
    // const real_t cx = static_cast<real_t>(d_cx[i]);
    // const real_t cy = static_cast<real_t>(d_cy[i]);
    // const real_t cz = static_cast<real_t>(d_cz[i]);
    // const real_t wi = static_cast<real_t>(d_w[i]);

    const real_t cu = velocity_sum<i>(ux, uy, uz);
    const real_t usq = ux * ux + uy * uy + uz * uz;
    const real_t A2eq = (cu / cs2) + (cu * cu) / (static_cast<real_t>(2) * cs4) - (usq / (static_cast<real_t>(2) * cs2));

    return w<i>() * rho * (static_cast<real_t>(1) + A2eq);
}

static constexpr const std::size_t xAxis = 0;
static constexpr const std::size_t yAxis = 1;
static constexpr const std::size_t zAxis = 2;

template <const std::size_t alpha, const std::size_t beta, const std::size_t i>
__device__ [[nodiscard]] inline consteval bool H2_returns_zero() noexcept
{
    if constexpr (alpha == xAxis)
    {
        if constexpr (beta == xAxis)
        {
            return ((3 * cx<i>() * cx<i>()) - 1) == 0;
        }

        if constexpr (beta == yAxis)
        {
            return (3 * cx<i>() * cy<i>()) == 0;
        }

        if constexpr (beta == zAxis)
        {
            return (3 * cx<i>() * cx<i>()) == 0;
        }
    }

    if constexpr (alpha == yAxis)
    {
        if constexpr (beta == yAxis)
        {
            return ((3 * cy<i>() * cy<i>()) - 1) == 0;
        }

        if constexpr (beta == zAxis)
        {
            return (3 * cy<i>() * cz<i>()) == 0;
        }
    }

    if constexpr (alpha == zAxis)
    {
        if constexpr (beta == zAxis)
        {
            return ((3 * cz<i>() * cz<i>()) - 1) == 0;
        }
    }
}

template <const std::size_t alpha, const std::size_t beta, const std::size_t i>
__device__ [[nodiscard]] inline consteval real_t H2() noexcept
{
    if constexpr (alpha == xAxis)
    {
        if constexpr (beta == xAxis)
        {
            return static_cast<real_t>(cx<i>() * cx<i>()) - cs2;
        }

        if constexpr (beta == yAxis)
        {
            return static_cast<real_t>(cx<i>() * cy<i>());
        }

        if constexpr (beta == zAxis)
        {
            return static_cast<real_t>(cx<i>() * cx<i>());
        }
    }

    if constexpr (alpha == yAxis)
    {
        if constexpr (beta == yAxis)
        {
            return static_cast<real_t>(cy<i>() * cy<i>()) - cs2;
        }

        if constexpr (beta == zAxis)
        {
            return static_cast<real_t>(cy<i>() * cz<i>());
        }
    }

    if constexpr (alpha == zAxis)
    {
        if constexpr (beta == zAxis)
        {
            return static_cast<real_t>(cz<i>() * cz<i>()) - cs2;
        }
    }
}

__device__ __forceinline__ void Mfields_calculation(
    const real_t *fr, real_t *rhor,
    const real_t *fb, real_t *rhob,
    real_t *ux, real_t *uy, real_t *uz,
    real_t *Pixx, real_t *Pixy, real_t *Piyy,
    real_t *Piyz, real_t *Pizz, real_t *Pixz,
    const std::size_t x, const std::size_t y, const std::size_t z)
{

    const std::size_t id = idx(x, y, z);

    real_t sumr = real_t(0.0);
    real_t sumb = real_t(0.0);

    real_t jx = real_t(0.0);
    real_t jy = real_t(0.0);
    real_t jz = real_t(0.0);

    device::constexpr_for<0, Q>(
        [&](const auto i)
        {
            const real_t fr_i = fr[fidx(id, i)];
            const real_t fb_i = fb[fidx(id, i)];

            const real_t gi = fr_i + fb_i;

            sumr += fr_i;
            sumb += fb_i;

            if constexpr (!(cx<i>() == 0))
            {
                jx += process_element<cx<i>()>(gi);
            }
            if constexpr (!(cy<i>() == 0))
            {
                jy += process_element<cy<i>()>(gi);
            }
            if constexpr (!(cz<i>() == 0))
            {
                jz += process_element<cz<i>()>(gi);
            }
        });

    // #pragma unroll 27
    //     for (int i = 0; i < Q; ++i)
    //     {

    //         const real_t fr_i = fr[fidx(id, i)];
    //         const real_t fb_i = fb[fidx(id, i)];

    //         const real_t gi = fr_i + fb_i;

    //         const real_t cx = static_cast<real_t>(d_cx[i]);
    //         const real_t cy = static_cast<real_t>(d_cy[i]);
    //         const real_t cz = static_cast<real_t>(d_cz[i]);

    //         sumr += fr_i;
    //         sumb += fb_i;

    //         jx += gi * cx;
    //         jy += gi * cy;
    //         jz += gi * cz;
    //     }

    rhor[id] = sumr;
    rhob[id] = sumb;

    const real_t rhogi = sumr + sumb;

    const real_t vx = (jx) / rhogi;
    const real_t vy = (jy) / rhogi;
    const real_t vz = (jz) / rhogi;

    ux[id] = vx;
    uy[id] = vy;
    uz[id] = vz;

    real_t pixx = real_t(0.0);
    real_t pixy = real_t(0.0);
    real_t piyy = real_t(0.0);
    real_t piyz = real_t(0.0);
    real_t pizz = real_t(0.0);
    real_t pixz = real_t(0.0);

    device::constexpr_for<0, Q>(
        [&](const auto i)
        {
            const std::size_t idx = fidx(id, i);

            const real_t fir = fr[idx];
            const real_t fib = fb[idx];

            const real_t gieq = feq<i>(rhogi, vx, vy, vz);

            const real_t gi_neq = (fir + fib) - gieq;

            if constexpr (!H2_returns_zero<xAxis, xAxis, i>())
            {
                pixx += gi_neq * H2<xAxis, xAxis, i>();
            }

            if constexpr (!H2_returns_zero<xAxis, yAxis, i>())
            {
                pixy += gi_neq * H2<xAxis, yAxis, i>();
            }

            if constexpr (!H2_returns_zero<yAxis, yAxis, i>())
            {
                piyy += gi_neq * H2<yAxis, yAxis, i>();
            }

            if constexpr (!H2_returns_zero<yAxis, zAxis, i>())
            {
                piyz += gi_neq * H2<yAxis, zAxis, i>();
            }
            if constexpr (!H2_returns_zero<zAxis, zAxis, i>())
            {
                pizz += gi_neq * H2<zAxis, zAxis, i>();
            }
            if constexpr (!H2_returns_zero<xAxis, zAxis, i>())
            {
                pixz += gi_neq * H2<xAxis, zAxis, i>();
            }
        });

    // #pragma unroll 27
    //     for (int i = 0; i < Q; ++i)
    //     {
    //         const real_t fir = fr[fidx(id, i)];
    //         const real_t fib = fb[fidx(id, i)];

    //         const real_t gieq = feq(i, rhogi, vx, vy, vz);

    //         const real_t gi_neq = (fir + fib) - gieq;

    //         real_t Hxx, Hxy, Hyy, Hyz, Hzz, Hxz;
    //         H2(i, Hxx, Hxy, Hyy, Hyz, Hzz, Hxz);

    //         pixx += gi_neq * Hxx;
    //         pixy += gi_neq * Hxy;
    //         piyy += gi_neq * Hyy;
    //         piyz += gi_neq * Hyz;
    //         pizz += gi_neq * Hzz;
    //         pixz += gi_neq * Hxz;
    //     }

    Pixx[id] = pixx;
    Pixy[id] = pixy;
    Piyy[id] = piyy;
    Piyz[id] = piyz;
    Pizz[id] = pizz;
    Pixz[id] = pixz;
}

#endif