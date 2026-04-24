#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "../utilities/bounds.cuh"
#include "../utilities/mathUtilities.cuh"
#include "../utilities/indexing.cuh"
#include "../utilities/types.cuh"
#include "../constants.cuh"
#include "../stencil_ct.cuh"
#include "../lbm.cuh"

// Jet initialization
__device__ void init_density_jet(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                                 pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                                 const label_t x, const label_t y, const label_t z)
{
    const label_t id = idx(x, y, z);

    rhor[id] = rhor0;
    rhob[id] = static_cast<real_t>(0.0);

    real_t vx = static_cast<real_t>(0.0);
    real_t vy = static_cast<real_t>(0.0);
    real_t vz = static_cast<real_t>(0.0);

    constexpr_for<0, Q>([&] __device__(auto I)
                        {
                            constexpr label_t i = decltype(I)::value;

                            real_t fir = feq<i>(rhor[id], vx, vy, vz);
                            real_t fib = feq<i>(rhob[id], vx, vy, vz);

                            fr[fidx(id, i)] = save_pop(fir);
                            fb[fidx(id, i)] = save_pop(fib); });
}

__device__ void jet_mask(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                         pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                         const label_t x, const label_t z)
{

    const label_t yB = static_cast<label_t>(0);
    const label_t id = idx(x, yB, z);
    const label_t is_jet = isJet(x, z);

    rhor[id] = (real_t(1.0) - static_cast<real_t>(is_jet)) * rhor0;
    rhob[id] = static_cast<real_t>(is_jet) * rhob0;

    real_t vx = static_cast<real_t>(0.0);
    real_t vy = static_cast<real_t>(0.0);
    real_t vz = static_cast<real_t>(0.0);

    constexpr_for<0, Q>([&] __device__(auto I)
                        {
        constexpr label_t i = decltype(I)::value;

        real_t fir = feq<i>(rhor[id], vx, vy, vz);
        real_t fib = feq<i>(rhob[id], vx, vy, vz);

        fr[fidx(id, i)] = save_pop(fir);
        fb[fidx(id, i)] = save_pop(fib); });
}

#endif