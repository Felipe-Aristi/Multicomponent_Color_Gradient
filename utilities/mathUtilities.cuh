
#ifndef MATH_UTILITIES_CUH
#define MATH_UTILITIES_CUH

#include <cstddef>
#include <cmath>
#include <type_traits>
#include <utility>
#include "../constants.cuh"
#include "../stencil.cuh"
#include "types.cuh"

// jet shape inlet
__device__ [[nodiscard]] constexpr label_t isJet(label_t x, label_t z) noexcept
{
    const real_t dx = static_cast<real_t>(x) - jet_x0;
    const real_t dz = static_cast<real_t>(z) - jet_z0;

    return (dx * dx + dz * dz <= jet_radius * jet_radius) ? 1 : 0;
}

// Phase- Field
__device__ __forceinline__ real_t psi(const real_t rho_self, const real_t rho_other) noexcept
{
    return real_t(rho_self - rho_other) / real_t(rho_self + rho_other);
}

// Constexpr_for ---> for esquisito pra os amigos
template <typename T, T V>
using integralConstant = std::integral_constant<T, V>;

template <const label_t Start, const label_t End, typename F>
__device__ inline constexpr void constexpr_for(F &&f) noexcept
{
    if constexpr (Start < End)
    {
        f(integralConstant<label_t, Start>());
        if constexpr (Start + 1 < End)
        {
            constexpr_for<Start + 1, End>(std::forward<F>(f));
        }
    }
}

#endif
