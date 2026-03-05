#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <array>
#include "utilities/types.cuh"

// Steps
inline constexpr int NSTEP = 5000;
inline constexpr int NOUTPUT = 100;

// Grid
inline constexpr std::size_t NX = 32;
inline constexpr std::size_t NZ = 32;
inline constexpr std::size_t NY = 32;
inline constexpr std::size_t Ncells = NX * NY * NZ;

// VELOCITY SET D2Q27 definition
inline constexpr std::size_t Q = 27;

// Memory sizes
inline constexpr std::size_t fSize = Ncells * (std::size_t)Q;

inline constexpr std::size_t bytesCell = Ncells * sizeof(real_t);
inline constexpr std::size_t bytesF = fSize * sizeof(real_t);

// Jet parameters
inline constexpr real_t jet_radius = 2.0;
inline constexpr real_t jet_x0 = NX / 2.0;
inline constexpr real_t jet_z0 = NZ / 2.0;
inline constexpr real_t jet_velocity = 0.0465;

// Bubble parameters
inline constexpr real_t bubble_radius = 5.0;
inline constexpr real_t bubble_x0 = NX / 2.0;
inline constexpr real_t bubble_y0 = NY / 2.0;
inline constexpr real_t bubble_z0 = NZ / 2.0;
inline constexpr real_t pi = static_cast<real_t>(3.141592653589793);

inline constexpr real_t cs2 = static_cast<real_t>(1.0 / 3.0);
inline constexpr real_t cs4 = cs2 * cs2;

inline constexpr real_t rho_r0 = static_cast<real_t>(1);
inline constexpr real_t rho_b0 = static_cast<real_t>(0.95);

inline constexpr real_t Re = static_cast<real_t>(1000);
inline constexpr real_t nu = static_cast<real_t>(2 * jet_radius * jet_velocity) / Re;

inline constexpr real_t taur = static_cast<real_t>(0.5) + nu / (cs2); // static_cast<real_t>(0.6)

inline constexpr real_t taub = static_cast<real_t>(0.5) + nu / (cs2);

inline constexpr real_t omegar = static_cast<real_t>(1) / taur;
inline constexpr real_t omegab = static_cast<real_t>(1) / taub;

inline constexpr real_t sigma = static_cast<real_t>(0.0000000010);
inline constexpr real_t beta_recolor = real_t(0.95);

#endif