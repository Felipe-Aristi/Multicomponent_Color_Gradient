#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <array>
#include "utilities/types.cuh"

// Steps
inline constexpr int NSTEP = 200000;
inline constexpr int NOUTPUT = 2000;
inline constexpr bool write_vti_output = true;

// Grid
inline constexpr label_t NX = static_cast<label_t>(128);
inline constexpr label_t NZ = static_cast<label_t>(128);
inline constexpr label_t NY = static_cast<label_t>(400);
inline constexpr label_t sponge_cells = static_cast<label_t>(34);

inline constexpr label_t Ncells = NX * NY * NZ;
inline constexpr label_t NR_BINS = NX / static_cast<label_t>(2);
inline constexpr label_t NradialProfileCells = NY * NR_BINS;

// VELOCITY SET D2Q27 definition
inline constexpr label_t Q = 27;

// Memory sizes
inline constexpr std::size_t bytesCell = std::size_t(Ncells) * sizeof(real_t);
inline constexpr std::size_t fSize = std::size_t(Ncells) * (std::size_t)Q;
inline constexpr std::size_t bytesF = fSize * sizeof(pop_t);

// Jet parameters
inline constexpr real_t jet_radius = static_cast<real_t>(8.0);
inline constexpr real_t jet_x0 = static_cast<real_t>(NX - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_z0 = static_cast<real_t>(NZ - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_velocity = static_cast<real_t>(0.05);

// Bubble parameters
inline constexpr real_t bubble_radius = static_cast<real_t>(10.0);
inline constexpr real_t bubble_x0 = static_cast<real_t>(NX - 1) / static_cast<real_t>(2);
inline constexpr real_t bubble_y0 = static_cast<real_t>(NY - 1) / static_cast<real_t>(2);
inline constexpr real_t bubble_z0 = static_cast<real_t>(NZ - 1) / static_cast<real_t>(2);
inline constexpr real_t pi = static_cast<real_t>(3.141592653589793);

// Some useful constants
inline constexpr real_t cs2 = static_cast<real_t>(1.0 / 3.0);
inline constexpr real_t cs4 = cs2 * cs2;
inline constexpr real_t cs6 = cs4 * cs2;

inline constexpr real_t inv_cs2 = static_cast<real_t>(1) / cs2;
inline constexpr real_t inv_2cs2 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs2);
inline constexpr real_t inv_2cs4 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs4);
inline constexpr real_t inv_6cs6 = static_cast<real_t>(1.0) / (static_cast<real_t>(6.0) * cs6);
inline constexpr real_t inv_2cs6 = static_cast<real_t>(1.0) / (static_cast<real_t>(2.0) * cs6);

// FLuid parameters
inline constexpr real_t rhor0 = static_cast<real_t>(1);
inline constexpr real_t rhob0 = static_cast<real_t>(0.95);

inline constexpr real_t Re = static_cast<real_t>(5000);
inline constexpr real_t nu = (static_cast<real_t>(2) * jet_radius * jet_velocity) / Re;

inline constexpr real_t taur = static_cast<real_t>(0.5) + nu / (cs2); // static_cast<real_t>(0.6)

inline constexpr real_t taub = static_cast<real_t>(0.5) + nu / (cs2);

inline constexpr real_t omegar = static_cast<real_t>(1) / taur;
inline constexpr real_t omegab = static_cast<real_t>(1) / taub;

// Weber number
// inline constexpr std::size_t WeberInteger = WEBER;
inline constexpr real_t We = static_cast<real_t>(2500);

inline constexpr real_t sigma = static_cast<real_t>((rhob0 * jet_velocity * jet_velocity * static_cast<real_t>(2.0) * jet_radius) / We);
inline constexpr real_t beta_recolor = static_cast<real_t>(0.80);

inline constexpr real_t sigma_over_4cs4 = sigma / (static_cast<real_t>(4) * cs4);

#endif
