#ifndef COUNTMLUPS_CUH
#define COUNTMLUPS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#include "../constants.cuh"
#include "cudaUtilities.cuh"

// Mlups calculation
struct MlupsStats
{
    real_t window_ms = static_cast<real_t>(0.0);
    real_t total_ms = static_cast<real_t>(0.0);
    real_t mlups_window = static_cast<real_t>(0.0);
    real_t mlups_avg = static_cast<real_t>(0.0);
};

inline bool update_mlups_stats(cudaEvent_t evStart,
                               cudaEvent_t evStop,
                               const int step,
                               MlupsStats &stats,
                               const int steps_per_window = 10)
{
    if ((step + 1) % steps_per_window != 0)
    {
        return false;
    }

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    stats.window_ms = static_cast<real_t>(0.0);
    CUDA_CHECK(cudaEventElapsedTime(&stats.window_ms, evStart, evStop));

    stats.total_ms += stats.window_ms;

    const real_t steps_window = static_cast<real_t>(steps_per_window);
    const real_t steps_done = static_cast<real_t>(step + 1);

    stats.mlups_window =
        (steps_window * static_cast<real_t>(Ncells)) /
        (stats.window_ms * static_cast<real_t>(1.0e3));

    stats.mlups_avg =
        (steps_done * static_cast<real_t>(Ncells)) /
        (stats.total_ms * static_cast<real_t>(1.0e3));

    CUDA_CHECK(cudaEventRecord(evStart));

    return true;
}

inline void print_mlups_stats(const MlupsStats &stats,
                              const int step)
{
    std::cout << std::fixed << std::setprecision(3)
              << "step " << (step + 1) << "/" << NSTEP
              << " | dt_window = " << stats.window_ms << " ms"
              << " | MLUPS = " << stats.mlups_window
              << " | MLUPS(avg) = " << stats.mlups_avg
              << "\n";
}

#endif
