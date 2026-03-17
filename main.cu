#include <cstdio>
#include <iostream>

#include "utilities/cudaUtilities.cuh"
#include "utilities/cudaConfig.cuh"

#include "constants.cuh"
#include "stencil.cuh"
#include "memory.cuh"
#include "launch.cuh"
#include "io/save_data.cuh"

int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    CudaConfig cfg = make_cudaConfig();
    std::cout << "block=(" << cfg.block.x << "," << cfg.block.y << "," << cfg.block.z << ")\n";
    std::cout << "grid =(" << cfg.grid.x << "," << cfg.grid.y << "," << cfg.grid.z << ")\n";

    LbmDevice d = allocate_device_memory();
    LbmHost h = allocate_host_memory();

    launch_jetDensity(cfg, d);
    launch_Injet(cfg, d);
    // launch_bubble(cfg, d);

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    real_t total_ms = static_cast<real_t>(0.0);
    CUDA_CHECK(cudaEventRecord(evStart));

    for (int step = 0; step < NSTEP; ++step)
    {

        launch_Macros(cfg, d);

        launch_collistream(cfg, d);

        launch_inlet_bc(cfg, d);
        launch_neumann_bc(cfg, d);

        if (step % 10 == 0)
        {
            CUDA_CHECK(cudaEventRecord(evStop));
            CUDA_CHECK(cudaEventSynchronize(evStop));

            real_t window_ms = static_cast<real_t>(0.0);
            CUDA_CHECK(cudaEventElapsedTime(&window_ms, evStart, evStop));

            total_ms += window_ms;

            const real_t steps_in_window = static_cast<real_t>(10.0);
            const real_t mlups_window =
                (steps_in_window * static_cast<real_t>(Ncells)) / (window_ms * static_cast<real_t>(1.0e3));

            const real_t mlups_avg =
                (static_cast<real_t>(step + 1) * static_cast<real_t>(Ncells)) / (total_ms * static_cast<real_t>(1.0e3));

            std::cout << std::fixed << std::setprecision(3)
                      << "step " << (step + 1) << "/" << NSTEP
                      << " | dt_window = " << window_ms << " ms"
                      << " | MLUPS = " << mlups_window
                      << " | MLUPS(avg) = " << mlups_avg
                      << "\n";

            CUDA_CHECK(cudaEventRecord(evStart));
        }

        // output
        if (step % NOUTPUT == 0)
        {
            // CUDA_CHECK(cudaDeviceSynchronize());

            write_vti_step_device(step, d, h);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    free_device_memory(d);
    free_host_memory(h);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}