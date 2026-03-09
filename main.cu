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

    stencil();

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
    double total_ms = 0.0;

    for (int step = 0; step < NSTEP; ++step)
    {

        CUDA_CHECK(cudaEventRecord(evStart));

        launch_Macros(cfg, d);

        launch_collistream(cfg, d);

        launch_inlet_bc(cfg, d);
        launch_neumann_bc(cfg, d);

        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));

        float step_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&step_ms, evStart, evStop));

        total_ms += step_ms;

        const double mlups_step = static_cast<double>(Ncells) / (static_cast<double>(step_ms) * 1.0e3);
        const double mlups_avg = static_cast<double>((step + 1) * Ncells) / (total_ms * 1.0e3);

        if (step % 10 == 0)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << "step " << step << "/" << NSTEP
                      << " | dt = " << step_ms << " ms"
                      << " | MLUPS(step) = " << mlups_step
                      << " | MLUPS(avg) = " << mlups_avg
                      << "\n";
        }

        // output
        if (step % NOUTPUT == 0)
        {
            // CUDA_CHECK(cudaDeviceSynchronize());

            write_vtk_step_device(step, d, h);
        }
    }

    free_device_memory(d);
    free_host_memory(h);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}