#include <cstdio>
#include <iostream>

#include "utilities/cudaUtilities.cuh"
#include "utilities/cudaConfig.cuh"

#include "constants.cuh"
#include "stencil.cuh"
#include "memory.cuh"
#include "launch.cuh"
#include "io/save_data.cuh"

#include "nanchechk.cuh"

int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    stencil();

    CudaConfig cfg = make_cuda_config();
    std::cout << "block=(" << cfg.block.x << "," << cfg.block.y << "," << cfg.block.z << ")\n";
    std::cout << "grid =(" << cfg.grid.x << "," << cfg.grid.y << "," << cfg.grid.z << ")\n";

    LbmDevice d = allocate_device_memory();
    LbmHost h = allocate_host_memory();

    launch_bubble(cfg, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int step = 0; step < NSTEP; ++step)
    {

        launch_density(cfg, d);
        launch_velocity(cfg, d);

        launch_pineq(cfg, d);

        /*         check_nan_D2H(d.rhor, (int)Ncells, "rhor");
                check_nan_D2H(d.rhob, (int)Ncells, "rhob");
                check_nan_D2H(d.fir, (int)fSize, "fir");
                check_nan_D2H(d.fib, (int)fSize, "fib"); */

        launch_collistream(cfg, d);

        /*         check_nan_D2H(d.rhor, (int)Ncells, "rhor");
                check_nan_D2H(d.rhob, (int)Ncells, "rhob");
                check_nan_D2H(d.fir, (int)fSize, "fir");
                check_nan_D2H(d.fib, (int)fSize, "fib"); */

        if (step % 10 == 0)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << "step " << step << "/" << NSTEP << "\n";
        }

        // output
        if (step % NOUTPUT == 0)
        {
            CUDA_CHECK(cudaDeviceSynchronize());

            write_vtk_step_device(step, d, h);
        }
    }

    free_device_memory(d);
    free_host_memory(h);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}