#include <iostream>

#include "utilities/cudaUtilities.cuh"
#include "utilities/cudaConfig.cuh"
#include "utilities/countmlups.cuh"

#include "meanF/meanFieldsRT.cuh"

#include "constants.cuh"
#include "stencil.cuh"
#include "memory.cuh"
#include "launch.cuh"
#include "io/save_data.cuh"

constexpr int deviceID = 0;

int main()
{
    MeanFieldsRuntime mf{};

    if (!initialize_mean_fields_runtime(deviceID, mf))
    {
        CUDA_CHECK(cudaDeviceReset());
        return EXIT_FAILURE;
    }

    CudaConfig cfg = print_device_and_make_config(deviceID);

    LbmDevice d = allocate_device_memory();
    LbmHost h = allocate_host_memory();

    launch_jetDensity(cfg, d);
    launch_Injet(cfg, d);
    // launch_bubble(cfg, d);

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    MlupsStats perf{};
    CUDA_CHECK(cudaEventRecord(evStart));

    for (int step = 0; step < NSTEP; ++step)
    {
        launch_Macros(cfg, d);

        if (mf.state.start_uy_average)
        {
            launch_accumulate_radial_moments(cfg, d, mf.device);
            ++mf.state.uy_avg_samples;
        }

        launch_collistream(cfg, d);
        launch_inlet_bc(cfg, d);
        launch_neumann_bc(cfg, d);

        if (update_mlups_stats(evStart, evStop, step, perf, 100))
        {
            print_mlups_stats(perf, step);
        }

        if (step % NOUTPUT == 0)
        {
            CUDA_CHECK(cudaMemset(mf.device.tke_total, 0, sizeof(real_t)));

            launch_compute_total_tke(cfg, d, mf.device.tke_total);
            launch_update_tke_average(mf.device.tke_avg, mf.device.tke_total, step, 0);

            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&mf.host.tke_total,
                                  mf.device.tke_total,
                                  sizeof(real_t),
                                  cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(&mf.host.tke_avg,
                                  mf.device.tke_avg,
                                  sizeof(real_t),
                                  cudaMemcpyDeviceToHost));

            process_tke_sample(mf, step);

            if (mf.state.start_uy_average && mf.state.uy_avg_samples > 0)
            {
                write_radial_profile_outputs(mf);
            }

            if constexpr (write_vti_output)
            {
                write_vti_step_device(step, d, h);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    write_radial_profile_outputs(mf);

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    free_mean_fields_runtime(mf);

    free_device_memory(d);
    free_host_memory(h);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
