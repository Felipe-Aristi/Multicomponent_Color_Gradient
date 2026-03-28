#include <cstdio>
#include <iostream>


#include "utilities/cudaUtilities.cuh"
#include "utilities/cudaConfig.cuh"

#include "constants.cuh"
#include "stencil.cuh"
#include "memory.cuh"
#include "launch.cuh"
#include "io/save_data.cuh"
                                                           

constexpr const int deviceID = 0;



int main()
{
    // ----------------------Kinetic energy inicialization ---------------------------
    real_t *d_tke_total;
    real_t *d_tke_avg;
    real_t h_tke_total = static_cast<real_t>(0);
    real_t h_tke_avg   = static_cast<real_t>(0); 
    real_t h_tke_avg_prev = static_cast<real_t>(0);
    // real_t h_tke_avg_diff = static_cast<real_t>(0);
    real_t delta = static_cast<real_t>(0);
    real_t abs_delta = static_cast<real_t>(0);


    bool first_tke_sample = true;

    CUDA_CHECK(cudaMalloc((void **)&d_tke_avg, sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_tke_total, sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d_tke_total, 0, sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d_tke_avg, 0, sizeof(real_t)));

    std::string path_total = std::string("./JET_VTK/") + "/tke_total.bin";
	std::string path_avg = std::string("./JET_VTK/")  + "/tke_avg.bin";
	std::string path_diff = std::string("./JET_VTK/")  + "/tke_diff.bin";

	FILE *f_tke_total = fopen(path_total.c_str(), "wb");
	FILE *f_tke_avg = fopen(path_avg.c_str(), "wb");
	FILE *f_tke_diff = fopen(path_diff.c_str(), "wb");

    // --------------------- End of kinetic energy inicialization -----------------------------

    CUDA_CHECK(cudaSetDevice(deviceID));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceID));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << std::endl;

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

            // std::cout << std::fixed << std::setprecision(3)
            //           << "step " << (step + 1) << "/" << NSTEP
            //           << " | dt_window = " << window_ms << " ms"
            //           << " | MLUPS = " << mlups_window
            //           << " | MLUPS(avg) = " << mlups_avg
            //           << "\n";

            CUDA_CHECK(cudaEventRecord(evStart));
        }

        // output
        if (step % NOUTPUT == 0)
        {
            CUDA_CHECK(cudaMemset(d_tke_total, 0, sizeof(real_t)));

            launch_compute_total_tke(cfg, d, d_tke_total);
            launch_update_tke_average(d_tke_avg, d_tke_total, step, 0);

            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&h_tke_total, d_tke_total, sizeof(real_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_tke_avg, d_tke_avg, sizeof(real_t), cudaMemcpyDeviceToHost));


            if (first_tke_sample)
            {

                first_tke_sample = false;
            }
            else
            {
                delta = h_tke_avg - h_tke_avg_prev;
                abs_delta = std::abs(delta);
            }
            std::cout << std::scientific
                << "step " << step
                << " | tke_prev = " << h_tke_avg_prev
                << " | tke_avg = " << h_tke_avg
                << " | delta_tke_avg = " << abs_delta
                << "\n";

            h_tke_avg_prev = h_tke_avg;

            size_t written_total = fwrite(&h_tke_total, sizeof(real_t), 1, f_tke_total);
            size_t written_avg = fwrite(&h_tke_avg, sizeof(real_t), 1, f_tke_avg);
            size_t written_diff = fwrite(&abs_delta, sizeof(real_t), 1, f_tke_diff);
            fflush(f_tke_total);
            fflush(f_tke_avg);
            fflush(f_tke_diff);

            // write_vti_step_device(step, d, h);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    fclose(f_tke_total);
    fclose(f_tke_avg);
    fclose(f_tke_diff);

    cudaFree(d_tke_total);
    cudaFree(d_tke_avg);

    free_device_memory(d);
    free_host_memory(h);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}