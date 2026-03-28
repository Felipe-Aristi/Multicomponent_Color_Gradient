#ifndef MEANFIELDSRT_CUH
#define MEANFIELDSRT_CUH

#include <cstdio>
#include <cstdlib>
#include <string>

#include "../utilities/cudaUtilities.cuh"
#include "memoryMeanFields.cuh"

struct MeanFieldsFiles
{
    FILE *tke_total = nullptr;
    FILE *tke_avg = nullptr;
    FILE *tke_diff = nullptr;
    FILE *uy_avg = nullptr;
};

struct MeanFieldsRuntime
{
    MeanFieldsDevice device{};
    MeanFieldsHost host{};
    MeanFieldsState state{};
    MeanFieldsFiles files{};
};

inline bool open_mean_fields_files(MeanFieldsFiles &f)
{
    const std::string path_uy_avg = "./JET_VTK/uy_avg.bin";
    const std::string path_total = "./JET_VTK/tke_total.bin";
    const std::string path_avg = "./JET_VTK/tke_avg.bin";
    const std::string path_diff = "./JET_VTK/tke_diff.bin";

    f.tke_total = std::fopen(path_total.c_str(), "wb");
    f.tke_avg = std::fopen(path_avg.c_str(), "wb");
    f.tke_diff = std::fopen(path_diff.c_str(), "wb");
    f.uy_avg = std::fopen(path_uy_avg.c_str(), "wb");

    if (f.tke_total == nullptr || f.tke_avg == nullptr ||
        f.tke_diff == nullptr || f.uy_avg == nullptr)
    {
        return false;
    }

    return true;
}

inline void close_mean_fields_files(MeanFieldsFiles &f)
{
    if (f.tke_total != nullptr)
        std::fclose(f.tke_total);
    if (f.tke_avg != nullptr)
        std::fclose(f.tke_avg);
    if (f.tke_diff != nullptr)
        std::fclose(f.tke_diff);
    if (f.uy_avg != nullptr)
        std::fclose(f.uy_avg);

    f = MeanFieldsFiles{};
}

inline bool initialize_mean_fields_runtime(const int deviceID,
                                           MeanFieldsRuntime &mf)
{
    CUDA_CHECK(cudaSetDevice(deviceID));

    mf.device = allocate_mean_fields_device();
    mf.host = allocate_mean_fields_host();
    mf.state = MeanFieldsState{};

    if (!open_mean_fields_files(mf.files))
    {
        std::fprintf(stderr, "Error opening one or more mean-field output files.\n");

        close_mean_fields_files(mf.files);
        free_mean_fields_device(mf.device);
        free_mean_fields_host(mf.host);

        return false;
    }

    return true;
}

inline void free_mean_fields_runtime(MeanFieldsRuntime &mf)
{
    close_mean_fields_files(mf.files);
    free_mean_fields_device(mf.device);
    free_mean_fields_host(mf.host);

    mf = MeanFieldsRuntime{};
}

inline void update_tke_state(MeanFieldsRuntime &mf, const int step)
{
    if (mf.state.first_tke_sample)
    {
        mf.host.delta = static_cast<real_t>(0);
        mf.host.abs_delta = static_cast<real_t>(0);
        mf.state.first_tke_sample = false;
    }
    else
    {
        mf.host.delta = mf.host.tke_avg - mf.host.tke_avg_prev;
        mf.host.abs_delta = std::abs(mf.host.delta);

        if (!mf.state.start_uy_average &&
            mf.host.abs_delta < MeanFieldsState::tke_tolerance)
        {
            mf.state.start_uy_average = true;
            mf.state.step_uy_avg_start = static_cast<unsigned int>(step);

            std::cout << "Starting temporal average of uy at step "
                      << mf.state.step_uy_avg_start
                      << " | abs_delta_tke_avg = " << mf.host.abs_delta
                      << "\n";
        }
    }

    std::cout << std::scientific
              << "step " << step
              << " | tke_prev = " << mf.host.tke_avg_prev
              << " | tke_avg = " << mf.host.tke_avg
              << " | delta_tke_avg = " << mf.host.abs_delta
              << "\n";

    mf.host.tke_avg_prev = mf.host.tke_avg;
}

inline void write_tke_outputs(MeanFieldsRuntime &mf)
{
    std::fwrite(&mf.host.tke_total, sizeof(real_t), 1, mf.files.tke_total);
    std::fwrite(&mf.host.tke_avg, sizeof(real_t), 1, mf.files.tke_avg);
    std::fwrite(&mf.host.abs_delta, sizeof(real_t), 1, mf.files.tke_diff);

    std::fflush(mf.files.tke_total);
    std::fflush(mf.files.tke_avg);
    std::fflush(mf.files.tke_diff);
}

inline void process_tke_sample(MeanFieldsRuntime &mf, const int step)
{
    update_tke_state(mf, step);
    write_tke_outputs(mf);
}

#endif