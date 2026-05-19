#ifndef MEANFIELDSRT_CUH
#define MEANFIELDSRT_CUH

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include "../utilities/cudaUtilities.cuh"
#include "../constants.cuh"
#include "memoryMeanFields.cuh"

struct MeanFieldsFiles
{
    FILE *tke_total = nullptr;
    FILE *tke_avg = nullptr;
    FILE *tke_diff = nullptr;
};

struct MeanFieldsRuntime
{
    MeanFieldsDevice device{};
    MeanFieldsHost host{};
    MeanFieldsState state{};
    MeanFieldsFiles files{};

    std::string case_dir{};
    std::string mean_dir{};
    std::string vti_dir{};
};

// ------------- Create directory ---------------
inline std::string make_case_folder_name()
{
    std::ostringstream oss;
    oss << "./JET_VTK/"
        << "Re" << static_cast<int>(std::round(Re))
        << "_We" << static_cast<int>(std::round(We));
    return oss.str();
}

inline bool ensure_case_directories(MeanFieldsRuntime &mf)
{
    mf.case_dir = make_case_folder_name();
    mf.mean_dir = mf.case_dir + "/mean_profiles";
    mf.vti_dir = mf.case_dir + "/vti_data";

    std::error_code ec;

    std::filesystem::create_directories(mf.mean_dir, ec);
    if (ec)
    {
        std::fprintf(stderr, "Failed to create mean_profiles directory: %s\n", mf.mean_dir.c_str());
        return false;
    }

    ec.clear();
    std::filesystem::create_directories(mf.vti_dir, ec);
    if (ec)
    {
        std::fprintf(stderr, "Failed to create vti_data directory: %s\n", mf.vti_dir.c_str());
        return false;
    }

    return true;
}

// ---------------------------------------------------

inline bool open_mean_fields_files(MeanFieldsFiles &f, const std::string &mean_dir)
{
    const std::string path_total = mean_dir + "/tke_total.bin";
    const std::string path_avg = mean_dir + "/tke_avg.bin";
    const std::string path_diff = mean_dir + "/tke_diff.bin";

    f.tke_total = std::fopen(path_total.c_str(), "wb");
    f.tke_avg = std::fopen(path_avg.c_str(), "wb");
    f.tke_diff = std::fopen(path_diff.c_str(), "wb");

    if (f.tke_total == nullptr || f.tke_avg == nullptr ||
        f.tke_diff == nullptr)
    {
        std::fprintf(stderr, "Error opening one or more files in: %s\n", mean_dir.c_str());
        return false;
    }

    return true;
}

inline void write_case_metadata(const MeanFieldsRuntime &mf)
{
    const std::string metadata_path = mf.case_dir + "/case_info.txt";
    FILE *file = std::fopen(metadata_path.c_str(), "w");
    if (file == nullptr)
    {
        std::fprintf(stderr, "Warning: could not write case metadata: %s\n", metadata_path.c_str());
        return;
    }

    std::fprintf(file, "Re %d\n", static_cast<int>(std::round(Re)));
    std::fprintf(file, "We %d\n", static_cast<int>(std::round(We)));
    std::fprintf(file, "NX %u\n", static_cast<unsigned int>(NX));
    std::fprintf(file, "NY %u\n", static_cast<unsigned int>(NY));
    std::fprintf(file, "NZ %u\n", static_cast<unsigned int>(NZ));
    std::fprintf(file, "NSTEP %d\n", NSTEP);
    std::fprintf(file, "NOUTPUT %d\n", NOUTPUT);
    std::fprintf(file, "NSTATS_START_STEP %d\n", NSTATS_START_STEP);
    std::fprintf(file, "NR_BINS %u\n", static_cast<unsigned int>(NR_BINS));
    std::fprintf(file, "write_vti_output %d\n", write_vti_output ? 1 : 0);
    std::fprintf(file, "jet_radius %.9g\n", static_cast<double>(jet_radius));
    std::fprintf(file, "jet_diameter %.9g\n", static_cast<double>(static_cast<real_t>(2.0) * jet_radius));
    std::fprintf(file, "jet_velocity %.9g\n", static_cast<double>(jet_velocity));

    std::fclose(file);
}

inline void close_mean_fields_files(MeanFieldsFiles &f)
{
    if (f.tke_total != nullptr)
        std::fclose(f.tke_total);
    if (f.tke_avg != nullptr)
        std::fclose(f.tke_avg);
    if (f.tke_diff != nullptr)
        std::fclose(f.tke_diff);

    f = MeanFieldsFiles{};
}

inline bool initialize_mean_fields_runtime(const int deviceID,
                                           MeanFieldsRuntime &mf)
{
    CUDA_CHECK(cudaSetDevice(deviceID));

    mf.device = allocate_mean_fields_device();
    mf.host = allocate_mean_fields_host();
    mf.state = MeanFieldsState{};

    if (!ensure_case_directories(mf))
    {
        free_mean_fields_device(mf.device);
        free_mean_fields_host(mf.host);
        return false;
    }

    if (!open_mean_fields_files(mf.files, mf.mean_dir))
    {
        std::fprintf(stderr, "Error opening one or more mean-field output files.\n");

        close_mean_fields_files(mf.files);
        free_mean_fields_device(mf.device);
        free_mean_fields_host(mf.host);

        return false;
    }

    write_case_metadata(mf);

    return true;
}

inline void free_mean_fields_runtime(MeanFieldsRuntime &mf)
{
    close_mean_fields_files(mf.files);
    free_mean_fields_device(mf.device);
    free_mean_fields_host(mf.host);

    mf = MeanFieldsRuntime{};
}

inline void update_radial_statistics_start(MeanFieldsRuntime &mf, const int step)
{
    if (mf.state.start_uy_average || step < NSTATS_START_STEP)
    {
        return;
    }

    mf.state.start_uy_average = true;
    mf.state.step_uy_avg_start = static_cast<unsigned int>(step);
    mf.state.uy_avg_samples = 0;

    std::cout << "Starting radial temporal statistics at step "
              << mf.state.step_uy_avg_start
              << " | stats_start_step = " << NSTATS_START_STEP
              << " | last_abs_delta_tke_avg = " << mf.host.abs_delta
              << "\n";
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
    }

    std::cout << std::scientific
              << "step " << step
              << " | tke_prev = " << mf.host.tke_avg_prev
              << " | tke_avg = " << mf.host.tke_avg
              << " | delta_tke_avg = " << mf.host.abs_delta;

    if (mf.state.start_uy_average)
    {
        std::cout << " | uy_radial_avg = on"
                  << " | uy_avg_start_step = " << mf.state.step_uy_avg_start
                  << " | uyavg_samples = " << mf.state.uy_avg_samples
                  << " | radial_count = on";
    }
    else
    {
        std::cout << " | uy_radial_avg = waiting"
                  << " | uyavg_samples = 0";
    }

    std::cout << "\n";

    mf.host.tke_avg_prev = mf.host.tke_avg;
}

inline void write_radial_profile_metadata(const MeanFieldsRuntime &mf)
{
    const std::string path = mf.mean_dir + "/radial_profile_metadata.txt";
    FILE *file = std::fopen(path.c_str(), "w");
    if (file == nullptr)
    {
        std::fprintf(stderr, "Warning: could not write radial profile metadata: %s\n", path.c_str());
        return;
    }

    std::fprintf(file, "NX %u\n", static_cast<unsigned int>(NX));
    std::fprintf(file, "NY %u\n", static_cast<unsigned int>(NY));
    std::fprintf(file, "NZ %u\n", static_cast<unsigned int>(NZ));
    std::fprintf(file, "NR_BINS %u\n", static_cast<unsigned int>(NR_BINS));
    std::fprintf(file, "NradialProfileCells %u\n", static_cast<unsigned int>(NradialProfileCells));
    std::fprintf(file, "NOUTPUT %d\n", NOUTPUT);
    std::fprintf(file, "NPROFILE_OUTPUT %d\n", NOUTPUT);
    std::fprintf(file, "NSTATS_SAMPLE %d\n", 1);
    std::fprintf(file, "NSTATS_START_STEP %d\n", NSTATS_START_STEP);
    std::fprintf(file, "Re %d\n", static_cast<int>(std::round(Re)));
    std::fprintf(file, "We %d\n", static_cast<int>(std::round(We)));
    std::fprintf(file, "D %.9g\n", static_cast<double>(static_cast<real_t>(2.0) * jet_radius));
    std::fprintf(file, "U_jet %.9g\n", static_cast<double>(jet_velocity));
    std::fprintf(file, "jet_radius %.9g\n", static_cast<double>(jet_radius));
    std::fprintf(file, "jet_velocity %.9g\n", static_cast<double>(jet_velocity));
    std::fprintf(file, "uy_avg_started %d\n", mf.state.start_uy_average ? 1 : 0);
    std::fprintf(file, "start_step %u\n", mf.state.step_uy_avg_start);
    std::fprintf(file, "uy_avg_start_step %u\n", mf.state.step_uy_avg_start);
    std::fprintf(file, "uy_avg_samples %u\n", mf.state.uy_avg_samples);
    std::fprintf(file, "radial_sample_count %u\n", mf.state.uy_avg_samples);

    if (mf.state.start_uy_average && mf.state.uy_avg_samples > 0)
    {
        const unsigned int first_sample_step = mf.state.step_uy_avg_start;
        const unsigned int last_sample_step = first_sample_step + mf.state.uy_avg_samples - 1;

        std::fprintf(file, "uy_avg_first_sample_step %u\n", first_sample_step);
        std::fprintf(file, "uy_avg_last_sample_step %u\n", last_sample_step);
    }

    std::fclose(file);
}

inline bool write_binary_file(const std::string &path,
                              const void *data,
                              const std::size_t element_size,
                              const std::size_t element_count)
{
    FILE *file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
    {
        std::fprintf(stderr, "Warning: could not write binary file: %s\n", path.c_str());
        return false;
    }

    const std::size_t written = std::fwrite(data, element_size, element_count, file);
    std::fclose(file);

    if (written != element_count)
    {
        std::fprintf(stderr, "Warning: partial write for %s (%zu/%zu)\n",
                     path.c_str(), written, element_count);
        return false;
    }

    return true;
}

inline bool write_radial_profile_outputs(MeanFieldsRuntime &mf)
{
    if (!mf.state.start_uy_average || mf.state.uy_avg_samples == 0)
    {
        write_radial_profile_metadata(mf);
        return false;
    }

    CUDA_CHECK(cudaMemcpy(mf.host.radial_sum_uy,
                          mf.device.radial_sum_uy,
                          radial_stat_bytes(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.radial_sum_uy2,
                          mf.device.radial_sum_uy2,
                          radial_stat_bytes(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.radial_sum_ur,
                          mf.device.radial_sum_ur,
                          radial_stat_bytes(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.radial_sum_ur2,
                          mf.device.radial_sum_ur2,
                          radial_stat_bytes(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.radial_sum_uruy,
                          mf.device.radial_sum_uruy,
                          radial_stat_bytes(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.radial_count,
                          mf.device.radial_count,
                          radial_count_bytes(),
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    ok = write_binary_file(mf.mean_dir + "/radial_sum_uy.bin",
                           mf.host.radial_sum_uy,
                           sizeof(profile_stat_t),
                           static_cast<std::size_t>(NradialProfileCells)) &&
         ok;
    ok = write_binary_file(mf.mean_dir + "/radial_sum_uy2.bin",
                           mf.host.radial_sum_uy2,
                           sizeof(profile_stat_t),
                           static_cast<std::size_t>(NradialProfileCells)) &&
         ok;
    ok = write_binary_file(mf.mean_dir + "/radial_sum_ur.bin",
                           mf.host.radial_sum_ur,
                           sizeof(profile_stat_t),
                           static_cast<std::size_t>(NradialProfileCells)) &&
         ok;
    ok = write_binary_file(mf.mean_dir + "/radial_sum_ur2.bin",
                           mf.host.radial_sum_ur2,
                           sizeof(profile_stat_t),
                           static_cast<std::size_t>(NradialProfileCells)) &&
         ok;
    ok = write_binary_file(mf.mean_dir + "/radial_sum_uruy.bin",
                           mf.host.radial_sum_uruy,
                           sizeof(profile_stat_t),
                           static_cast<std::size_t>(NradialProfileCells)) &&
         ok;
    ok = write_binary_file(mf.mean_dir + "/radial_count.bin",
                           mf.host.radial_count,
                           sizeof(profile_count_t),
                           static_cast<std::size_t>(NradialProfileCells)) &&
         ok;

    write_radial_profile_metadata(mf);
    return ok;
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
