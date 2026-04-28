#ifndef MEMORY_MEAN_FIELDS_CUH
#define MEMORY_MEAN_FIELDS_CUH

#include <cstdlib>
#include <cstdio>

#include "../constants.cuh"

#include "../utilities/cudaUtilities.cuh"
#include "../utilities/types.cuh"

struct MeanFieldsDevice
{
    real_t *tke_total = nullptr;
    real_t *tke_avg = nullptr;
    profile_stat_t *radial_sum_uy = nullptr;
    profile_stat_t *radial_sum_uy2 = nullptr;
    profile_stat_t *radial_sum_ur = nullptr;
    profile_stat_t *radial_sum_ur2 = nullptr;
    profile_stat_t *radial_sum_uruy = nullptr;
    profile_count_t *radial_count = nullptr;
};

struct MeanFieldsHost
{
    real_t tke_total = static_cast<real_t>(0);
    real_t tke_avg = static_cast<real_t>(0);
    real_t tke_avg_prev = static_cast<real_t>(0);
    real_t delta = static_cast<real_t>(0);
    real_t abs_delta = static_cast<real_t>(0);
    profile_stat_t *radial_sum_uy = nullptr;
    profile_stat_t *radial_sum_uy2 = nullptr;
    profile_stat_t *radial_sum_ur = nullptr;
    profile_stat_t *radial_sum_ur2 = nullptr;
    profile_stat_t *radial_sum_uruy = nullptr;
    profile_count_t *radial_count = nullptr;
};

struct MeanFieldsState
{
    bool start_uy_average = false;
    bool first_tke_sample = true;
    unsigned int step_uy_avg_start = 0;
    unsigned int uy_avg_samples = 0;
    static constexpr real_t tke_tolerance = static_cast<real_t>(1e-1);
};

inline constexpr std::size_t radial_stat_bytes()
{
    return static_cast<std::size_t>(NradialProfileCells) * sizeof(profile_stat_t);
}

inline constexpr std::size_t radial_count_bytes()
{
    return static_cast<std::size_t>(NradialProfileCells) * sizeof(profile_count_t);
}

inline MeanFieldsDevice allocate_mean_fields_device()
{
    MeanFieldsDevice d{};

    CUDA_CHECK(cudaMalloc((void **)&d.tke_avg, sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void **)&d.tke_total, sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void **)&d.radial_sum_uy, radial_stat_bytes()));
    CUDA_CHECK(cudaMalloc((void **)&d.radial_sum_uy2, radial_stat_bytes()));
    CUDA_CHECK(cudaMalloc((void **)&d.radial_sum_ur, radial_stat_bytes()));
    CUDA_CHECK(cudaMalloc((void **)&d.radial_sum_ur2, radial_stat_bytes()));
    CUDA_CHECK(cudaMalloc((void **)&d.radial_sum_uruy, radial_stat_bytes()));
    CUDA_CHECK(cudaMalloc((void **)&d.radial_count, radial_count_bytes()));

    CUDA_CHECK(cudaMemset(d.tke_total, 0, sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d.tke_avg, 0, sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d.radial_sum_uy, 0, radial_stat_bytes()));
    CUDA_CHECK(cudaMemset(d.radial_sum_uy2, 0, radial_stat_bytes()));
    CUDA_CHECK(cudaMemset(d.radial_sum_ur, 0, radial_stat_bytes()));
    CUDA_CHECK(cudaMemset(d.radial_sum_ur2, 0, radial_stat_bytes()));
    CUDA_CHECK(cudaMemset(d.radial_sum_uruy, 0, radial_stat_bytes()));
    CUDA_CHECK(cudaMemset(d.radial_count, 0, radial_count_bytes()));

    return d;
}

inline MeanFieldsHost allocate_mean_fields_host()
{
    MeanFieldsHost h{};
    h.radial_sum_uy = static_cast<profile_stat_t *>(std::malloc(radial_stat_bytes()));
    h.radial_sum_uy2 = static_cast<profile_stat_t *>(std::malloc(radial_stat_bytes()));
    h.radial_sum_ur = static_cast<profile_stat_t *>(std::malloc(radial_stat_bytes()));
    h.radial_sum_ur2 = static_cast<profile_stat_t *>(std::malloc(radial_stat_bytes()));
    h.radial_sum_uruy = static_cast<profile_stat_t *>(std::malloc(radial_stat_bytes()));
    h.radial_count = static_cast<profile_count_t *>(std::malloc(radial_count_bytes()));

    if (h.radial_sum_uy == nullptr || h.radial_sum_uy2 == nullptr ||
        h.radial_sum_ur == nullptr || h.radial_sum_ur2 == nullptr ||
        h.radial_sum_uruy == nullptr || h.radial_count == nullptr)
    {
        std::fprintf(stderr, "Host allocation failed for radial mean-profile buffers\n");
        std::exit(EXIT_FAILURE);
    }

    return h;
}

inline void free_mean_fields_device(MeanFieldsDevice &d)
{
    CUDA_CHECK(cudaFree(d.tke_total));
    CUDA_CHECK(cudaFree(d.tke_avg));
    CUDA_CHECK(cudaFree(d.radial_sum_uy));
    CUDA_CHECK(cudaFree(d.radial_sum_uy2));
    CUDA_CHECK(cudaFree(d.radial_sum_ur));
    CUDA_CHECK(cudaFree(d.radial_sum_ur2));
    CUDA_CHECK(cudaFree(d.radial_sum_uruy));
    CUDA_CHECK(cudaFree(d.radial_count));

    d = MeanFieldsDevice{};
}

inline void free_mean_fields_host(MeanFieldsHost &h)
{
    std::free(h.radial_sum_uy);
    std::free(h.radial_sum_uy2);
    std::free(h.radial_sum_ur);
    std::free(h.radial_sum_ur2);
    std::free(h.radial_sum_uruy);
    std::free(h.radial_count);
    h = MeanFieldsHost{};
}

#endif
