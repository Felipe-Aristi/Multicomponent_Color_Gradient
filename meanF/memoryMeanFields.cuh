#ifndef MEMORY_MEAN_FIELDS_CUH
#define MEMORY_MEAN_FIELDS_CUH

#include <cstdlib>
#include <cstdio>

#include "../constants.cuh"

#include "../utilities/cudaUtilities.cuh"
#include "../utilities/types.cuh"

struct MeanFieldsDevice
{
    real_t *uy_avg = nullptr;
    real_t *tke_total = nullptr;
    real_t *tke_avg = nullptr;
};

struct MeanFieldsHost
{
    real_t *uy_avg = nullptr;
    real_t tke_total = static_cast<real_t>(0);
    real_t tke_avg = static_cast<real_t>(0);
    real_t tke_avg_prev = static_cast<real_t>(0);
    real_t delta = static_cast<real_t>(0);
    real_t abs_delta = static_cast<real_t>(0);
};

struct MeanFieldsState
{
    bool start_uy_average = false;
    bool first_tke_sample = true;
    unsigned int step_uy_avg_start = 0;
    static constexpr real_t tke_tolerance = static_cast<real_t>(1e-4);
};

inline MeanFieldsDevice allocate_mean_fields_device()
{
    MeanFieldsDevice d{};

    CUDA_CHECK(cudaMalloc((void **)&d.tke_avg, sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void **)&d.tke_total, sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void **)&d.uy_avg, Ncells * sizeof(real_t)));

    CUDA_CHECK(cudaMemset(d.tke_total, 0, sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d.tke_avg, 0, sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d.uy_avg, 0, Ncells * sizeof(real_t)));

    return d;
}

inline MeanFieldsHost allocate_mean_fields_host()
{
    MeanFieldsHost h{};
    h.uy_avg = static_cast<real_t *>(std::malloc(Ncells * sizeof(real_t)));

    if (h.uy_avg == nullptr)
    {
        std::fprintf(stderr, "Host allocation failed for MeanFieldsHost::uy_avg\n");
        std::exit(EXIT_FAILURE);
    }

    return h;
}

inline void free_mean_fields_device(MeanFieldsDevice &d)
{
    CUDA_CHECK(cudaFree(d.tke_total));
    CUDA_CHECK(cudaFree(d.tke_avg));
    CUDA_CHECK(cudaFree(d.uy_avg));

    d = MeanFieldsDevice{};
}

inline void free_mean_fields_host(MeanFieldsHost &h)
{
    std::free(h.uy_avg);
    h = MeanFieldsHost{};
}

#endif