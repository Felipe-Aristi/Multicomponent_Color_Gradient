// memory.cuh
#ifndef MEMORY_CUH
#define MEMORY_CUH

#include <cuda_runtime.h>
#include <cstring> // std::memset
#include "constants.cuh"
#include "utilities/cudaUtilities.cuh"

// -------------------- Device fields --------------------
struct LbmDevice
{
    real_t *fir = nullptr;
    real_t *fib = nullptr;

    real_t *rhor = nullptr;
    real_t *rhob = nullptr;

    real_t *ux = nullptr;
    real_t *uy = nullptr;
    real_t *uz = nullptr;

    real_t *Pixxr = nullptr;
    real_t *Pixyr = nullptr;
    real_t *Piyyr = nullptr;
    real_t *Piyzr = nullptr;
    real_t *Pizzr = nullptr;
    real_t *Pixzr = nullptr;

    real_t *Pixxb = nullptr;
    real_t *Pixyb = nullptr;
    real_t *Piyyb = nullptr;
    real_t *Piyzb = nullptr;
    real_t *Pizzb = nullptr;
    real_t *Pixzb = nullptr;
};

// -------------------- Host buffers (for VTK output) --------------------
struct LbmHost
{
    real_t *rhor = nullptr;
    real_t *rhob = nullptr;

    real_t *ux = nullptr;
    real_t *uy = nullptr;
    real_t *uz = nullptr;
};

// -------------------- Allocation helpers --------------------

inline LbmHost allocate_host_memory()
{
    LbmHost h{};

    CUDA_CHECK(cudaMallocHost(&h.rhor, bytesCell));
    CUDA_CHECK(cudaMallocHost(&h.rhob, bytesCell));

    CUDA_CHECK(cudaMallocHost(&h.ux, bytesCell));
    CUDA_CHECK(cudaMallocHost(&h.uy, bytesCell));
    CUDA_CHECK(cudaMallocHost(&h.uz, bytesCell));

    return h;
}

inline LbmDevice allocate_device_memory()
{
    LbmDevice d{};

    CUDA_CHECK(cudaMalloc(&d.fir, bytesF));
    CUDA_CHECK(cudaMalloc(&d.fib, bytesF));

    CUDA_CHECK(cudaMalloc(&d.rhor, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.rhob, bytesCell));

    CUDA_CHECK(cudaMalloc(&d.ux, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.uy, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.uz, bytesCell));

    CUDA_CHECK(cudaMalloc(&d.Pixxr, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pixyr, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Piyyr, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Piyzr, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pizzr, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pixzr, bytesCell));

    CUDA_CHECK(cudaMalloc(&d.Pixxb, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pixyb, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Piyyb, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Piyzb, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pizzb, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pixzb, bytesCell));

    CUDA_CHECK(cudaMemset(d.fir, 0, bytesF));
    CUDA_CHECK(cudaMemset(d.fib, 0, bytesF));

    CUDA_CHECK(cudaMemset(d.rhor, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.rhob, 0, bytesCell));

    CUDA_CHECK(cudaMemset(d.ux, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.uy, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.uz, 0, bytesCell));

    CUDA_CHECK(cudaMemset(d.Pixxr, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pixyr, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Piyyr, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Piyzr, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pizzr, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pixzr, 0, bytesCell));

    CUDA_CHECK(cudaMemset(d.Pixxb, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pixyb, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Piyyb, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Piyzb, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pizzb, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pixzb, 0, bytesCell));

    return d;
}

// -------------------- Free memory helpers --------------------

inline void free_host_memory(LbmHost &h)
{
    CUDA_CHECK(cudaFreeHost(h.rhor));
    CUDA_CHECK(cudaFreeHost(h.rhob));

    CUDA_CHECK(cudaFreeHost(h.ux));
    CUDA_CHECK(cudaFreeHost(h.uy));
    CUDA_CHECK(cudaFreeHost(h.uz));

    h = LbmHost{};
}

inline void free_device_memory(LbmDevice &d)
{
    CUDA_CHECK(cudaFree(d.rhor));
    CUDA_CHECK(cudaFree(d.rhob));

    CUDA_CHECK(cudaFree(d.ux));
    CUDA_CHECK(cudaFree(d.uy));
    CUDA_CHECK(cudaFree(d.uz));

    CUDA_CHECK(cudaFree(d.Pixxr));
    CUDA_CHECK(cudaFree(d.Pixyr));
    CUDA_CHECK(cudaFree(d.Piyyr));
    CUDA_CHECK(cudaFree(d.Piyzr));
    CUDA_CHECK(cudaFree(d.Pizzr));
    CUDA_CHECK(cudaFree(d.Pixzr));

    CUDA_CHECK(cudaFree(d.Pixxb));
    CUDA_CHECK(cudaFree(d.Pixyb));
    CUDA_CHECK(cudaFree(d.Piyyb));
    CUDA_CHECK(cudaFree(d.Piyzb));
    CUDA_CHECK(cudaFree(d.Pizzb));
    CUDA_CHECK(cudaFree(d.Pixzb));

    CUDA_CHECK(cudaFree(d.fir));
    CUDA_CHECK(cudaFree(d.fib));

    d = LbmDevice{};
}

inline void copy_out_D2H(LbmHost &h, const LbmDevice &d)
{
    CUDA_CHECK(cudaMemcpy(h.rhor, d.rhor, bytesCell, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.rhob, d.rhob, bytesCell, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h.ux, d.ux, bytesCell, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.uy, d.uy, bytesCell, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.uz, d.uz, bytesCell, cudaMemcpyDeviceToHost));
}

#endif