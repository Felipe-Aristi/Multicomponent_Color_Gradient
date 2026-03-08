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

    real_t *Pixx = nullptr;
    real_t *Pixy = nullptr;
    real_t *Piyy = nullptr;
    real_t *Piyz = nullptr;
    real_t *Pizz = nullptr;
    real_t *Pixz = nullptr;
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

    CUDA_CHECK(cudaMalloc(&d.Pixx, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pixy, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Piyy, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Piyz, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pizz, bytesCell));
    CUDA_CHECK(cudaMalloc(&d.Pixz, bytesCell));

    CUDA_CHECK(cudaMemset(d.fir, 0, bytesF));
    CUDA_CHECK(cudaMemset(d.fib, 0, bytesF));

    CUDA_CHECK(cudaMemset(d.rhor, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.rhob, 0, bytesCell));

    CUDA_CHECK(cudaMemset(d.ux, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.uy, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.uz, 0, bytesCell));

    CUDA_CHECK(cudaMemset(d.Pixx, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pixy, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Piyy, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Piyz, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pizz, 0, bytesCell));
    CUDA_CHECK(cudaMemset(d.Pixz, 0, bytesCell));

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

    CUDA_CHECK(cudaFree(d.Pixx));
    CUDA_CHECK(cudaFree(d.Pixy));
    CUDA_CHECK(cudaFree(d.Piyy));
    CUDA_CHECK(cudaFree(d.Piyz));
    CUDA_CHECK(cudaFree(d.Pizz));
    CUDA_CHECK(cudaFree(d.Pixz));

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