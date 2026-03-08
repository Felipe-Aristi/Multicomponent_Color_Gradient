#ifndef LAUNCH_CUH
#define LAUNCH_CUH

#include <cuda_runtime.h>

#include "utilities/cudaUtilities.cuh"
#include "utilities/cudaConfig.cuh"

#include "memory.cuh"
#include "kernels.cuh"

inline void launch_bubble(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    bubble<<<cfg.grid, cfg.block, 0, stream>>>(d.fir, d.fib, d.rhor, d.rhob);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_jetDensity(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    jetDensity<<<cfg.grid, cfg.block, 0, stream>>>(d.fir, d.fib, d.rhor, d.rhob);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_Injet(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    Injet<<<grid2D_xz(cfg), block2D_xz(cfg), 0, stream>>>(d.fir, d.fib, d.rhor, d.rhob);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_Macros(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    Mfields<<<cfg.grid, cfg.block, 0, stream>>>(d.fir, d.rhor, d.fib, d.rhob,
                                                d.ux, d.uy, d.uz,
                                                d.Pixx, d.Pixy, d.Piyy, d.Piyz, d.Pizz, d.Pixz);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_collistream(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    ColliStream<<<cfg.grid, cfg.block, 0, stream>>>(
        d.fir, d.rhor, d.fib, d.rhob,
        d.ux, d.uy, d.uz,
        d.Pixx, d.Pixy, d.Piyy, d.Piyz, d.Pizz, d.Pixz);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_inlet_bc(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    inlet<<<grid2D_xz(cfg), block2D_xz(cfg), 0, stream>>>(
        d.fir, d.rhor, d.fib, d.rhob,
        d.Pixx, d.Pixy, d.Piyy, d.Piyz, d.Pizz, d.Pixz);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_neumann_bc(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    neumann<<<grid2D_xz(cfg), block2D_xz(cfg), 0, stream>>>(
        d.fir, d.rhor, d.fib, d.rhob,
        d.ux, d.uy, d.uz,
        d.Pixx, d.Pixy, d.Piyy, d.Piyz, d.Pizz, d.Pixz);

    CUDA_CHECK(cudaGetLastError());
}

#endif