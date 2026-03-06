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

inline void launch_density(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    density<<<cfg.grid, cfg.block, 0, stream>>>(d.fir, d.rhor);
    density<<<cfg.grid, cfg.block, 0, stream>>>(d.fib, d.rhob);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_velocity(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    velocity<<<cfg.grid, cfg.block, 0, stream>>>(d.fir, d.rhor, d.fib, d.rhob, d.ux, d.uy, d.uz);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_pineq(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    PIneq<<<cfg.grid, cfg.block, 0, stream>>>(
        d.fir, d.rhor, d.ux, d.uy, d.uz,
        d.Pixxr, d.Pixyr, d.Piyyr, d.Piyzr, d.Pizzr, d.Pixzr);

    PIneq<<<cfg.grid, cfg.block, 0, stream>>>(
        d.fib, d.rhob, d.ux, d.uy, d.uz,
        d.Pixxb, d.Pixyb, d.Piyyb, d.Piyzb, d.Pizzb, d.Pixzb);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_collistream(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    ColliStream<<<cfg.grid, cfg.block, 0, stream>>>(
        d.fir, d.rhor,
        d.Pixxr, d.Pixyr, d.Piyyr, d.Piyzr, d.Pizzr, d.Pixzr,
        d.fib, d.rhob,
        d.Pixxb, d.Pixyb, d.Piyyb, d.Piyzb, d.Pizzb, d.Pixzb,
        d.ux, d.uy, d.uz);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_inlet_bc(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    inlet<<<grid2D_xz(cfg), block2D_xz(cfg), 0, stream>>>(
        d.fir, d.rhor,
        d.Pixxr, d.Pixyr, d.Piyyr, d.Piyzr, d.Pizzr, d.Pixzr,
        d.fib, d.rhob,
        d.Pixxb, d.Pixyb, d.Piyyb, d.Piyzb, d.Pizzb, d.Pixzb);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_neumann_bc(const CudaConfig &cfg, const LbmDevice &d, cudaStream_t stream = 0)
{
    neumann<<<grid2D_xz(cfg), block2D_xz(cfg), 0, stream>>>(
        d.fir, d.rhor,
        d.Pixxr, d.Pixyr, d.Piyyr, d.Piyzr, d.Pizzr, d.Pixzr,
        d.fib, d.rhob,
        d.Pixxb, d.Pixyb, d.Piyyb, d.Piyzb, d.Pizzb, d.Pixzb,
        d.ux, d.uy, d.uz);

    CUDA_CHECK(cudaGetLastError());
}

#endif