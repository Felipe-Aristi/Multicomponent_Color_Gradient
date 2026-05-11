#include "bubble/bubbleIn.cuh"
#include "jet/jetIn.cuh"
#include "jet/boundary_conditions.cuh"
#include "lbm.cuh"
#include "stencil_ct.cuh"
#include "collisionOperators.cuh"
#include "utilities/constexprFor.cuh"
#include "utilities/cudaConfig.cuh"
#include "meanF/meanFields.cuh"

//--------------------- Initialize fields --------------------------------------------------

__global__ void bubble(pop_t *fr, pop_t *fb, real_t *rhor, real_t *rhob)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    bubble_mask(rhor, rhob, x, y, z);

    init_equilibrium(fr, fb, rhor, rhob, x, y, z);
}

__global__ void jetDensity(pop_t __restrict__ *fr, pop_t __restrict__ *fb,
                           real_t __restrict__ *rhor, real_t __restrict__ *rhob)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    init_density_jet(fr, rhor, fb, rhob, x, y, z);
}

__global__ void Injet(pop_t __restrict__ *fr, pop_t __restrict__ *fb,
                      real_t __restrict__ *rhor, real_t __restrict__ *rhob)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

    if (inlet_oulet_interior(x, z))
    {
        return;
    }

    jet_mask(fr, rhor, fb, rhob, x, z);
}

//--------------- Main loop ----------------

__global__ void Mfields(const pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                        const pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                        real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                        real_t __restrict__ *Pixx, real_t __restrict__ *Pixy, real_t __restrict__ *Piyy,
                        real_t __restrict__ *Piyz, real_t __restrict__ *Pizz, real_t __restrict__ *Pixz)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    Mfields_calculation(fr, rhor, fb, rhob, ux, uy, uz, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, y, z);
}

__global__ void ColliStream(pop_t __restrict__ *fir, const real_t __restrict__ *rhor,
                            pop_t __restrict__ *fib, const real_t __restrict__ *rhob,
                            const real_t __restrict__ *ux, const real_t __restrict__ *uy, const real_t __restrict__ *uz,
                            const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                            const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    ColliStream_calculations(fir, rhor, fib, rhob, ux, uy, uz, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, y, z);
}

//----------------- Boundary conditions -------------------------

__global__ void inlet(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                      pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                      const real_t __restrict__ *ux, const real_t __restrict__ *uy, const real_t __restrict__ *uz,
                      const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                      const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz)
{

    const label_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const label_t z = blockIdx.y * blockDim.y + threadIdx.y;

    if (inlet_oulet_interior(x, z))
    {
        return;
    }

    inlet_calculation(fr, rhor, fb, rhob, ux, uy, uz, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, z);
}

__global__ void neumann(pop_t __restrict__ *fir, real_t __restrict__ *rhor,
                        pop_t __restrict__ *fib, real_t __restrict__ *rhob,
                        real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                        const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                        const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz)
{

    const label_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const label_t z = blockIdx.y * blockDim.y + threadIdx.y;

    if (inlet_oulet_interior(x, z))
    {
        return;
    }

    neumann_calculation(fir, rhor, fib, rhob, ux, uy, uz, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, z);
}

//------------- Postprocessing -------------------------------

__global__ void compute_total_tke(const real_t *__restrict__ ux,
                                  const real_t *__restrict__ uy,
                                  const real_t *__restrict__ uz,
                                  real_t *__restrict__ tke_total)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    const real_t ke = tke(x, y, z, ux, uy, uz);
    atomicAdd(tke_total, ke);
}

__global__ void update_tke_average(real_t *tke_avg,
                                   const real_t *tke_total,
                                   unsigned int step,
                                   unsigned int init_step)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const unsigned int sample_count = (step - init_step) / NOUTPUT;
        const real_t count = static_cast<real_t>(sample_count);

        *tke_avg = (*tke_avg * count + *tke_total) / (count + static_cast<real_t>(1));
    }
}

__global__ void update_uy_average(const real_t *__restrict__ uy,
                                  real_t *__restrict__ uy_avg,
                                  unsigned int sample_count)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    const size_t id = idx(x, y, z);

    const real_t count = static_cast<real_t>(sample_count);

    uy_avg[id] = (uy_avg[id] * count + uy[id]) / (count + static_cast<real_t>(1));
}

__global__ void accumulate_radial_moments(const real_t *__restrict__ ux,
                                          const real_t *__restrict__ uy,
                                          const real_t *__restrict__ uz,
                                          profile_stat_t *__restrict__ sum_uy,
                                          profile_stat_t *__restrict__ sum_uy2,
                                          profile_stat_t *__restrict__ sum_ur,
                                          profile_stat_t *__restrict__ sum_ur2,
                                          profile_stat_t *__restrict__ sum_uruy,
                                          profile_count_t *__restrict__ count)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    const real_t dx = static_cast<real_t>(x) - jet_x0;
    const real_t dz = static_cast<real_t>(z) - jet_z0;
    const real_t r = sqrt(dx * dx + dz * dz);
    const label_t rb = static_cast<label_t>(r);

    if (rb >= NR_BINS)
    {
        return;
    }

    const size_t id = idx(x, y, z);
    const size_t mid = static_cast<size_t>(y) * static_cast<size_t>(NR_BINS) + static_cast<size_t>(rb);
    const real_t uy_value = uy[id];

    real_t ur_value = static_cast<real_t>(0);
    if (r > static_cast<real_t>(1.0e-12))
    {
        ur_value = (ux[id] * dx + uz[id] * dz) / r;
    }

    atomicAdd(&sum_uy[mid], static_cast<profile_stat_t>(uy_value));
    atomicAdd(&sum_uy2[mid], static_cast<profile_stat_t>(uy_value) * static_cast<profile_stat_t>(uy_value));
    atomicAdd(&sum_ur[mid], static_cast<profile_stat_t>(ur_value));
    atomicAdd(&sum_ur2[mid], static_cast<profile_stat_t>(ur_value) * static_cast<profile_stat_t>(ur_value));
    atomicAdd(&sum_uruy[mid], static_cast<profile_stat_t>(ur_value) * static_cast<profile_stat_t>(uy_value));
    atomicAdd(&count[mid], static_cast<profile_count_t>(1));
}
