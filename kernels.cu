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
                      const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                      const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz)
{

    const label_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const label_t z = blockIdx.y * blockDim.y + threadIdx.y;

    if (inlet_oulet_interior(x, z))
    {
        return;
    }

    inlet_calculation(fr, rhor, fb, rhob, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, z);
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
                                  unsigned int step,
                                  unsigned int step_uy_avg_start)
{
    const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    const size_t id = idx(x, y, z);

    const unsigned int sample_count = step - step_uy_avg_start;
    const real_t count = static_cast<real_t>(sample_count);

    uy_avg[id] = (uy_avg[id] * count + uy[id]) / (count + static_cast<real_t>(1));
}
