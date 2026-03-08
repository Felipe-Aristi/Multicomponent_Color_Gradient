#include "bubble/bubbleIn.cuh"
#include "jet/jetIn.cuh"
#include "jet/boundary_conditions.cuh"
#include "lbm.cuh"
#include "collisionOperators.cuh"

//--------------------- Initialize fields --------------------------------------------------

__global__ void bubble(real_t *fr, real_t *fb, real_t *rhor, real_t *rhob)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    bubble_mask(rhor, rhob, x, y, z);

    init_equilibrium(fr, fb, rhor, rhob, x, y, z);
}

__global__ void jetDensity(real_t *fr, real_t *fb, real_t *rhor, real_t *rhob)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    init_density_jet(fr, rhor, fb, rhob, x, y, z);
}

__global__ void Injet(real_t *fr, real_t *fb, real_t *rhor, real_t *rhob)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;

    if (inlet_oulet_interior(x, z))
    {
        return;
    }

    jet_mask(fr, rhor, fb, rhob, x, z);
}

//--------------- Main loop ----------------

__global__ void Mfields(const real_t *fr, real_t *rhor,
                        const real_t *fb, real_t *rhob,
                        real_t *ux, real_t *uy, real_t *uz,
                        real_t *Pixx, real_t *Pixy, real_t *Piyy,
                        real_t *Piyz, real_t *Pizz, real_t *Pixz)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    Mfields_calculation(fr, rhor, fb, rhob, ux, uy, uz, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, y, z);
}

__global__ void ColliStream(real_t *fir, const real_t *rhor,
                            real_t *fib, const real_t *rhob,
                            const real_t *ux, const real_t *uy, const real_t *uz,
                            const real_t *Pixx, const real_t *Pixy, const real_t *Piyy,
                            const real_t *Piyz, const real_t *Pizz, const real_t *Pixz)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    int id = idx(x, y, z);

    real_t Fxr, Fyr, Fzr, Ar, absforcer;
    preOmega2(rhor, rhob, x, y, z, taur, taub, Fxr, Fyr, Fzr, absforcer, Ar);

    real_t Fxb, Fyb, Fzb, Ab, absforceb;
    preOmega2(rhob, rhor, x, y, z, taub, taur, Fxb, Fyb, Fzb, absforceb, Ab);

    const real_t rr = rhor[id];
    const real_t rb = rhob[id];
    const real_t rT = rr + rb;

    const real_t vx = ux[id];
    const real_t vy = uy[id];
    const real_t vz = uz[id];

    const real_t pixx = Pixx[id];
    const real_t pixy = Pixy[id];
    const real_t piyy = Piyy[id];
    const real_t piyz = Piyz[id];
    const real_t pizz = Pizz[id];
    const real_t pixz = Pixz[id];

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {

        const real_t gieq = feq(i, rT, vx, vy, vz);
        const real_t gineqr = fneqr(i, pixx, pixy, piyy, piyz, pizz, pixz);

        const real_t Omega1 = gieq + (real_t(1.0) - omegab) * gineqr;

        const real_t Omega2R = Omega2(i, rr, taur, rb, taub, Fxr, Fyr, Fzr, absforcer, Ar);
        const real_t Omega2B = Omega2(i, rb, taub, rr, taur, Fxb, Fyb, Fzb, absforceb, Ab);

        const real_t gi = Omega1 + Omega2R + Omega2B;

        const real_t Deltai = recolorDelta(i, rr, rb, Fxr, Fyr, Fzr, absforcer);

        const int idn = idx(x + d_cx[i], y + d_cy[i], z + d_cz[i]);

        fir[fidx(idn, i)] = real_t(rr / (rr + rb)) * gi + Deltai;

        fib[fidx(idn, i)] = real_t(rb / (rr + rb)) * gi - Deltai;
    }
}

//----------------- Boundary conditions -------------------------

__global__ void inlet(real_t *fr, real_t *rhor, real_t *fb, real_t *rhob,
                      const real_t *Pixx, const real_t *Pixy, const real_t *Piyy, const real_t *Piyz, const real_t *Pizz, const real_t *Pixz)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (inlet_oulet_interior(x, z))
    {
        return;
    }

    inlet_calculation(fr, rhor, fb, rhob, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, z);
}

__global__ void neumann(real_t *fir, real_t *rhor,
                        real_t *fib, real_t *rhob,
                        real_t *ux, real_t *uy, real_t *uz,
                        const real_t *Pixx, const real_t *Pixy, const real_t *Piyy, const real_t *Piyz, const real_t *Pizz, const real_t *Pixz)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (inlet_oulet_interior(x, z))
    {
        return;
    }

    neumann_calculation(fir, rhor, fib, rhob, ux, uy, uz, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, z);
}
