#include "bubble/bubbleIn.cuh"
#include "lbm.cuh"
#include "collisionOperators.cuh"

__global__ void bubble(real_t *f_r, real_t *f_b, real_t *rho_r, real_t *rho_b)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    bubble_mask(rho_r, rho_b, x, y, z);

    init_equilibrium(f_r, f_b, rho_r, rho_b, x, y, z);
}

__global__ void density(const real_t *f, real_t *rho)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    density_calculation(f, rho, x, y, z);
}

__global__ void velocity(const real_t *f_r, const real_t *rho_r,
                         const real_t *f_b, const real_t *rho_b,
                         real_t *ux, real_t *uy, real_t *uz)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (interior(x, y, z))
    {
        return;
    }

    velocity_calculation(f_r, rho_r, f_b, rho_b, ux, uy, uz, x, y, z);
}

__global__ void PIneq(const real_t *f, const real_t *rho, const real_t *ux, const real_t *uy, const real_t *uz,
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

    PIneq_calculation(f, rho, ux, uy, uz, Pixx, Pixy, Piyy, Piyz, Pizz, Pixz, x, y, z);
}

__global__ void ColliStream(real_t *fir, const real_t *rhor,
                            const real_t *Pixxr, const real_t *Pixyr, const real_t *Piyyr,
                            const real_t *Piyzr, const real_t *Pizzr, const real_t *Pixzr,
                            real_t *fib, const real_t *rhob,
                            const real_t *Pixxb, const real_t *Pixyb, const real_t *Piyyb,
                            const real_t *Piyzb, const real_t *Pizzb, const real_t *Pixzb,
                            const real_t *ux, const real_t *uy, const real_t *uz)
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

    /*     real_t Fxb, Fyb, Fzb, Ab, absforceb;
        preOmega2(rhob, rhor, x, y, z, taub, taur, Fxb, Fyb, Fzb, absforceb, Ab); */

#pragma unroll 27
    for (int i = 0; i < Q; ++i)
    {

        const real_t Omega1R = Omega1(i, rhor[id], ux[id], uy[id], uz[id], Pixxr[id], Pixyr[id], Piyyr[id], Piyzr[id], Pizzr[id], Pixzr[id], omegar);
        const real_t Omega2R = Omega2(i, rhor[id], taur, rhob[id], taub, Fxr, Fyr, Fzr, absforcer, Ar);

        const real_t Omega1B = Omega1(i, rhob[id], ux[id], uy[id], uz[id], Pixxb[id], Pixyb[id], Piyyb[id], Piyzb[id], Pizzb[id], Pixzb[id], omegab);
        // const real_t Omega2B = Omega2(i, rhob[id], taub, rhor[id], taur, Fxb, Fyb, Fzb, absforceb, Ab);

        const real_t gi = Omega1R + Omega1B + Omega2R; //  + Omega2R

        const real_t Deltai = recolorDelta(i, rhor[id], rhob[id], Fxr, Fyr, Fzr, absforcer);

        const int idn = idx(x + d_cx[i], y + d_cy[i], z + d_cz[i]);

        fir[fidx(idn, i)] = real_t(rhor[id] / (rhor[id] + rhob[id])) * gi + Deltai; //

        fib[fidx(idn, i)] = real_t(rhob[id] / (rhor[id] + rhob[id])) * gi - Deltai; //
    }
}
