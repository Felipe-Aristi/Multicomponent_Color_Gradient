#include "bubble/bubbleIn.cuh"
#include "jet/jetIn.cuh"
#include "jet/boundary_conditions.cuh"
#include "lbm.cuh"
#include "stencil_ct.cuh"
#include "collisionOperators.cuh"
#include "utilities/constexprFor.cuh"

//--------------------- Initialize fields --------------------------------------------------

static constexpr const std::size_t THREADS_PER_BLOCK = 32 * 4 * 2;
static constexpr const std::size_t BLOCKS_PER_MP = 1;

__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP) __global__ void bubble(pop_t *fr, pop_t *fb, real_t *rhor, real_t *rhob)
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

__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP) __global__ void jetDensity(pop_t __restrict__ *fr, pop_t __restrict__ *fb,
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

__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP) __global__ void Injet(pop_t __restrict__ *fr, pop_t __restrict__ *fb,
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

__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP) __global__ void Mfields(const pop_t __restrict__ *fr, real_t __restrict__ *rhor,
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

__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP) __global__ void ColliStream(pop_t __restrict__ *fir, const real_t __restrict__ *rhor,
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

    const label_t id = idx(x, y, z);

    const real_t rr = rhor[id];
    const real_t rb = rhob[id];
    const real_t rT = rr + rb;
    const real_t invrT = real_t(1 / rT);

    const real_t aR = real_t(rr * invrT);
    const real_t aB = real_t(rb * invrT);

    const real_t vx = ux[id];
    const real_t vy = uy[id];
    const real_t vz = uz[id];
    const real_t pixx = Pixx[id];
    const real_t pixy = Pixy[id];
    const real_t piyy = Piyy[id];
    const real_t piyz = Piyz[id];
    const real_t pizz = Pizz[id];
    const real_t pixz = Pixz[id];

    const real_t omega = omegab; // omega_sponge(y)
    const real_t oms = (static_cast<real_t>(1.0) - omega);

    const real_t I = real_t(4.0) * rr * rb * invrT * invrT;

    if (I <= real_t(1e-4))
    {
        constexpr_for<0, Q>(
            [&] __device__(auto I)
            {
                constexpr label_t i = decltype(I)::value;

                const real_t gieq = feq<i>(rT, vx, vy, vz);
                const real_t gineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz);

                const real_t Omega1 = gieq + oms * gineqr;

                const real_t gi = Omega1;

                const int xn = static_cast<int>(x) + D3Q27::cx<i>();
                const int yn = static_cast<int>(y) + D3Q27::cy<i>();
                const int zn = static_cast<int>(z) + D3Q27::cz<i>();

                const label_t idn = idx(static_cast<label_t>(xn),
                                        static_cast<label_t>(yn),
                                        static_cast<label_t>(zn));

                fir[fidx(idn, i)] = save_pop(aR * gi);
                fib[fidx(idn, i)] = save_pop(aB * gi);
            });
        return;
    }

    real_t Fxr, Fyr, Fzr, Ar, absforcer;
    preOmega2(rhor, rhob, id, taur, taub, Fxr, Fyr, Fzr, absforcer, Ar);

    // const real_t Fxb = -Fxr;
    // const real_t Fyb = -Fyr;
    // const real_t Fzb = -Fzr;
    // const real_t Ab = Ar;

    // real_t Fxb, Fyb, Fzb, Ab, absforceb;
    // preOmega2(rhob, rhor, id, taub, taur, Fxb, Fyb, Fzb, absforceb, Ab);

    constexpr_for<0, Q>(
        [&] __device__(auto I)
        {
            constexpr label_t i = decltype(I)::value;

            const real_t gieq = feq<i>(rT, vx, vy, vz);
            const real_t gineqr = fneqr<i>(pixx, pixy, piyy, piyz, pizz, pixz);

            const real_t Omega1 = gieq + oms * gineqr;

            const real_t Omega2R = Omega2<i>(Fxr, Fyr, Fzr, absforcer, Ar);
            // const real_t Omega2B = Omega2<i>(Fxb, Fyb, Fzb, absforcer, Ab);

            const real_t gi = Omega1 + Omega2R; //+ Omega2B

            const real_t Deltai = recolorDelta<i>(rr, rb, Fxr, Fyr, Fzr, absforcer);

            const int xn = static_cast<int>(x) + D3Q27::cx<i>();
            const int yn = static_cast<int>(y) + D3Q27::cy<i>();
            const int zn = static_cast<int>(z) + D3Q27::cz<i>();

            const label_t idn = idx(static_cast<label_t>(xn),
                                    static_cast<label_t>(yn),
                                    static_cast<label_t>(zn));

            fir[fidx(idn, i)] = save_pop(aR * gi + Deltai);
            fib[fidx(idn, i)] = save_pop(aB * gi - Deltai);
        });
}

//----------------- Boundary conditions -------------------------

__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP) __global__ void inlet(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
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

__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP) __global__ void neumann(pop_t __restrict__ *fir, real_t __restrict__ *rhor,
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
