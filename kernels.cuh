#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "constants.cuh"
#include "utilities/types.cuh"

__global__ void bubble(real_t *fr, real_t *fb, real_t *rhor, real_t *rhob);

__global__ void jetDensity(real_t __restrict__ *fr, real_t __restrict__ *fb,
                           real_t __restrict__ *rhor, real_t __restrict__ *rhob);

__global__ void Injet(real_t __restrict__ *fr, real_t __restrict__ *fb,
                      real_t __restrict__ *rhor, real_t __restrict__ *rhob);

__global__ void Mfields(const __restrict__ real_t *fr, real_t __restrict__ *rhor,
                        const real_t __restrict__ *fb, real_t __restrict__ *rhob,
                        real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                        real_t __restrict__ *Pixx, real_t __restrict__ *Pixy, real_t __restrict__ *Piyy,
                        real_t __restrict__ *Piyz, real_t __restrict__ *Pizz, real_t __restrict__ *Pixz);

__global__ void ColliStream(real_t __restrict__ *fir, const real_t __restrict__ *rhor,
                            real_t __restrict__ *fib, const real_t __restrict__ *rhob,
                            const real_t __restrict__ *ux, const real_t __restrict__ *uy, const real_t __restrict__ *uz,
                            const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                            const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

__global__ void inlet(real_t __restrict__ *fr, real_t __restrict__ *rhor,
                      real_t __restrict__ *fb, real_t __restrict__ *rhob,
                      const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                      const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

__global__ void neumann(real_t __restrict__ *fir, real_t __restrict__ *rhor,
                        real_t __restrict__ *fib, real_t __restrict__ *rhob,
                        real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                        const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                        const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

#endif