#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "constants.cuh"
#include "utilities/types.cuh"

__global__ void bubble(real_t *fr, real_t *fb, real_t *rhor, real_t *rhob);

__global__ void jetDensity(real_t *fr, real_t *fb, real_t *rhor, real_t *rhob);

__global__ void Injet(real_t *fr, real_t *fb, real_t *rhor, real_t *rhob);

__global__ void Mfields(const real_t *fr, real_t *rhor,
                        const real_t *fb, real_t *rhob,
                        real_t *ux, real_t *uy, real_t *uz,
                        real_t *Pixx, real_t *Pixy, real_t *Piyy,
                        real_t *Piyz, real_t *Pizz, real_t *Pixz);

__global__ void ColliStream(real_t *fir, const real_t *rhor,
                            real_t *fib, const real_t *rhob,
                            const real_t *ux, const real_t *uy, const real_t *uz,
                            const real_t *Pixx, const real_t *Pixy, const real_t *Piyy,
                            const real_t *Piyz, const real_t *Pizz, const real_t *Pixz);

__global__ void inlet(real_t *fr, real_t *rhor, real_t *fb, real_t *rhob,
                      const real_t *Pixx, const real_t *Pixy, const real_t *Piyy, const real_t *Piyz, const real_t *Pizz, const real_t *Pixz);

__global__ void neumann(real_t *fir, real_t *rhor, real_t *fib, real_t *rhob,
                        real_t *ux, real_t *uy, real_t *uz,
                        const real_t *Pixx, const real_t *Pixy, const real_t *Piyy, const real_t *Piyz, const real_t *Pizz, const real_t *Pixz);

#endif