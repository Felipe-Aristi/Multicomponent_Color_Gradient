#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "constants.cuh"
#include "utilities/types.cuh"

__global__ void bubble(real_t *f_r, real_t *f_b, real_t *rho_r, real_t *rho_b);

__global__ void jetDensity(real_t *f_r, real_t *f_b, real_t *rho_r, real_t *rho_b);

__global__ void Injet(real_t *f_r, real_t *f_b, real_t *rho_r, real_t *rho_b);

__global__ void density(const real_t *f, real_t *rho);

__global__ void velocity(const real_t *f_r, const real_t *rho_r, const real_t *f_b, const real_t *rho_b, real_t *ux, real_t *uy, real_t *uz);

__global__ void PIneq(const real_t *f, const real_t *rho, const real_t *ux, const real_t *uy, const real_t *uz,
                      real_t *Pixx, real_t *Pixy, real_t *Piyy,
                      real_t *Piyz, real_t *Pizz, real_t *Pixz);

__global__ void ColliStream(real_t *fir, const real_t *rhor,
                            const real_t *Pixxr, const real_t *Pixyr, const real_t *Piyyr,
                            const real_t *Piyzr, const real_t *Pizzr, const real_t *Pixzr,
                            real_t *fib, const real_t *rhob,
                            const real_t *Pixxb, const real_t *Pixyb, const real_t *Piyyb,
                            const real_t *Piyzb, const real_t *Pizzb, const real_t *Pixzb,
                            const real_t *ux, const real_t *uy, const real_t *uz);

__global__ void inlet(real_t *f_r, real_t *rho_r,
                      const real_t *Pixx_r, const real_t *Pixy_r, const real_t *Piyy_r,
                      const real_t *Piyz_r, const real_t *Pizz_r, const real_t *Pixz_r,
                      real_t *f_b, real_t *rho_b,
                      const real_t *Pixx_b, const real_t *Pixy_b, const real_t *Piyy_b,
                      const real_t *Piyz_b, const real_t *Pizz_b, const real_t *Pixz_b);

__global__ void neumann(real_t *fir, real_t *rhor,
                        const real_t *Pixxr, const real_t *Pixyr, const real_t *Piyyr,
                        const real_t *Piyzr, const real_t *Pizzr, const real_t *Pixzr,
                        real_t *fib, real_t *rhob,
                        const real_t *Pixxb, const real_t *Pixyb, const real_t *Piyyb,
                        const real_t *Piyzb, const real_t *Pizzb, const real_t *Pixzb,
                        real_t *ux, real_t *uy, real_t *uz);

#endif