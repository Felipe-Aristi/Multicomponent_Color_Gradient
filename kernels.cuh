#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "constants.cuh"
#include "utilities/types.cuh"

__global__ void bubble(pop_t *fr, pop_t *fb, real_t *rhor, real_t *rhob);

__global__ void jetDensity(pop_t __restrict__ *fr, pop_t __restrict__ *fb,
                           real_t __restrict__ *rhor, real_t __restrict__ *rhob);

__global__ void Injet(pop_t __restrict__ *fr, pop_t __restrict__ *fb,
                      real_t __restrict__ *rhor, real_t __restrict__ *rhob);

__global__ void Mfields(const pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                        const pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                        real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                        real_t __restrict__ *Pixx, real_t __restrict__ *Pixy, real_t __restrict__ *Piyy,
                        real_t __restrict__ *Piyz, real_t __restrict__ *Pizz, real_t __restrict__ *Pixz);

__global__ void ColliStream(pop_t __restrict__ *fir, const real_t __restrict__ *rhor,
                            pop_t __restrict__ *fib, const real_t __restrict__ *rhob,
                            const real_t __restrict__ *ux, const real_t __restrict__ *uy, const real_t __restrict__ *uz,
                            const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                            const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

__global__ void inlet(pop_t __restrict__ *fr, real_t __restrict__ *rhor,
                      pop_t __restrict__ *fb, real_t __restrict__ *rhob,
                      const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                      const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

__global__ void neumann(pop_t __restrict__ *fir, real_t __restrict__ *rhor,
                        pop_t __restrict__ *fib, real_t __restrict__ *rhob,
                        real_t __restrict__ *ux, real_t __restrict__ *uy, real_t __restrict__ *uz,
                        const real_t __restrict__ *Pixx, const real_t __restrict__ *Pixy, const real_t __restrict__ *Piyy,
                        const real_t __restrict__ *Piyz, const real_t __restrict__ *Pizz, const real_t __restrict__ *Pixz);

__global__ void compute_total_tke(const real_t *__restrict__ ux,
                                  const real_t *__restrict__ uy,
                                  const real_t *__restrict__ uz,
                                  real_t *__restrict__ tke_total);

__global__ void update_tke_average(real_t *tke_avg,
                                   const real_t *tke_total,
                                   unsigned int step,
                                   unsigned int init_step);

__global__ void update_uy_average(const real_t *__restrict__ uy,
                                  real_t *__restrict__ uy_avg,
                                  unsigned int sample_count);

__global__ void accumulate_radial_moments(const real_t *__restrict__ ux,
                                          const real_t *__restrict__ uy,
                                          const real_t *__restrict__ uz,
                                          profile_stat_t *__restrict__ sum_uy,
                                          profile_stat_t *__restrict__ sum_uy2,
                                          profile_stat_t *__restrict__ sum_ur,
                                          profile_stat_t *__restrict__ sum_ur2,
                                          profile_stat_t *__restrict__ sum_uruy,
                                          profile_count_t *__restrict__ count);

#endif
