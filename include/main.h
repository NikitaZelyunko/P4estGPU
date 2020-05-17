#ifndef MAIN_HEADER
#define MAIN_HEADER
 #include <cstring>
 #include <iostream>

#include "p4est_to_cuda.h"
#include "memory_utils.h"

#include "main_cuda_callbacks.h"

 #ifndef P4_TO_P8
 #include <p4est_vtk.h>
 #include <p4est_bits.h>
 #include <p4est_extended.h>
 #include <cuda_iterate.h>
 #include <p4est_iterate.h>
 #else
 #include <p8est_vtk.h>
 #include <p8est_bits.h>
 #include <p8est_extended.h>
 #include <p8est_iterate.h>
 #endif

 /* In this example we store data with each quadrant/octant. */
 
 /** Per-quadrant data for this example.
  *
  * In this problem, we keep track of the state variable u, its
  * derivatives in space, and in time.
  */
 typedef struct step3_data
 {
   double              u;             /**< the state variable */
   double              du[P4EST_DIM]; /**< the spatial derivatives */
   double              dudt;          /**< the time derivative */
 }
 step3_data_t;

 typedef struct step3_quad_user_data_to_cuda {
   step3_data *d_step3_user_data;
   double *d_du;
 }step3_quad_user_data_to_cuda_t;

  /** The example parameters.
  *
  * This describes the advection problem and time-stepping used in this
  * example.
  */
 typedef struct step3_ctx
 {
   double              center[P4EST_DIM];  /**< coordinates of the center of
                                                the initial condition Gaussian
                                                bump */
   double              bump_width;         /**< width of the initial condition
                                                Gaussian bump */
   double              max_err;            /**< maximum allowed global
                                                interpolation error */
   double              v[P4EST_DIM];       /**< the advection velocity */
   int                 refine_period;      /**< the number of time steps
                                                between mesh refinement */
   int                 repartition_period; /**< the number of time steps
                                                between repartitioning */
   int                 write_period;       /**< the number of time steps
                                                between writing vtk files */
 }
 step3_ctx_t;

typedef struct step3_ctx_to_cuda
{
  step3_ctx_t *d_step3_ctx;
  double *center;
  double *v;
}step3_ctx_to_cuda_t;

typedef struct step3_compute_max_user_data_to_cuda {
  double *d_user_data;
}step3_compute_max_user_data_to_cuda_t;

typedef struct step3_timestep_update_user_data_to_cuda {
  double *d_user_data;
}step3_timestep_update_user_data_to_cuda_t;

typedef struct step3_divergence_flux_user_data_to_cuda {
  step3_data_t *d_user_data;
}step3_divergence_flux_user_data_to_cuda_t;

#endif