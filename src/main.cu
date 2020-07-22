/*
  This file is part of p4est.
  p4est is a C library to manage a collection (a forest) of multiple
  connected adaptive quadtrees or octrees in parallel.

  Copyright (C) 2010 The University of Texas System
  Additional copyright (C) 2011 individual authors
  Written by Carsten Burstedde, Lucas C. Wilcox, and Tobin Isaac

  p4est is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  p4est is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with p4est; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/** \file p4est_step3.c
 *
 * This 2D example program uses p4est to solve a simple advection problem.  It
 * is numerically very simple, and intended to demonstrate several methods of
 * interacting with the p4est data after it has been refined and partitioned.
 * It demonstrates the construction of ghost layers (see p4est_ghost_t in
 * p4est_ghost.h) and communication of ghost-layer data, and it demonstrates
 * interacting with the quadrants and quadrant boundaries through the
 * p4est_iterate() routine (see p4est_iterate.h).
 */

/* p4est has two separate interfaces for 2D and 3D, p4est*.h and p8est*.h.
 * Most API functions are available for both dimensions.  The header file
 * p4est_to_p8est.h #define's the 2D names to the 3D names such that most code
 * only needs to be written once.  In this example, we rely on this. */
 #include "main.h"
 #include <chrono>
 #include "time.h" 

using namespace std; 
using namespace std::chrono; 

const double difficulty = 0.0005;

__constant__ char cuda_ctx_ptr[sizeof(step3_ctx_t)];

 /** We had 1. / 0. here to create a NaN but that is not portable. */
 static const double step3_invalid = -1.;

 void alloc_cuda_memory_step3_quad_user_data(quad_user_data_allocate_info_t* quad_user_data_allocate_info) {
   step3_quad_user_data_to_cuda_t *user_data_to_cuda = (step3_quad_user_data_to_cuda_t*) malloc(sizeof(step3_quad_user_data_to_cuda_t));
   step3_data_t *user_data = (step3_data_t*) quad_user_data_allocate_info->user_data;
   // to do what is 0x1
   if(user_data != 0 && user_data != (void*)0x1) {
    step3_data_t *d_step3_user_data;
    gpuErrchk(cudaMalloc((void**)&d_step3_user_data, sizeof(step3_data_t)));
    gpuErrchk(cudaMemcpy(d_step3_user_data, user_data, sizeof(step3_data_t), cudaMemcpyHostToDevice));
    user_data_to_cuda->d_step3_user_data = d_step3_user_data;

    double *d_du;
    size_t d_du_size = P4EST_DIM;
    arrayPropMemoryAllocate((void**)&d_du, d_du_size, &(d_step3_user_data->du), sizeof(double*), user_data->du);
    user_data_to_cuda->d_du = d_du;
   } else {
     user_data_to_cuda->d_step3_user_data = user_data;
     user_data_to_cuda->d_du = NULL;
   }
   quad_user_data_allocate_info->cuda_memory_allocating_info = user_data_to_cuda;
 }

void alloc_all_quads_cuda_memory_step3(all_quads_user_data_allocate_info_t* all_quads_user_data_allocate_info, sc_array_t* quadrants) {
  size_t d_quadrants_array_size = quadrants->elem_count;
  all_quads_user_data_allocate_info->quads_count = d_quadrants_array_size;
  size_t d_all_quads_user_data_bytes_count = d_quadrants_array_size * sizeof(step3_data_t);

  step3_data_t *d_all_quads_user_data;
  void **all_quads_host_user_data = (void**) malloc(d_quadrants_array_size * sizeof(void*));
  step3_data_t *quads_user_data_temp = (step3_data_t*) malloc(d_all_quads_user_data_bytes_count);
  for(size_t i = 0; i < d_quadrants_array_size; i++) {
    p4est_quadrant_t *temp_quad = p4est_quadrant_array_index (quadrants, i);
    memcpy(quads_user_data_temp + i, temp_quad->p.user_data, sizeof(step3_data_t));
    all_quads_host_user_data[i] = temp_quad->p.user_data;
  }
  
  gpuErrchk(cudaMalloc((void**)&d_all_quads_user_data, d_all_quads_user_data_bytes_count));
  gpuErrchk(cudaMemcpy(d_all_quads_user_data, quads_user_data_temp, d_all_quads_user_data_bytes_count, cudaMemcpyHostToDevice));
  all_quads_user_data_allocate_info->d_all_quads_user_data = (void*)d_all_quads_user_data;
  all_quads_user_data_allocate_info->all_quads_user_data = all_quads_host_user_data;
}

 void update_quad_cuda_step3_user_data(quad_user_data_allocate_info_t* old_user_data_allocate_info, quad_user_data_allocate_info_t* new_user_data_allocate_info) {
  step3_quad_user_data_to_cuda_t *user_data_to_cuda = (step3_quad_user_data_to_cuda_t*) malloc(sizeof(step3_quad_user_data_to_cuda_t));
  step3_quad_user_data_to_cuda_t *old_user_data_to_cuda = (step3_quad_user_data_to_cuda_t*) old_user_data_allocate_info->cuda_memory_allocating_info;
  step3_data_t *old_user_data = (step3_data_t*) old_user_data_allocate_info->user_data;
  step3_data_t *new_user_data = (step3_data_t*) new_user_data_allocate_info->user_data;
  // to do what is 0x1
  if(old_user_data != 0 && old_user_data != (void*)0x1 && new_user_data != 0 && new_user_data != (void*)0x1) {
   step3_data_t *d_step3_user_data = old_user_data_to_cuda->d_step3_user_data;
   gpuErrchk(cudaMemcpy(d_step3_user_data, new_user_data, sizeof(step3_data_t), cudaMemcpyHostToDevice));
   user_data_to_cuda->d_step3_user_data = d_step3_user_data;

   double *d_du = old_user_data_to_cuda->d_du;
   size_t d_du_size = P4EST_DIM;
   arrayPropMemoryUpdate((void**)&d_du, d_du_size * sizeof(double), new_user_data->du);
   user_data_to_cuda->d_du = d_du;
  } else {
    user_data_to_cuda->d_step3_user_data = new_user_data;
    user_data_to_cuda->d_du = NULL;
  }
  new_user_data_allocate_info->cuda_memory_allocating_info = user_data_to_cuda;
 }

void update_all_quads_cuda_user_data_step3(all_quads_user_data_allocate_info* old_user_data_allocate_info, all_quads_user_data_allocate_info* new_user_data_allocate_info) {
  size_t d_quadrants_array_size = old_user_data_allocate_info->quads_count;
  size_t d_all_quads_user_data_bytes_count = d_quadrants_array_size * sizeof(step3_data_t);

  void *d_all_quads_user_data = old_user_data_allocate_info->d_all_quads_user_data;
  step3_data_t **all_quads_host_user_data = (step3_data_t**)new_user_data_allocate_info->all_quads_user_data;
  step3_data_t *quads_user_data_temp = (step3_data_t*) malloc(d_all_quads_user_data_bytes_count);
  step3_data_t *quad_user_data_cursor = quads_user_data_temp;
  for(size_t i = 0; i < d_quadrants_array_size; i++, quad_user_data_cursor++) {
    step3_data_t *new_user_data = (step3_data_t*)(all_quads_host_user_data[i]);
    if(new_user_data != NULL) {
      memcpy(quad_user_data_cursor, new_user_data, sizeof(step3_data_t));
    }
  }
  
  gpuErrchk(cudaMemcpy(d_all_quads_user_data, quads_user_data_temp, d_all_quads_user_data_bytes_count, cudaMemcpyHostToDevice));
  new_user_data_allocate_info->d_all_quads_user_data = d_all_quads_user_data;
  new_user_data_allocate_info->quads_count = d_quadrants_array_size;
}

void free_cuda_memory_step3_quad_user_data(quad_user_data_allocate_info_t* quad_user_data_allocate_info) {
  step3_quad_user_data_to_cuda_t *user_data_to_cuda = (step3_quad_user_data_to_cuda_t*) quad_user_data_allocate_info->cuda_memory_allocating_info;
  if(user_data_to_cuda->d_du != NULL) {
    gpuErrchk(cudaFree(user_data_to_cuda->d_du));
  }
  if(user_data_to_cuda->d_step3_user_data != NULL && user_data_to_cuda->d_step3_user_data != (void*)0x1) {
    gpuErrchk(cudaFree(user_data_to_cuda->d_step3_user_data));
  }
}

void free_all_quads_cuda_memory_step3(all_quads_user_data_allocate_info_t* all_quads_user_data_allocate_info) {
  gpuErrchk(cudaFree(all_quads_user_data_allocate_info->d_all_quads_user_data));
  free(all_quads_user_data_allocate_info->all_quads_user_data);
}

void* get_cuda_allocated_user_data_step3_quad_user_data(quad_user_data_allocate_info_t* quad_user_data_allocate_info) {
  step3_quad_user_data_to_cuda_t *user_data_to_cuda = (step3_quad_user_data_to_cuda_t*) quad_user_data_allocate_info->cuda_memory_allocating_info;
  return user_data_to_cuda != NULL ? (void*) user_data_to_cuda->d_step3_user_data : NULL;
}

void download_quad_cuda_user_data_step3_to_host (quad_user_data_allocate_info_t* user_data_allocate_info) {
  step3_data_t *user_data = (step3_data_t*) user_data_allocate_info->user_data;
  step3_quad_user_data_to_cuda_t *user_data_to_cuda = (step3_quad_user_data_to_cuda_t*) user_data_allocate_info->cuda_memory_allocating_info;
  gpuErrchk(cudaMemcpy(user_data, user_data_to_cuda->d_step3_user_data, sizeof(step3_data_t), cudaMemcpyDeviceToHost));
}

void download_all_quads_cuda_user_data_to_host_t_step3(all_quads_user_data_allocate_info_t* all_quads_user_data_allocate_info, sc_array_t* quadrants) {
  size_t quads_count = all_quads_user_data_allocate_info->quads_count;
  size_t user_data_size = sizeof(step3_data_t);
  size_t user_data_bytes_alloc = quads_count * user_data_size;
  step3_data_t *copied_user_data = (step3_data_t*)malloc(user_data_bytes_alloc);
  
  gpuErrchk(cudaMemcpy(copied_user_data, all_quads_user_data_allocate_info->d_all_quads_user_data, user_data_bytes_alloc, cudaMemcpyDeviceToHost));
  step3_data_t *copied_user_data_cursor = copied_user_data;
  for(size_t i = 0; i < quads_count; i++, copied_user_data_cursor++) {
    p4est_quadrant_t *quad = p4est_quadrant_array_index(quadrants, i);
    memcpy(quad->p.user_data, copied_user_data_cursor, user_data_size);
  }
  free(copied_user_data);
}


void alloc_cuda_memory_step3_ctx(user_data_for_cuda_t* user_data_api) {
  step3_ctx_to_cuda_t *ctx_to_cuda = (step3_ctx_to_cuda_t*) malloc(sizeof(step3_ctx_to_cuda_t));
  step3_ctx_t *ctx = (step3_ctx*) user_data_api->user_data;
  step3_ctx_t *d_step3_ctx;

  gpuErrchk(cudaMalloc((void**)&d_step3_ctx, sizeof(step3_ctx_t)));
  gpuErrchk(cudaMemcpy(d_step3_ctx, ctx, sizeof(step3_ctx_t), cudaMemcpyHostToDevice));
  ctx_to_cuda->d_step3_ctx = d_step3_ctx;
  user_data_api->cuda_memory_allocating_info = (void*) ctx_to_cuda;
}

void free_cuda_memory_step3_ctx(user_data_for_cuda_t* user_data_api) {
  step3_ctx_to_cuda *ctx_to_cuda = (step3_ctx_to_cuda*) user_data_api->cuda_memory_allocating_info;

  gpuErrchk(cudaFree(ctx_to_cuda->d_step3_ctx));
}

void* get_cuda_allocated_user_data_step3_ctx(user_data_for_cuda_t* user_data_api) {
  step3_ctx_to_cuda *ctx_to_cuda = (step3_ctx_to_cuda*) user_data_api->cuda_memory_allocating_info;
  return (void*) ctx_to_cuda->d_step3_ctx;
}
 
 /** Compute the value and derivatives of the initial condition.
  *
  * \param [in]  x   the coordinates
  * \param [out] du  the derivative at \a x
  * \param [in]  ctx the example parameters
  *
  * \return the initial condition at \a x
  */
 static double
 step3_initial_condition (double x[], double du[], step3_ctx_t * ctx)
 {
   int                 i;
   double             *c = ctx->center;
   double              bump_width = ctx->bump_width;
   double              r2, d[P4EST_DIM];
   double              arg, retval;
 
   r2 = 0.;
   for (i = 0; i < P4EST_DIM; i++) {
     d[i] = x[i] - c[i];
     r2 += d[i] * d[i];
   }
 
   arg = -(1. / 2.) * r2 / bump_width / bump_width;
   retval = exp (arg);
 
   if (du) {
     for (i = 0; i < P4EST_DIM; i++) {
       du[i] = -(1. / bump_width / bump_width) * d[i] * retval;
     }
   }
 
   return retval;
 }
 
 /** Get the coordinates of the midpoint of a quadrant.
  *
  * \param [in]  p4est      the forest
  * \param [in]  which_tree the tree in the forest containing \a q
  * \param [in]  q          the quadrant
  * \param [out] xyz        the coordinates of the midpoint of \a q
  */
 static void
 step3_get_midpoint (p4est_t * p4est, p4est_topidx_t which_tree,
                     p4est_quadrant_t * q, double xyz[3])
 {
   p4est_qcoord_t      half_length = P4EST_QUADRANT_LEN (q->level) / 2;
 
   p4est_qcoord_to_vertex (p4est->connectivity, which_tree,
                           q->x + half_length, q->y + half_length,
 #ifdef P4_TO_P8
                           q->z + half_length,
 #endif
                           xyz);
 }
 
 /** Initialize the initial condition data of a quadrant.
  *
  * This function matches the p4est_init_t prototype that is used by
  * p4est_new(), p4est_refine(), p4est_coarsen(), and p4est_balance().
  *
  * \param [in] p4est          the forest
  * \param [in] which_tree     the tree in the forest containing \a q
  * \param [in,out] q          the quadrant whose data gets initialized
  */
 static void
 step3_init_initial_condition (p4est_t * p4est, p4est_topidx_t which_tree,
                               p4est_quadrant_t * q)
 {
   /* the data associated with a forest is accessible by user_pointer */
   step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
   /* the data associated with a quadrant is accessible by p.user_data */
   step3_data_t       *data = (step3_data_t *) q->p.user_data;
   double              midpoint[3];
 
   step3_get_midpoint (p4est, which_tree, q, midpoint);
   /* initialize the data */
   data->u = step3_initial_condition (midpoint, data->du, ctx);
 }
 
 /** Estimate the square of the approximation error on a quadrant.
  *
  * We compute our estimate by integrating the difference of a constant
  * approximation at the midpoint and a linear approximation that interpolates
  * at the midpoint.
  *
  * \param [in] q a quadrant
  *
  * \return the square of the error estimate for the state variables contained
  * in \a q's data.
  */
 static double
 step3_error_sqr_estimate (p4est_quadrant_t * q)
 {
   step3_data_t       *data = (step3_data_t *) q->p.user_data;
   int                 i;
   double              diff2;
   double             *du = data->du;
   double              h =
     (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;
   double              vol;
 
 #ifdef P4_TO_P8
   vol = h * h * h;
 #else
   vol = h * h;
 #endif
 
   diff2 = 0.;
   /* use the approximate derivative to estimate the L2 error */
   for (i = 0; i < P4EST_DIM; i++) {
     diff2 += du[i] * du[i] * (1. / 12.) * h * h * vol;
   }
 
   return diff2;
 }
 
 /** Refine by the L2 error estimate.
  *
  * Given the maximum global error, we enforce that each quadrant's portion of
  * the error must not exceed is fraction of the total volume of the domain
  * (which is 1).
  *
  * This function matches the p4est_refine_t prototype that is used by
  * p4est_refine() and p4est_refine_ext().
  *
  * \param [in] p4est          the forest
  * \param [in] which_tree     the tree in the forest containing \a q
  * \param [in] q              the quadrant
  *
  * \return 1 if \a q should be refined, 0 otherwise.
  */
 static int
 step3_refine_err_estimate (p4est_t * p4est, p4est_topidx_t which_tree,
                            p4est_quadrant_t * q)
 {
   step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
   double              global_err = ctx->max_err;
   double              global_err2 = global_err * global_err;
   double              h =
     (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;
   double              vol, err2;
 
   /* the quadrant's volume is also its volume fraction */
 #ifdef P4_TO_P8
   vol = h * h * h;
 #else
   vol = h * h;
 #endif
 
   err2 = step3_error_sqr_estimate (q);
   if (err2 > (global_err2 * vol * difficulty)) {
   //if (err2 > (global_err2 * vol)) {
     return 1;
   }
   else {
     return 0;
   }
 }
 
 /** Coarsen by the L2 error estimate of the initial condition.
  *
  * Given the maximum global error, we enforce that each quadrant's portion of
  * the error must not exceed is fraction of the total volume of the domain
  * (which is 1).
  *
  * \param [in] p4est          the forest
  * \param [in] which_tree     the tree in the forest containing \a children
  * \param [in] children       a family of quadrants
  *
  * \return 1 if \a children should be coarsened, 0 otherwise.
  */
 static int
 step3_coarsen_initial_condition (p4est_t * p4est,
                                  p4est_topidx_t which_tree,
                                  p4est_quadrant_t * children[])
 {
   p4est_quadrant_t    parent;
   step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
   double              global_err = ctx->max_err;
   double              global_err2 = global_err * global_err;
   double              h;
   step3_data_t        parentdata;
   double              parentmidpoint[3];
   double              vol, err2;
 
   /* get the parent of the first child (the parent of all children) */
   p4est_quadrant_parent (children[0], &parent);
   step3_get_midpoint (p4est, which_tree, &parent, parentmidpoint);
   parentdata.u = step3_initial_condition (parentmidpoint, parentdata.du, ctx);
   h = (double) P4EST_QUADRANT_LEN (parent.level) / (double) P4EST_ROOT_LEN;
   /* the quadrant's volume is also its volume fraction */
 #ifdef P4_TO_P8
   vol = h * h * h;
 #else
   vol = h * h;
 #endif
   parent.p.user_data = (void *) (&parentdata);
 
   err2 = step3_error_sqr_estimate (&parent);
   if (err2 < global_err2 * vol * difficulty) {
     return 1;
   }
   else {
     return 0;
   }
 }
 
 /** Coarsen by the L2 error estimate of the current state approximation.
  *
  * Given the maximum global error, we enforce that each quadrant's portion of
  * the error must not exceed its fraction of the total volume of the domain
  * (which is 1).
  *
  * This function matches the p4est_coarsen_t prototype that is used by
  * p4est_coarsen() and p4est_coarsen_ext().
  *
  * \param [in] p4est          the forest
  * \param [in] which_tree     the tree in the forest containing \a children
  * \param [in] children       a family of quadrants
  *
  * \return 1 if \a children should be coarsened, 0 otherwise.
  */
 static int
 step3_coarsen_err_estimate (p4est_t * p4est,
                             p4est_topidx_t which_tree,
                             p4est_quadrant_t * children[])
 {
   step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
   double              global_err = ctx->max_err;
   double              global_err2 = global_err * global_err;
   double              h;
   step3_data_t       *data;
   double              vol, err2, childerr2;
   double              parentu;
   double              diff;
   int                 i;
 
   h =
     (double) P4EST_QUADRANT_LEN (children[0]->level) /
     (double) P4EST_ROOT_LEN;
   /* the quadrant's volume is also its volume fraction */
 #ifdef P4_TO_P8
   vol = h * h * h;
 #else
   vol = h * h;
 #endif
 
   /* compute the average */
   parentu = 0.;
   for (i = 0; i < P4EST_CHILDREN; i++) {
     data = (step3_data_t *) children[i]->p.user_data;
     parentu += data->u / P4EST_CHILDREN;
   }
 
   err2 = 0.;
   for (i = 0; i < P4EST_CHILDREN; i++) {
     childerr2 = step3_error_sqr_estimate (children[i]);
 
     if (childerr2 > global_err2 * vol* difficulty) {
       return 0;
     }
     err2 += step3_error_sqr_estimate (children[i]);
     diff = (parentu - data->u) * (parentu - data->u);
     err2 += diff * vol;
   }
   if (err2 < global_err2 * (vol * P4EST_CHILDREN) * difficulty) {
     return 1;
   }
   else {
     return 0;
   }
 }
 
 /** Initialize the state variables of incoming quadrants from outgoing
  * quadrants.
  *
  * The functions p4est_refine_ext(), p4est_coarsen_ext(), and
  * p4est_balance_ext() take as an argument a p4est_replace_t callback function,
  * which allows one to setup the quadrant data of incoming quadrants from the
  * data of outgoing quadrants, before the outgoing data is destroyed.  This
  * function matches the p4est_replace_t prototype.
  *
  * In this example, we linearly interpolate the state variable of a quadrant
  * that is refined to its children, and we average the midpoints of children
  * that are being coarsened to the parent.
  *
  * \param [in] p4est          the forest
  * \param [in] which_tree     the tree in the forest containing \a children
  * \param [in] num_outgoing   the number of quadrants that are being replaced:
  *                            either 1 if a quadrant is being refined, or
  *                            P4EST_CHILDREN if a family of children are being
  *                            coarsened.
  * \param [in] outgoing       the outgoing quadrants
  * \param [in] num_incoming   the number of quadrants that are being added:
  *                            either P4EST_CHILDREN if a quadrant is being refined, or
  *                            1 if a family of children are being
  *                            coarsened.
  * \param [in,out] incoming   quadrants whose data are initialized.
  */
 static void
 step3_replace_quads (p4est_t * p4est, p4est_topidx_t which_tree,
                      int num_outgoing,
                      p4est_quadrant_t * outgoing[],
                      int num_incoming, p4est_quadrant_t * incoming[])
 {
   step3_data_t       *parent_data, *child_data;
   int                 i, j;
   double              h;
   double              du_old, du_est;
 
   if (num_outgoing > 1) {
     /* this is coarsening */
     parent_data = (step3_data_t *) incoming[0]->p.user_data;
     parent_data->u = 0.;
     for (j = 0; j < P4EST_DIM; j++) {
       parent_data->du[j] = step3_invalid;
 
     }
     for (i = 0; i < P4EST_CHILDREN; i++) {
       child_data = (step3_data_t *) outgoing[i]->p.user_data;
       parent_data->u += child_data->u / P4EST_CHILDREN;
       for (j = 0; j < P4EST_DIM; j++) {
         du_old = parent_data->du[j];
         du_est = child_data->du[j];
 
         if (du_old == du_old) {
           if (du_est * du_old >= 0.) {
             if (fabs (du_est) < fabs (du_old)) {
               parent_data->du[j] = du_est;
             }
           }
           else {
             parent_data->du[j] = 0.;
           }
         }
         else {
           parent_data->du[j] = du_est;
         }
       }
     }
   }
   else {
     /* this is refinement */
     parent_data = (step3_data_t *) outgoing[0]->p.user_data;
     h =
       (double) P4EST_QUADRANT_LEN (outgoing[0]->level) /
       (double) P4EST_ROOT_LEN;
 
     for (i = 0; i < P4EST_CHILDREN; i++) {
       child_data = (step3_data_t *) incoming[i]->p.user_data;
       child_data->u = parent_data->u;
       for (j = 0; j < P4EST_DIM; j++) {
         child_data->du[j] = parent_data->du[j];
         child_data->u +=
           (h / 4.) * parent_data->du[j] * ((i & (1 << j)) ? 1. : -1);
       }
     }
   }
 }
 
 /** Callback function for interpolating the solution from quadrant midpoints to
  * corners.
  *
  * The function p4est_iterate() takes as an argument a p4est_iter_volume_t
  * callback function, which it executes at every local quadrant (see
  * p4est_iterate.h).  This function matches the p4est_iter_volume_t prototype.
  *
  * In this example, we use the callback function to interpolate the state
  * variable to the corners, and write those corners into an array so that they
  * can be written out.
  *
  * \param [in] info          the information about this quadrant that has been
  *                           populated by p4est_iterate()
  * \param [in,out] user_data the user_data that was given as an argument to
  *                           p4est_iterate: in this case, it points to the
  *                           array of corner values that we want to write.
  *                           The values for the corner of the quadrant
  *                           described by \a info are written during the
  *                           execution of the callback.
  */
 static void
 step3_interpolate_solution (p4est_iter_volume_info_t * info, void *user_data)
 {
   sc_array_t         *u_interp = (sc_array_t *) user_data;      /* we passed the array of values to fill as the user_data in the call to p4est_iterate */
   p4est_t            *p4est = info->p4est;
   p4est_quadrant_t   *q = info->quad;
   p4est_topidx_t      which_tree = info->treeid;
   p4est_locidx_t      local_id = info->quadid;  /* this is the index of q *within its tree's numbering*.  We want to convert it its index for all the quadrants on this process, which we do below */
   p4est_tree_t       *tree;
   step3_data_t       *data = (step3_data_t *) q->p.user_data;
   double              h;
   p4est_locidx_t      arrayoffset;
   double              this_u;
   double             *this_u_ptr;
   int                 i, j;
 
   tree = p4est_tree_array_index (p4est->trees, which_tree);
   local_id += tree->quadrants_offset;   /* now the id is relative to the MPI process */
   arrayoffset = P4EST_CHILDREN * local_id;      /* each local quadrant has 2^d (P4EST_CHILDREN) values in u_interp */
   h = (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;
 
   for (i = 0; i < P4EST_CHILDREN; i++) {
     this_u = data->u;
     /* loop over the derivative components and linearly interpolate from the
      * midpoint to the corners */
     for (j = 0; j < P4EST_DIM; j++) {
       /* In order to know whether the direction from the midpoint to the corner is
        * negative or positive, we take advantage of the fact that the corners
        * are in z-order.  If i is an odd number, it is on the +x side; if it
        * is even, it is on the -x side.  If (i / 2) is an odd number, it is on
        * the +y side, etc. */
       this_u += (h / 2) * data->du[j] * ((i & (1 << j)) ? 1. : -1.);
     }
     this_u_ptr = (double *) sc_array_index (u_interp, arrayoffset + i);
     this_u_ptr[0] = this_u;
   }
 }

 __device__ void step3_cuda_interpolate_solution (
  p4est_t            *p4est,
  p4est_ghost_t      *ghost_layer,
  p4est_quadrant_t   *quad,
  void               *quad_data,
  p4est_locidx_t      quadid,
  p4est_topidx_t      treeid,
  void *user_data
) {
  sc_array_t         *u_interp = (sc_array_t *) user_data;      /* we passed the array of values to fill as the user_data in the call to p4est_iterate */
  p4est_quadrant_t   *q = quad;
  p4est_topidx_t      which_tree = treeid;
  p4est_locidx_t      local_id = quadid;  /* this is the index of q *within its tree's numbering*.  We want to convert it its index for all the quadrants on this process, which we do below */
  p4est_tree_t       *tree;
  step3_data_t       *data = (step3_data_t *) quad_data;
  double              h;
  p4est_locidx_t      arrayoffset;
  double              this_u;
  double             *this_u_ptr;
  int                 i, j;

  tree = p4est_device_tree_array_index (p4est->trees, which_tree);
  local_id += tree->quadrants_offset;   /* now the id is relative to the MPI process */
  arrayoffset = P4EST_DEVICE_CHILDREN * local_id;      /* each local quadrant has 2^d (P4EST_CHILDREN) values in u_interp */
  h = (double) P4EST_DEVICE_QUADRANT_LEN (q->level) / (double) P4EST_DEVICE_ROOT_LEN;

  for (i = 0; i < P4EST_DEVICE_CHILDREN; i++) {
    this_u = data->u;
    /* loop over the derivative components and linearly interpolate from the
     * midpoint to the corners */
    for (j = 0; j < P4EST_DEVICE_DIM; j++) {
      /* In order to know whether the direction from the midpoint to the corner is
       * negative or positive, we take advantage of the fact that the corners
       * are in z-order.  If i is an odd number, it is on the +x side; if it
       * is even, it is on the -x side.  If (i / 2) is an odd number, it is on
       * the +y side, etc. */
      this_u += (h / 2) * data->du[j] * ((i & (1 << j)) ? 1. : -1.);
    }
    this_u_ptr = (double *) sc_device_array_index (u_interp, arrayoffset + i);
    this_u_ptr[0] = this_u;
  }
}

__global__ void setup_step3_cuda_interpolate_solution_kernel(cuda_iter_volume_t *callback) {
 *callback = step3_cuda_interpolate_solution;
}

void step3_u_interp_alloc_cuda_memory(user_data_for_cuda_t* user_data_api) {
  step3_compute_max_user_data_to_cuda_t *user_data_to_cuda = (step3_compute_max_user_data_to_cuda_t*) malloc(sizeof(step3_compute_max_user_data_to_cuda_t));
  sc_array_t *user_data = (sc_array_t*) user_data_api->user_data;
  user_data_api->cuda_memory_allocating_info = (void*) scArrayMemoryAllocate(user_data);
}
void step3_u_interp_free_cuda_memory(user_data_for_cuda_t* user_data_api) {
  sc_array_cuda_memory_allocate_info_t *allocate_info = (sc_array_cuda_memory_allocate_info_t*) user_data_api->cuda_memory_allocating_info;
  scArrayMemoryFree(allocate_info);
}
void* step3_u_interp_get_cuda_allocated_user_data(user_data_for_cuda_t* user_data_api) {
  sc_array_cuda_memory_allocate_info_t *allocate_info = (sc_array_cuda_memory_allocate_info_t*) user_data_api->cuda_memory_allocating_info;
  return (void*) allocate_info->d_sc_arr;
}

void step3_u_interp_copy_user_data_from_device(user_data_for_cuda_t* user_data_api) {
  sc_array_cuda_memory_allocate_info_t *allocate_info = (sc_array_cuda_memory_allocate_info_t*) user_data_api->cuda_memory_allocating_info;
  sc_array_t *user_data = (sc_array_t*) user_data_api->user_data;
  gpuErrchk(cudaMemcpy(user_data->array, allocate_info->d_sc_arr_temp, allocate_info->d_sc_arr_temp_size, cudaMemcpyDeviceToHost));
}

 
 /** Write the state variable to vtk format, one file per process.
  *
  * \param [in] p4est    the forest, whose quadrant data contains the state
  * \param [in] timestep the timestep number, used to name the output files
  */
 static void
 step3_write_solution (cuda4est_t * cuda4est, int timestep)
 {
   p4est_t *p4est = cuda4est->p4est;
   char                filename[BUFSIZ] = "";
   int                 retval;
   sc_array_t         *u_interp;
   p4est_locidx_t      numquads;
   p4est_vtk_context_t *context;
 
   snprintf (filename, BUFSIZ, P4EST_STRING "_step3_%04d", timestep);
 
   numquads = p4est->local_num_quadrants;
 
   /* create a vector with one value for the corner of every local quadrant
    * (the number of children is always the same as the number of corners) */
   u_interp = sc_array_new_size (sizeof (double), numquads * P4EST_CHILDREN);

   cuda_iter_volume_api_t *step3_cuda_u_interpolate_solution_api = (cuda_iter_volume_api_t*)malloc(sizeof(cuda_iter_volume_api_t));
   step3_cuda_u_interpolate_solution_api->callback = step3_cuda_interpolate_solution;
   step3_cuda_u_interpolate_solution_api->setup_kernel = setup_step3_cuda_interpolate_solution_kernel;

   user_data_for_cuda_t *step3_user_data_api_u_interp = (user_data_for_cuda_t*)malloc(sizeof(user_data_for_cuda_t));
   step3_user_data_api_u_interp->user_data = (void*)u_interp;
   step3_user_data_api_u_interp->alloc_cuda_memory = step3_u_interp_alloc_cuda_memory;
   step3_user_data_api_u_interp->free_cuda_memory = step3_u_interp_free_cuda_memory;
   step3_user_data_api_u_interp->get_cuda_allocated_user_data = step3_u_interp_get_cuda_allocated_user_data;
   step3_user_data_api_u_interp->copy_user_data_from_device = step3_u_interp_copy_user_data_from_device;
 
   /* Use the iterator to visit every cell and fill in the solution values.
    * Using the iterator is not absolutely necessary in this case: we could
    * also loop over every tree (there is only one tree in this case) and loop
    * over every quadrant within every tree, but we are trying to demonstrate
    * the usage of p4est_iterate in this example */
    /*
   p4est_iterate (p4est, NULL,   
                  (void *) u_interp,
                  step3_interpolate_solution,
                  NULL,          
 #ifdef P4_TO_P8
                  NULL,          
 #endif
                  NULL);
  */
  cuda_iterate (cuda4est, NULL,
                (void *) u_interp, 
                step3_user_data_api_u_interp,
                step3_interpolate_solution,
                step3_cuda_u_interpolate_solution_api,
                NULL,
                NULL,   
          #ifdef P4_TO_P8
                NULL,    
          #endif
                NULL
              );         
   /* create VTK output context and set its parameters */
   context = p4est_vtk_context_new (p4est, filename);
   p4est_vtk_context_set_scale (context, 0.99);  /* quadrant at almost full scale */
 
   /* begin writing the output files */
   context = p4est_vtk_write_header (context);
   SC_CHECK_ABORT (context != NULL,
                   P4EST_STRING "_vtk: Error writing vtk header");
 
   /* do not write the tree id's of each quadrant
    * (there is only one tree in this example) */
   context = p4est_vtk_write_cell_dataf (context, 0, 1,  /* do write the refinement level of each quadrant */
                                         1,      /* do write the mpi process id of each quadrant */
                                         0,      /* do not wrap the mpi rank (if this were > 0, the modulus of the rank relative to this number would be written instead of the rank) */
                                         0,      /* there is no custom cell scalar data. */
                                         0,      /* there is no custom cell vector data. */
                                         context);       /* mark the end of the variable cell data. */
   SC_CHECK_ABORT (context != NULL,
                   P4EST_STRING "_vtk: Error writing cell data");
 
   /* write one scalar field: the solution value */
   context = p4est_vtk_write_point_dataf (context, 1, 0, /* write no vector fields */
                                          "solution", u_interp, context);        /* mark the end of the variable cell data. */
   SC_CHECK_ABORT (context != NULL,
                   P4EST_STRING "_vtk: Error writing cell data");
 
   retval = p4est_vtk_write_footer (context);
   SC_CHECK_ABORT (!retval, P4EST_STRING "_vtk: Error writing footer");
 
   sc_array_destroy (u_interp);
 }
 
 /** Approximate the divergence of (vu) on each quadrant
  *
  * We use piecewise constant approximations on each quadrant, so the value is
  * always 0.
  *
  * Like step3_interpolate_solution(), this function matches the
  * p4est_iter_volume_t prototype used by p4est_iterate().
  *
  * \param [in] info          the information about the quadrant populated by
  *                           p4est_iterate()
  * \param [in] user_data     not used
  */
 static void
 step3_quad_divergence (p4est_iter_volume_info_t * info, void *user_data)
 {
   p4est_quadrant_t   *q = info->quad;
   step3_data_t       *data = (step3_data_t *) q->p.user_data;
   data->dudt = 0.;
 }
 
 /** Approximate the flux across a boundary between quadrants.
  *
  * We use a very simple upwind numerical flux.
  *
  * This function matches the p4est_iter_face_t prototype used by
  * p4est_iterate().
  *
  * \param [in] info the information about the quadrants on either side of the
  *                  interface, populated by p4est_iterate()
  * \param [in] user_data the user_data given to p4est_iterate(): in this case,
  *                       it points to the ghost_data array, which contains the
  *                       step3_data_t data for all of the ghost cells, which
  *                       was populated by p4est_ghost_exchange_data()
  */
 static void
 step3_upwind_flux (p4est_iter_face_info_t * info, void *user_data)
 {
   int                 i, j;
   p4est_t            *p4est = info->p4est;
   step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
   step3_data_t       *ghost_data = (step3_data_t *) user_data;
   step3_data_t       *udata;
   p4est_quadrant_t   *quad;
   double              vdotn = 0.;
   double              uavg;
   double              q;
   double              h, facearea;
   int                 which_face;
   int                 upwindside;
   p4est_iter_face_side_t *side[2];
   sc_array_t         *sides = &(info->sides);
 
   /* because there are no boundaries, every face has two sides */
   P4EST_ASSERT (sides->elem_count == 2);
 
   side[0] = p4est_iter_fside_array_index_int (sides, 0);
   side[1] = p4est_iter_fside_array_index_int (sides, 1);
 
   /* which of the quadrant's faces the interface touches */
   which_face = side[0]->face;
 
   switch (which_face) {
   case 0:                      /* -x side */
     vdotn = -ctx->v[0];
     break;
   case 1:                      /* +x side */
     vdotn = ctx->v[0];
     break;
   case 2:                      /* -y side */
     vdotn = -ctx->v[1];
     break;
   case 3:                      /* +y side */
     vdotn = ctx->v[1];
     break;
 #ifdef P4_TO_P8
   case 4:                      /* -z side */
     vdotn = -ctx->v[2];
     break;
   case 5:                      /* +z side */
     vdotn = ctx->v[2];
     break;
 #endif
   }
   upwindside = vdotn >= 0. ? 0 : 1;
 
   /* Because we have non-conforming boundaries, one side of an interface can
    * either have one large ("full") quadrant or 2^(d-1) small ("hanging")
    * quadrants: we have to compute the average differently in each case.  The
    * info populated by p4est_iterate() gives us the context we need to
    * proceed. */
   uavg = 0;
   if (side[upwindside]->is_hanging) {
     /* there are 2^(d-1) (P4EST_HALF) subfaces */
     for (j = 0; j < P4EST_HALF; j++) {
       if (side[upwindside]->is.hanging.is_ghost[j]) {
         /* *INDENT-OFF* */
         udata =
           (step3_data_t *) &ghost_data[side[upwindside]->is.hanging.quadid[j]];
         /* *INDENT-ON* */
       }
       else {
         udata =
           (step3_data_t *) side[upwindside]->is.hanging.quad[j]->p.user_data;
       }
       uavg += udata->u;
     }
     uavg /= P4EST_HALF;
   }
   else {
     if (side[upwindside]->is.full.is_ghost) {
       udata = (step3_data_t *) & ghost_data[side[upwindside]->is.full.quadid];
     }
     else {
       udata = (step3_data_t *) side[upwindside]->is.full.quad->p.user_data;
     }
     uavg = udata->u;
   }
   /* flux from side 0 to side 1 */
   q = vdotn * uavg;
   for (i = 0; i < 2; i++) {
     if (side[i]->is_hanging) {
       /* there are 2^(d-1) (P4EST_HALF) subfaces */
       for (j = 0; j < P4EST_HALF; j++) {
         quad = side[i]->is.hanging.quad[j];
         h =
           (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
 #ifndef P4_TO_P8
         facearea = h;
 #else
         facearea = h * h;
 #endif
         if (!side[i]->is.hanging.is_ghost[j]) {
           udata = (step3_data_t *) quad->p.user_data;
           if (i == upwindside) {
             udata->dudt += vdotn * udata->u * facearea * (i ? 1. : -1.);
           }
           else {
             udata->dudt += q * facearea * (i ? 1. : -1.);
           }
         }
       }
     }
     else {
       quad = side[i]->is.full.quad;
       h = (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
 #ifndef P4_TO_P8
       facearea = h;
 #else
       facearea = h * h;
 #endif
       if (!side[i]->is.full.is_ghost) {
         udata = (step3_data_t *) quad->p.user_data;
         udata->dudt += q * facearea * (i ? 1. : -1.);
       }
     }
   }
 }
 
 /** Compute the new value of the state from the computed time derivative.
  *
  * We use a simple forward Euler scheme.
  *
  * The derivative was computed by a p4est_iterate() loop by the callbacks
  * step3_quad_divergence() and step3_upwind_flux(). Now we multiply this by
  * the timestep and add to the current solution.
  *
  * This function matches the p4est_iter_volume_t prototype used by
  * p4est_iterate().
  *
  * \param [in] info          the information about this quadrant that has been
  *                           populated by p4est_iterate()
  * \param [in] user_data the user_data given to p4est_iterate(): in this case,
  *                       it points to the timestep.
  */
 static void
 step3_timestep_update (p4est_iter_volume_info_t * info, void *user_data)
 {
   p4est_quadrant_t   *q = info->quad;
   step3_data_t       *data = (step3_data_t *) q->p.user_data;
   double              dt = *((double *) user_data);
   double              vol;
   double              h =
     (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;
 
 #ifdef P4_TO_P8
   vol = h * h * h;
 #else
   vol = h * h;
 #endif
 
   data->u += dt * data->dudt / vol;
 }
 
 /** Reset the approximate derivatives.
  *
  * p4est_iterate() has an invariant to the order of callback execution: the
  * p4est_iter_volume_t callback will be executed on a quadrant before the
  * p4est_iter_face_t callbacks are executed on its faces.  This function
  * resets the derivative stored in the quadrant's data before
  * step3_minmod_estimate() updates the derivative based on the face neighbors.
  *
  * This function matches the p4est_iter_volume_t prototype used by
  * p4est_iterate().
  *
  * \param [in] info          the information about this quadrant that has been
  *                           populated by p4est_iterate()
  * \param [in] user_data     not used
  */
 static void
 step3_reset_derivatives (p4est_iter_volume_info_t * info, void *user_data)
 {
   p4est_quadrant_t   *q = info->quad;
   step3_data_t       *data = (step3_data_t *) q->p.user_data;
   int                 j;
 
   for (j = 0; j < P4EST_DIM; j++) {
     data->du[j] = step3_invalid;
   }
 }

 // compute max
__device__ void step3_cuda_reset_derivatives (
  p4est_t            *p4est,
  p4est_ghost_t      *ghost_layer,
  p4est_quadrant_t   *quad,
  void               *quad_data,
  p4est_locidx_t      quadid,
  p4est_topidx_t      treeid,
  void *user_data
) {
//  p4est_quadrant_t   *q = quad;
  step3_data_t       *data = (step3_data_t *) quad_data;
  int                 j;

  //printf("step3_invalid: %f\n", step3_invalid);
  for (j = 0; j < P4EST_DIM; j++) {
    data->du[j] = step3_invalid;
  }
}

__global__ void setup_step3_cuda_reset_derivatives_kernel(cuda_iter_volume_t *callback) {
 *callback = step3_cuda_reset_derivatives;
}
 
 /** For two quadrants on either side of a face, estimate the derivative normal
  * to the face.
  *
  * This function matches the p4est_iter_face_t prototype used by
  * p4est_iterate().
  *
  * \param [in] info          the information about this quadrant that has been
  *                           populated by p4est_iterate()
  * \param [in] user_data the user_data given to p4est_iterate(): in this case,
  *                       it points to the ghost_data array, which contains the
  *                       step3_data_t data for all of the ghost cells, which
  *                       was populated by p4est_ghost_exchange_data()
  */
 static void
 step3_minmod_estimate (p4est_iter_face_info_t * info, void *user_data)
 {
   int                 i, j;
   p4est_iter_face_side_t *side[2];
   sc_array_t         *sides = &(info->sides);
   step3_data_t       *ghost_data = (step3_data_t *) user_data;
   step3_data_t       *udata;
   p4est_quadrant_t   *quad;
   double              uavg[2];
   double              h[2];
   double              du_est, du_old;
   int                 which_dir;
 
   /* because there are no boundaries, every face has two sides */
   P4EST_ASSERT (sides->elem_count == 2);
 
   side[0] = p4est_iter_fside_array_index_int (sides, 0);
   side[1] = p4est_iter_fside_array_index_int (sides, 1);
 
   which_dir = side[0]->face / 2;        /* 0 == x, 1 == y, 2 == z */
 
   for (i = 0; i < 2; i++) {
     uavg[i] = 0;
     if (side[i]->is_hanging) {
       /* there are 2^(d-1) (P4EST_HALF) subfaces */
       for (j = 0; j < P4EST_HALF; j++) {
         quad = side[i]->is.hanging.quad[j];
         h[i] =
           (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
         if (side[i]->is.hanging.is_ghost[j]) {
           udata = &ghost_data[side[i]->is.hanging.quadid[j]];
         }
         else {
           udata = (step3_data_t *) side[i]->is.hanging.quad[j]->p.user_data;
         }
         uavg[i] += udata->u;
       }
       uavg[i] /= P4EST_HALF;
     }
     else {
       quad = side[i]->is.full.quad;
       h[i] =
         (double) P4EST_QUADRANT_LEN (quad->level) / (double) P4EST_ROOT_LEN;
       if (side[i]->is.full.is_ghost) {
         udata = &ghost_data[side[i]->is.full.quadid];
       }
       else {
         udata = (step3_data_t *) side[i]->is.full.quad->p.user_data;
       }
       uavg[i] = udata->u;
     }
   }
   du_est = (uavg[1] - uavg[0]) / ((h[0] + h[1]) / 2.);
   for (i = 0; i < 2; i++) {
     if (side[i]->is_hanging) {
       /* there are 2^(d-1) (P4EST_HALF) subfaces */
       for (j = 0; j < P4EST_HALF; j++) {
         quad = side[i]->is.hanging.quad[j];
         if (!side[i]->is.hanging.is_ghost[j]) {
           udata = (step3_data_t *) quad->p.user_data;
           du_old = udata->du[which_dir];
           if (du_old == du_old) {
             /* there has already been an update */
             if (du_est * du_old >= 0.) {
               if (fabs (du_est) < fabs (du_old)) {
                 udata->du[which_dir] = du_est;
               }
             }
             else {
               udata->du[which_dir] = 0.;
             }
           }
           else {
             udata->du[which_dir] = du_est;
           }
         }
       }
     }
     else {
       quad = side[i]->is.full.quad;
       if (!side[i]->is.full.is_ghost) {
         udata = (step3_data_t *) quad->p.user_data;
         du_old = udata->du[which_dir];
         if (du_old == du_old) {
           /* there has already been an update */
           if (du_est * du_old >= 0.) {
             if (fabs (du_est) < fabs (du_old)) {
               udata->du[which_dir] = du_est;
             }
           }
           else {
             udata->du[which_dir] = 0.;
           }
         }
         else {
           udata->du[which_dir] = du_est;
         }
       }
     }
   }
 }

 __device__ void
 step3_cuda_minmod_estimate (
  p4est_t* p4est,
  p4est_ghost_t* ghost_layer,
  p4est_iter_face_side_t* side,
  void *user_data)
 {
  int                 i, j;
  //p4est_iter_face_side_t *side[2];
  //sc_array_t         *sides = &(info->sides);
  step3_data_t       *ghost_data = (step3_data_t *) user_data;
  step3_data_t       *udata;
  p4est_quadrant_t   *quad;
  double              uavg[2];
  double              h[2];
  double              du_est, du_old;
  int                 which_dir;

  /* because there are no boundaries, every face has two sides */
  //P4EST_ASSERT (sides->elem_count == 2);

  //side[0] = p4est_iter_fside_array_index_int (sides, 0);
  //side[1] = p4est_iter_fside_array_index_int (sides, 1);

  which_dir = side[0].face / 2;        /* 0 == x, 1 == y, 2 == z */

  for (i = 0; i < 2; i++) {
    uavg[i] = 0;
    if (side[i].is_hanging) {
      /* there are 2^(d-1) (P4EST_HALF) subfaces */
      for (j = 0; j < P4EST_DEVICE_HALF; j++) {
        quad = side[i].is.hanging.quad[j];
        h[i] =
          (double) P4EST_DEVICE_QUADRANT_LEN (quad->level) / (double) P4EST_DEVICE_ROOT_LEN;
        if (side[i].is.hanging.is_ghost[j]) {
          udata = &ghost_data[side[i].is.hanging.quadid[j]];
        }
        else {
          udata = (step3_data_t *) side[i].is.hanging.quad[j]->p.user_data;
        }
        uavg[i] += udata->u;
      }
      uavg[i] /= P4EST_DEVICE_HALF;
    }
    else {
      quad = side[i].is.full.quad;
      h[i] =
        (double) P4EST_DEVICE_QUADRANT_LEN (quad->level) / (double) P4EST_DEVICE_ROOT_LEN;
      if (side[i].is.full.is_ghost) {
        udata = &ghost_data[side[i].is.full.quadid];
      }
      else {
        udata = (step3_data_t *) side[i].is.full.quad->p.user_data;
      }
      uavg[i] = udata->u;
    }
  }
  du_est = (uavg[1] - uavg[0]) / ((h[0] + h[1]) / 2.);
  for (i = 0; i < 2; i++) {
    if (side[i].is_hanging) {
      /* there are 2^(d-1) (P4EST_DEVICE_HALF) subfaces */
      for (j = 0; j < P4EST_DEVICE_HALF; j++) {
        quad = side[i].is.hanging.quad[j];
        if (!side[i].is.hanging.is_ghost[j]) {
          udata = (step3_data_t *) quad->p.user_data;
          du_old = udata->du[which_dir];
          if (du_old == du_old) {
            /* there has already been an update */
            if (du_est * du_old >= 0.) {
              if (fabs (du_est) < fabs (du_old)) {
                udata->du[which_dir] = du_est;
              }
            }
            else {
              udata->du[which_dir] = 0.;
            }
          }
          else {
            udata->du[which_dir] = du_est;
          }
        }
      }
    }
    else {
      quad = side[i].is.full.quad;
      if (!side[i].is.full.is_ghost) {
        udata = (step3_data_t *) quad->p.user_data;
        du_old = udata->du[which_dir];
        if (du_old == du_old) {
          /* there has already been an update */
          if (du_est * du_old >= 0.) {
            if (fabs (du_est) < fabs (du_old)) {
              udata->du[which_dir] = du_est;
            }
          }
          else {
            udata->du[which_dir] = 0.;
          }
        }
        else {
          udata->du[which_dir] = du_est;
        }
      }
    }
  }
}
__global__ void setup_step3_cuda_minmod_estimate_kernel(cuda_iter_face_t *callback) {
  *callback = step3_cuda_minmod_estimate;
}
 
 /** Compute the maximum state value.
  *
  * This function updates the maximum value from the value of a single cell.
  *
  * This function matches the p4est_iter_volume_t prototype used by
  * p4est_iterate().
  *
  * \param [in] info              the information about this quadrant that has been
  *                               populated by p4est_iterate()
  * \param [in,out] user_data     the user_data given to p4est_iterate(): in this case,
  *                               it points to the maximum value that will be updated
  */
 static void
 step3_compute_max (p4est_iter_volume_info_t * info, void *user_data)
 {
   p4est_quadrant_t   *q = info->quad;
   step3_data_t       *data = (step3_data_t *) q->p.user_data;
   double              umax = *((double *) user_data);
 
   umax = SC_MAX (data->u, umax);
 
   *((double *) user_data) = umax;
 }

 __device__ static double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_i = (unsigned long long int*) address;
    unsigned long long int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// compute max
__device__ void step3_cuda_compute_max (
   p4est_t            *p4est,
   p4est_ghost_t      *ghost_layer,
   p4est_quadrant_t   *quad,
   void               *quad_data,
   p4est_locidx_t      quadid,
   p4est_topidx_t      treeid,
   void *user_data
) {
   //p4est_quadrant_t   *q = quad;
   step3_data_t       *data = (step3_data_t *) quad_data;
   // i don't know
   printf("");
   atomicMax((double *)user_data, data->u);
}

__global__ void setup_step3_cuda_compute_max_kernel(cuda_iter_volume_t *callback) {
  *callback = step3_cuda_compute_max;
}

void step3_compute_max_alloc_cuda_memory(user_data_for_cuda_t* user_data_api) {
  step3_compute_max_user_data_to_cuda_t *user_data_to_cuda = (step3_compute_max_user_data_to_cuda_t*) malloc(sizeof(step3_compute_max_user_data_to_cuda_t));
  double *user_data = (double*) user_data_api->user_data;

  double *d_compute_max_user_data;
  gpuErrchk(cudaMalloc((void**)&d_compute_max_user_data, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_compute_max_user_data, user_data, sizeof(double), cudaMemcpyHostToDevice));
  user_data_to_cuda->d_user_data = d_compute_max_user_data;

  user_data_api->cuda_memory_allocating_info = user_data_to_cuda;
}
void step3_compute_max_free_cuda_memory(user_data_for_cuda_t* user_data_api) {
  step3_compute_max_user_data_to_cuda_t *allocate_info = (step3_compute_max_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
  gpuErrchk(cudaFree(allocate_info->d_user_data));
}
void* step3_compute_max_get_cuda_allocated_user_data(user_data_for_cuda_t* user_data_api) {
  step3_compute_max_user_data_to_cuda_t *allocate_info = (step3_compute_max_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
  return (void*) allocate_info->d_user_data;
}

void step3_compute_max_copy_user_data_from_device(user_data_for_cuda_t* user_data_api) {
  step3_compute_max_user_data_to_cuda_t *allocate_info = (step3_compute_max_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
  gpuErrchk(cudaMemcpy(user_data_api->user_data, allocate_info->d_user_data, sizeof(double), cudaMemcpyDeviceToHost));
}
// compute max

// timestep update

__device__ void step3_cuda_timestep_update (
  p4est_t            *p4est,
  p4est_ghost_t      *ghost_layer,
  p4est_quadrant_t   *quad,
  void               *quad_data,
  p4est_locidx_t      quadid,
  p4est_topidx_t      treeid,
  void *user_data
) {
  p4est_quadrant_t   *q = quad;
  step3_data_t       *data = (step3_data_t *) quad_data;

  double              dt = *((double *) user_data);
  double              vol;
  double              h =
    (double) P4EST_QUADRANT_LEN (q->level) / (double) P4EST_ROOT_LEN;

  #ifdef P4_TO_P8
    vol = h * h * h;
  #else
    vol = h * h;
  #endif

  data->u += dt * data->dudt / vol;
}

__global__ void setup_step3_cuda_timestep_update_kernel(cuda_iter_volume_t *callback) {
 *callback = step3_cuda_timestep_update;
}

__device__ void
step3_cuda_quad_divergence (
  p4est_t            *p4est,
  p4est_ghost_t      *ghost_layer,
  p4est_quadrant_t   *quad,
  void               *quad_data,
  p4est_locidx_t      quadid,
  p4est_topidx_t      treeid,
  void *user_data
)
{
  //p4est_quadrant_t   *q = quad;
  step3_data_t       *data = (step3_data_t *) quad_data;
  data->dudt = 0.;
}

__global__ void setup_step3_cuda_quad_divergence_kernel(cuda_iter_volume_t *callback) {
  *callback = step3_cuda_quad_divergence;
}

__device__ void
step3_new_cuda_quad_divergence (
  p4est_t* p4est,
  char* ctx,
  size_t output_quads_count,
  unsigned char *block_user_data,
  unsigned char local_id
)
{
  step3_data_t       *data = ((step3_data_t *) block_user_data) + local_id;
  data->dudt = 0.;
}

__global__ void setup_step3_cuda_new_quad_divergence_kernel(cuda_new_iter_quad_t *callback) {
  *callback = step3_new_cuda_quad_divergence;
}

__device__ void
 step3_cuda_upwind_flux (
  p4est_t* p4est,
  p4est_ghost_t* ghost_layer,
  p4est_iter_face_side_t* side,
  void *user_data)
 {

   int                 i, j;
   step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
   step3_data_t       *ghost_data = (step3_data_t *) user_data;
   step3_data_t       *udata;
   p4est_quadrant_t   *quad;
   double              vdotn = 0.;
   double              uavg;
   double              q;
   double              h, facearea;
   int                 which_face;
   int                 upwindside;
 
   /* because there are no boundaries, every face has two sides */
   //P4EST_ASSERT (sides->elem_count == 2);
 
 
   /* which of the quadrant's faces the interface touches */
   which_face = side[0].face;
 
   switch (which_face) {
   case 0:                      /* -x side */
     vdotn = -ctx->v[0];
     break;
   case 1:                      /* +x side */
     vdotn = ctx->v[0];
     break;
   case 2:                      /* -y side */
     vdotn = -ctx->v[1];
     break;
   case 3:                      /* +y side */
     vdotn = ctx->v[1];
     break;
 #ifdef P4_TO_P8
   case 4:                      /* -z side */
     vdotn = -ctx->v[2];
     break;
   case 5:                      /* +z side */
     vdotn = ctx->v[2];
     break;
 #endif
   }
   upwindside = vdotn >= 0. ? 0 : 1;
 
   /* Because we have non-conforming boundaries, one side of an interface can
    * either have one large ("full") quadrant or 2^(d-1) small ("hanging")
    * quadrants: we have to compute the average differently in each case.  The
    * info populated by p4est_iterate() gives us the context we need to
    * proceed. */
   uavg = 0;
   if (side[upwindside].is_hanging) {
     /* there are 2^(d-1) (P4EST_HALF) subfaces */
     for (j = 0; j < P4EST_DEVICE_HALF; j++) {
       if (side[upwindside].is.hanging.is_ghost[j]) {
         /* *INDENT-OFF* */
         udata =
           (step3_data_t *) &ghost_data[side[upwindside].is.hanging.quadid[j]];
         /* *INDENT-ON* */
       }
       else {
         udata =
           (step3_data_t *) side[upwindside].is.hanging.quad[j]->p.user_data;
       }
       uavg += udata->u;
     }
     uavg /= P4EST_DEVICE_HALF;
   }
   else {
     if (side[upwindside].is.full.is_ghost) {
       udata = (step3_data_t *) & ghost_data[side[upwindside].is.full.quadid];
     }
     else {
       udata = (step3_data_t *) side[upwindside].is.full.quad->p.user_data;
     }
     uavg = udata->u;
   }
   /* flux from side 0 to side 1 */
   q = vdotn * uavg;
   for (i = 0; i < 2; i++) {
     if (side[i].is_hanging) {
       /* there are 2^(d-1) (P4EST_HALF) subfaces */
       for (j = 0; j < P4EST_DEVICE_HALF; j++) {
         quad = side[i].is.hanging.quad[j];
         h =
           (double) P4EST_DEVICE_QUADRANT_LEN (quad->level) / (double) P4EST_DEVICE_ROOT_LEN;
 #ifndef P4_TO_P8
         facearea = h;
 #else
         facearea = h * h;
 #endif
         if (!side[i].is.hanging.is_ghost[j]) {
           udata = (step3_data_t *) quad->p.user_data;
           if (i == upwindside) {
            udata->dudt += vdotn * udata->u * facearea * (i ? 1. : -1.);
           }
           else {
            udata->dudt += q * facearea * (i ? 1. : -1.);
           }
         }
       }
     }
     else {
       quad = side[i].is.full.quad;
       h = (double) P4EST_DEVICE_QUADRANT_LEN (quad->level) / (double) P4EST_DEVICE_ROOT_LEN;
 #ifndef P4_TO_P8
       facearea = h;
 #else
       facearea = h * h;
 #endif
       if (!side[i].is.full.is_ghost) {
         udata = (step3_data_t *) quad->p.user_data;
        udata->dudt += q * facearea * (i ? 1. : -1.);
       }
     }
   }
}
__global__ void setup_step3_cuda_upwind_flux_kernel(cuda_iter_face_t *callback) {
  *callback = step3_cuda_upwind_flux;
}


__device__ void
 step3_new_cuda_upwind_flux (
  p4est_t* p4est,
  char* d_ctx,
  size_t output_quads_count,
  unsigned char* block_user_data,
  cuda_next_light_face_side_t* sides)
 {
  step3_data_t *quads_user_data = (step3_data_t *)block_user_data;
   int                 i, j;
   step3_ctx_t        *ctx = (step3_ctx_t *) d_ctx;
   step3_data_t       *udata;
   unsigned char       quadid;
   double              vdotn = 0.;
   double              uavg;
   double              q;
   double              h, facearea;
   int                 which_face;
   int                 upwindside;
 

   which_face = sides[0].face;
 
   switch (which_face) {
   case 0:                      /* -x side */
     vdotn = -ctx->v[0];
     break;
   case 1:                      /* +x side */
     vdotn = ctx->v[0];
     break;
   case 2:                      /* -y side */
     vdotn = -ctx->v[1];
     break;
   case 3:                      /* +y side */
     vdotn = ctx->v[1];
     break;
 #ifdef P4_TO_P8
   case 4:                      /* -z side */
     vdotn = -ctx->v[2];
     break;
   case 5:                      /* +z side */
     vdotn = ctx->v[2];
     break;
 #endif
   }
   upwindside = vdotn >= 0. ? 0 : 1;
 
   /* Because we have non-conforming boundaries, one side of an interface can
    * either have one large ("full") quadrant or 2^(d-1) small ("hanging")
    * quadrants: we have to compute the average differently in each case.  The
    * info populated by p4est_iterate() gives us the context we need to
    * proceed. */
   uavg = 0;
   if (sides[upwindside].is_hanging) {
     /* there are 2^(d-1) (P4EST_HALF) subfaces */
     for (j = 0; j < P4EST_DEVICE_HALF; j++) {
        udata = (step3_data_t *) (quads_user_data + sides[upwindside].quadid[j]);
        //printf("[cuda]: quadid: %d udata->u: %f\n", sides[upwindside].is.hanging.quadid[j], udata->u);
        //printf("[cuda]: quadid: %lu\n", sides[upwindside].is.hanging.quadid[j]);
        //printf("[cuda]: quadid: %lu\n", sides[upwindside].quadid[j]);
        uavg += udata->u;
     }
     uavg /= P4EST_DEVICE_HALF;
   }
   else {
     udata = (step3_data_t *) (quads_user_data + sides[upwindside].quadid[0]);
     //printf("[cuda]: quadid: %d udata->u: %f\n", sides[upwindside].is.full.quadid, udata->u);
     uavg = udata->u;
   }
   /* flux from side 0 to side 1 */
   q = vdotn * uavg;
   for (i = 0; i < 2; i++) {
     if (sides[i].is_hanging) {
       /* there are 2^(d-1) (P4EST_HALF) subfaces */
       for (j = 0; j < P4EST_DEVICE_HALF; j++) {
        quadid = sides[i].quadid[j];
        h =
          (double) P4EST_DEVICE_QUADRANT_LEN (sides[i].levels[j]) / (double) P4EST_DEVICE_ROOT_LEN;
 #ifndef P4_TO_P8
         facearea = h;
 #else
         facearea = h * h;
 #endif
          udata = (step3_data_t *) (quads_user_data + quadid);
          if (i == upwindside) {
            udata->dudt += vdotn * udata->u * facearea * (i ? 1. : -1.);
          }
          else {
            udata->dudt += q * facearea * (i ? 1. : -1.);
          }
       }
     }
     else {
       quadid = sides[i].quadid[0];
       h = (double) P4EST_DEVICE_QUADRANT_LEN (sides[i].levels[0]) / (double) P4EST_DEVICE_ROOT_LEN;
 #ifndef P4_TO_P8
       facearea = h;
 #else
       facearea = h * h;
 #endif
      udata = (step3_data_t *) (quads_user_data + quadid);
      udata->dudt += q * facearea * (i ? 1. : -1.);
     }
   }
   //printf("[cuda] q: %f udata->dudt %f\n", q, udata->dudt);
}
__global__ void setup_step3_cuda_new_upwind_flux_kernel(cuda_new_iter_face_t *callback) {
  *callback = step3_new_cuda_upwind_flux;
}

void step3_timestep_update_alloc_cuda_memory(user_data_for_cuda_t* user_data_api) {
 step3_timestep_update_user_data_to_cuda_t *user_data_to_cuda = (step3_timestep_update_user_data_to_cuda_t*) malloc(sizeof(step3_timestep_update_user_data_to_cuda_t));
 double *user_data = (double*) user_data_api->user_data;

 double *d_timestep_update_user_data;
 gpuErrchk(cudaMalloc((void**)&d_timestep_update_user_data, sizeof(double)));
 gpuErrchk(cudaMemcpy(d_timestep_update_user_data, user_data, sizeof(double), cudaMemcpyHostToDevice));
 user_data_to_cuda->d_user_data = d_timestep_update_user_data;

 user_data_api->cuda_memory_allocating_info = user_data_to_cuda;
}

void step3_timestep_update_free_cuda_memory(user_data_for_cuda_t* user_data_api) {
 step3_timestep_update_user_data_to_cuda_t *allocate_info = (step3_timestep_update_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
 gpuErrchk(cudaFree(allocate_info->d_user_data));
}
void* step3_timestep_update_get_cuda_allocated_user_data(user_data_for_cuda_t* user_data_api) {
 step3_timestep_update_user_data_to_cuda_t *allocate_info = (step3_timestep_update_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
 return (void*) allocate_info->d_user_data;
}

void step3_timestep_update_copy_user_data_from_device(user_data_for_cuda_t* user_data_api) {
 step3_timestep_update_user_data_to_cuda_t *allocate_info = (step3_timestep_update_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
 gpuErrchk(cudaMemcpy(user_data_api->user_data, allocate_info->d_user_data, sizeof(double), cudaMemcpyDeviceToHost));
}


void step3_ghost_data_alloc_cuda_memory(user_data_for_cuda_t* user_data_api) {
  step3_ghost_data_user_data_to_cuda_t *user_data_to_cuda = (step3_ghost_data_user_data_to_cuda_t*) malloc(sizeof(step3_ghost_data_user_data_to_cuda_t));
  step3_data_t *user_data = (step3_data_t*) user_data_api->user_data;
 
  step3_data_t *d_ghost_data_user_data;
  size_t alloc_memory_size = user_data_api->user_data_elem_count * sizeof(step3_data_t);
  gpuErrchk(cudaMalloc((void**)&d_ghost_data_user_data, alloc_memory_size));
  gpuErrchk(cudaMemcpy(d_ghost_data_user_data, user_data, alloc_memory_size, cudaMemcpyHostToDevice));
  user_data_to_cuda->d_user_data = d_ghost_data_user_data;
 
  user_data_api->cuda_memory_allocating_info = user_data_to_cuda;
 }
 
 void step3_ghost_data_free_cuda_memory(user_data_for_cuda_t* user_data_api) {
  step3_ghost_data_user_data_to_cuda_t *allocate_info = (step3_ghost_data_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
  if(allocate_info->d_user_data) {
    gpuErrchk(cudaFree(allocate_info->d_user_data));
  }
 }
 void* step3_ghost_data_get_cuda_allocated_user_data(user_data_for_cuda_t* user_data_api) {
  step3_ghost_data_user_data_to_cuda_t *allocate_info = (step3_ghost_data_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
  return (void*) allocate_info->d_user_data;
 }
 
 void step3_ghost_data_copy_user_data_from_device(user_data_for_cuda_t* user_data_api) {
  if(user_data_api->user_data_elem_count) {
    step3_ghost_data_user_data_to_cuda_t *allocate_info = (step3_ghost_data_user_data_to_cuda_t*) user_data_api->cuda_memory_allocating_info;
    gpuErrchk(cudaMemcpy(user_data_api->user_data, allocate_info->d_user_data, sizeof(double), cudaMemcpyDeviceToHost));
  }
 }

// timestep update
 
 /** Compute the timestep.
  *
  * Find the smallest quadrant and scale the timestep based on that length and
  * the advection velocity.
  *
  * \param [in] p4est the forest
  * \return the timestep.
  */
 static double
 step3_get_timestep (p4est_t * p4est)
 {
   step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
   p4est_topidx_t      t, flt, llt;
   p4est_tree_t       *tree;
   int                 max_level, global_max_level;
   int                 mpiret, i;
   double              min_h, vnorm;
   double              dt;
 
   /* compute the timestep by finding the smallest quadrant */
   flt = p4est->first_local_tree;
   llt = p4est->last_local_tree;
 
   max_level = 0;
   for (t = flt; t <= llt; t++) {
     tree = p4est_tree_array_index (p4est->trees, t);
     max_level = SC_MAX (max_level, tree->maxlevel);
 
   }
   mpiret =
     sc_MPI_Allreduce (&max_level, &global_max_level, 1, sc_MPI_INT,
                       sc_MPI_MAX, p4est->mpicomm);
   SC_CHECK_MPI (mpiret);
 
   min_h =
     (double) P4EST_QUADRANT_LEN (global_max_level) / (double) P4EST_ROOT_LEN;
 
   vnorm = 0;
   for (i = 0; i < P4EST_DIM; i++) {
     vnorm += ctx->v[i] * ctx->v[i];
   }
   vnorm = sqrt (vnorm);
 
   dt = min_h / 2. / vnorm;
 
   return dt;
 }

 __device__ cuda_new_iter_face_t p_step3_new_cuda_upwind_flux = step3_new_cuda_upwind_flux;
 __device__ cuda_new_iter_quad_t p_step3_new_cuda_quad_divergence = step3_new_cuda_quad_divergence;
 
 /** Timestep the advection problem.
  *
  * Update the state, refine, repartition, and write the solution to file.
  *
  * \param [in,out] p4est the forest, whose state is updated
  * \param [in] time      the end time
  */
 static void
 step3_timestep (cuda4est_t *cuda4est, double time)
 {
   double ghost_allocation = 0;
   double p4est_reallocation = 0;
   double quadrants_reallocation = 0;
   double faces_reallocation = 0;
   double quadrants_blocks_reallocation = 0;
   double reset_derivatives_running = 0;
   double compute_max_running = 0;
   double flux_compute_running = 0;
   double new_flux_compute_running = 0;
   double timestep_update_running = 0;
   double downloading_quads = 0;

   bool quadrants_is_fresh = false;

   clock_t start = clock();
   clock_t stop = clock();
   double duration = (double)(stop - start) / CLOCKS_PER_SEC;
   p4est_t * p4est = cuda4est->p4est;
   double              t = 0.;
   double              dt = 0.;
   int                 i;
   step3_data_t       *ghost_data;
   step3_ctx_t        *ctx = (step3_ctx_t *) p4est->user_pointer;
   int                 refine_period = ctx->refine_period;
   int                 repartition_period = ctx->repartition_period;
   int                 write_period = ctx->write_period;
   int                 recursive = 0;
   int                 allowed_level = P4EST_QMAXLEVEL;
   int                 allowcoarsening = 1;
   int                 callbackorphans = 0;
   int                 mpiret;
   double              orig_max_err = ctx->max_err;
   double              umax, global_umax;
   p4est_ghost_t      *ghost;

   cuda_iter_volume_api_t *step3_cuda_compute_max_api = (cuda_iter_volume_api_t*)malloc(sizeof(cuda_iter_volume_api_t));
   step3_cuda_compute_max_api->callback = step3_cuda_compute_max;
   step3_cuda_compute_max_api->setup_kernel = setup_step3_cuda_compute_max_kernel;

   user_data_for_cuda_t *step3_user_data_api_compute_max = (user_data_for_cuda_t*) malloc(sizeof(user_data_for_cuda_t));
   step3_user_data_api_compute_max->user_data = &umax;
   step3_user_data_api_compute_max->alloc_cuda_memory = step3_compute_max_alloc_cuda_memory;
   step3_user_data_api_compute_max->free_cuda_memory = step3_compute_max_free_cuda_memory;
   step3_user_data_api_compute_max->get_cuda_allocated_user_data = step3_compute_max_get_cuda_allocated_user_data;
   step3_user_data_api_compute_max->copy_user_data_from_device = step3_compute_max_copy_user_data_from_device;

   cuda_iter_volume_api_t *step3_cuda_timestep_update_api = (cuda_iter_volume_api_t*)malloc(sizeof(cuda_iter_volume_api_t));
   step3_cuda_timestep_update_api->callback = step3_cuda_timestep_update;
   step3_cuda_timestep_update_api->setup_kernel = setup_step3_cuda_timestep_update_kernel;

   user_data_for_cuda_t *step3_user_data_api_timestep_update = (user_data_for_cuda_t*) malloc(sizeof(user_data_for_cuda_t));
   step3_user_data_api_timestep_update->user_data = &dt;
   step3_user_data_api_timestep_update->alloc_cuda_memory = step3_timestep_update_alloc_cuda_memory;
   step3_user_data_api_timestep_update->free_cuda_memory = step3_timestep_update_free_cuda_memory;
   step3_user_data_api_timestep_update->get_cuda_allocated_user_data = step3_timestep_update_get_cuda_allocated_user_data;
   step3_user_data_api_timestep_update->copy_user_data_from_device = step3_timestep_update_copy_user_data_from_device;

  cuda_iter_volume_api_t *step3_cuda_quad_divergence_api = (cuda_iter_volume_api_t*)malloc(sizeof(cuda_iter_volume_api_t));
  step3_cuda_quad_divergence_api->callback=  step3_cuda_quad_divergence;
  step3_cuda_quad_divergence_api->setup_kernel = setup_step3_cuda_quad_divergence_kernel;

  cuda_iter_face_api_t *step3_cuda_upwind_flux_api = (cuda_iter_face_api_t*)malloc(sizeof(cuda_iter_face_api_t));
  step3_cuda_upwind_flux_api->callback = step3_cuda_upwind_flux;
  step3_cuda_upwind_flux_api->setup_kernel = setup_step3_cuda_upwind_flux_kernel;

  user_data_for_cuda_t *step3_user_data_api_ghost_data = (user_data_for_cuda_t*)malloc(sizeof(user_data_for_cuda_t));
  step3_user_data_api_ghost_data->user_data_elem_count = 0;
  step3_user_data_api_ghost_data->alloc_cuda_memory = step3_ghost_data_alloc_cuda_memory;
  step3_user_data_api_ghost_data->free_cuda_memory = step3_ghost_data_free_cuda_memory;
  step3_user_data_api_ghost_data->get_cuda_allocated_user_data = step3_ghost_data_get_cuda_allocated_user_data;
  step3_user_data_api_ghost_data->copy_user_data_from_device = step3_ghost_data_copy_user_data_from_device;

  cuda_iter_volume_api_t *step3_cuda_reset_derivatives_api = (cuda_iter_volume_api_t*)malloc(sizeof(cuda_iter_volume_api_t));
  step3_cuda_reset_derivatives_api->callback = step3_cuda_reset_derivatives;
  step3_cuda_reset_derivatives_api->setup_kernel = setup_step3_cuda_reset_derivatives_kernel;

  cuda_iter_face_api_t *step3_cuda_minmod_estimate_api = (cuda_iter_face_api_t*)malloc(sizeof(cuda_iter_face_api_t));
  step3_cuda_minmod_estimate_api->callback = step3_cuda_minmod_estimate;
  step3_cuda_minmod_estimate_api->setup_kernel = setup_step3_cuda_minmod_estimate_kernel;

  cuda_new_iter_face_api_t *step3_new_cuda_upwind_flux_api = (cuda_new_iter_face_api_t*)malloc(sizeof(cuda_new_iter_face_api_t));
  //step3_new_cuda_upwind_flux_api->callback = step3_new_cuda_upwind_flux;
  step3_new_cuda_upwind_flux_api->setup_kernel = setup_step3_cuda_new_upwind_flux_kernel;
  gpuErrchk(cudaMemcpyFromSymbol(&(step3_new_cuda_upwind_flux_api->callback), p_step3_new_cuda_upwind_flux, sizeof(cuda_new_iter_face_t)));

  cuda_new_iter_quad_api_t *step3_new_cuda_quad_divergence_api = (cuda_new_iter_quad_api_t*)malloc(sizeof(cuda_new_iter_quad_api_t));
  step3_new_cuda_quad_divergence_api->setup_kernel = setup_step3_cuda_new_quad_divergence_kernel;
  gpuErrchk(cudaMemcpyFromSymbol(&(step3_new_cuda_quad_divergence_api->callback), p_step3_new_cuda_quad_divergence, sizeof(cuda_new_iter_quad_t)));

   /* create the ghost quadrants */
   ghost = p4est_ghost_new (p4est, P4EST_CONNECT_FULL);
   /* create space for storing the ghost data */
   ghost_data = P4EST_ALLOC (step3_data_t, ghost->ghosts.elem_count);
   /* synchronize the ghost data */
   p4est_ghost_exchange_data (p4est, ghost, ghost_data);
   start = clock();
   p4est_ghost_to_cuda_t* malloc_ghost = mallocForGhost(p4est, ghost);
   exchangeGhostDataToCuda(malloc_ghost, ghost);
   cuda4est->ghost_to_cuda = malloc_ghost;
   step3_user_data_api_ghost_data->user_data = ghost_data;
   step3_user_data_api_ghost_data->user_data_elem_count = ghost->ghosts.elem_count;
   stop = clock();
   duration = (double)(stop - start) / CLOCKS_PER_SEC; 
   ghost_allocation+=duration;
  // p4est memory allocation start
  start = clock();
  p4est_cuda_memory_allocate_info_t *p4est_memory_allocate_info = p4est_memory_alloc(cuda4est);
  cuda4est->p4est_memory_allocate_info = p4est_memory_allocate_info;
  stop = clock();
  duration = (double)(stop - start) / CLOCKS_PER_SEC; 
  p4est_reallocation+=duration;
  // p4est memory allocation end

  // quadrants memory allocation start
  sc_array_t         *trees = p4est->trees;
  p4est_tree_t       *tree;
  sc_array_t         *quadrants;
  start = clock();
  tree = p4est_tree_array_index (trees, p4est->first_local_tree);
  quadrants = &(tree->quadrants);
  p4est_quadrants_to_cuda_t *quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
  cuda4est->quads_to_cuda = quads_to_cuda;
  stop = clock();
  duration = (double)(stop - start) / CLOCKS_PER_SEC;
  quadrants_reallocation+=duration;

  start = clock();
  mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
  stop = clock();
  duration = (double)(stop - start) / CLOCKS_PER_SEC; 
  faces_reallocation+=duration;

  start = clock();
  mallocQuadrantsBlocks(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
  stop = clock();
  duration = (double)(stop - start) / CLOCKS_PER_SEC;
  quadrants_blocks_reallocation+=duration; 
  // quadrants memory allocation end
  
  start= clock();
  cuda_iterate (cuda4est, ghost,
    (void *) ghost_data, 
    step3_user_data_api_ghost_data,
    step3_reset_derivatives,
    step3_cuda_reset_derivatives_api,
    step3_minmod_estimate,
    step3_cuda_minmod_estimate_api,   
#ifdef P4_TO_P8
    NULL,    
#endif
    NULL);
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
    reset_derivatives_running+=duration;
  
   /* initialize du/dx estimates */
   /*
   p4est_iterate (p4est, ghost, (void *) ghost_data,
                  step3_reset_derivatives,
                  step3_minmod_estimate,
 #ifdef P4_TO_P8
                  NULL,
 #endif
                  NULL);
  // quadrants memory reallocation start
  freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
  tree = p4est_tree_array_index (trees, p4est->first_local_tree);
  quadrants = &(tree->quadrants);
  quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
  cuda4est->quads_to_cuda = quads_to_cuda;
  mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
  // quadrants memory reallocation end
  */
   quadrants_is_fresh = false;
   for (t = 0., i = 0; t < time; t += dt, i++) {
     P4EST_GLOBAL_PRODUCTIONF ("time %f\n", t);
 
     /* refine */
     if (!(i % refine_period)) {
       if (i) {
        start = clock();
        downloadQuadrantsFromCuda(quads_to_cuda, quadrants, cuda4est->quad_user_data_api);
        stop = clock();
        duration = (double)(stop - start) / CLOCKS_PER_SEC;
        downloading_quads+=duration; 
         /* compute umax */
         umax = 0.;
         /* initialize derivative estimates */
         //start = clock();
         start=clock();
         p4est_iterate (p4est, NULL, (void *) &umax,
                        step3_compute_max,       
                        NULL,    
 #ifdef P4_TO_P8
                        NULL,   
 #endif
                        NULL);
          stop = clock();
          duration = (double)(stop - start) / CLOCKS_PER_SEC; 
          compute_max_running+=duration;
        //stop = clock();
        //duration = (double)(stop - start) / CLOCKS_PER_SEC;
        //cout << "Time taken by p4est_find_max: "
        //<< duration << " seconds" << endl;

        /*
        start = clock();
         cuda_iterate (cuda4est, NULL,
                        &umax, 
                        step3_user_data_api_compute_max,
                        step3_compute_max,
                        step3_cuda_compute_max_api,
                        NULL,
                        NULL,   
 #ifdef P4_TO_P8
                        NULL,    
 #endif
                        NULL);  
        stop = clock();
        duration = (double)(stop - start) / CLOCKS_PER_SEC;
        cout << "Time taken by cuda_find_max: "
        << duration << " seconds" << endl;   
        */
         mpiret =
           sc_MPI_Allreduce (&umax, &global_umax, 1, sc_MPI_DOUBLE, sc_MPI_MAX,
                             p4est->mpicomm);
         SC_CHECK_MPI (mpiret);
         ctx->max_err = orig_max_err * global_umax;
         P4EST_GLOBAL_PRODUCTIONF ("u_max %f\n", global_umax);
 
         /* adapt */
         p4est_refine_ext (p4est, recursive, allowed_level,
                           step3_refine_err_estimate, NULL,
                           step3_replace_quads);
         p4est_coarsen_ext (p4est, recursive, callbackorphans,
                            step3_coarsen_err_estimate, NULL,
                            step3_replace_quads);
         p4est_balance_ext (p4est, P4EST_CONNECT_FACE, NULL,
                            step3_replace_quads);
 
         p4est_ghost_destroy (ghost);
         P4EST_FREE (ghost_data);
         ghost = NULL;
         ghost_data = NULL;
         step3_user_data_api_ghost_data->user_data = ghost_data;
         step3_user_data_api_ghost_data->user_data_elem_count = 0;
        // p4est memory reallocation start
        start = clock();
        //p4est_memory_free(p4est_memory_allocate_info, cuda4est->quad_user_data_api);
        //p4est_memory_allocate_info = p4est_memory_alloc(cuda4est);
        //cuda4est->p4est_memory_allocate_info = p4est_memory_allocate_info;
        stop = clock();
        duration = (double)(stop - start) / CLOCKS_PER_SEC;
        p4est_reallocation+=duration;
        //cout << "Time taken by p4est_reallocation: "
        //<< duration << " seconds" << endl;   
        // p4est memory reallocation end

        /*
        // quadrants memory reallocation start
        start = clock();
        freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
        tree = p4est_tree_array_index (trees, p4est->first_local_tree);
        quadrants = &(tree->quadrants);
        quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
        cuda4est->quads_to_cuda = quads_to_cuda;
        stop = clock();
        duration = (double)(stop - start) / CLOCKS_PER_SEC;
        quadrants_reallocation+=duration;
        //cout << "Time taken by quadrants_reallocation: "
        //<< duration << " seconds" << endl;
        
        start = clock();   
        mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
        stop = clock();
        duration = (double)(stop - start) / CLOCKS_PER_SEC;
        faces_reallocation+=duration;
        //cout << "Time taken by faces_reallocation: "
        //<< duration << " seconds" << endl;
        // quadrants memory reallocation end
        */
       }
       dt = step3_get_timestep (p4est);
     }
     long revision_before_exchange = p4est->revision;
     long revision_after_exchange = p4est->revision;
     /* repartition */
     if (i && !(i % repartition_period)) {
      p4est_partition (p4est, allowcoarsening, NULL);
      revision_after_exchange = p4est->revision;
      /*
      if(revision_after_exchange != revision_before_exchange) {
        start=clock();
        freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
        tree = p4est_tree_array_index (trees, p4est->first_local_tree);
        quadrants = &(tree->quadrants);
        quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
        cuda4est->quads_to_cuda = quads_to_cuda;
        stop = clock();
        duration = (double)(stop - start) / CLOCKS_PER_SEC; 
        quadrants_reallocation+=duration;
      }
      */
 
       if (ghost) {
         p4est_ghost_destroy (ghost);
         P4EST_FREE (ghost_data);
         ghost = NULL;
         ghost_data = NULL;
         step3_user_data_api_ghost_data->user_data = ghost_data;
         step3_user_data_api_ghost_data->user_data_elem_count = 0;
       }
     }

     if((!(i % refine_period) && i) || (i && !(i % repartition_period) && (revision_after_exchange != revision_before_exchange))) {
      start=clock();
      freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
      tree = p4est_tree_array_index (trees, p4est->first_local_tree);
      quadrants = &(tree->quadrants);
      quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
      cuda4est->quads_to_cuda = quads_to_cuda;
      stop = clock();
      duration = (double)(stop - start) / CLOCKS_PER_SEC; 
      quadrants_reallocation+=duration;
     }
     if((!(i % refine_period) && i)) {
      start = clock();   
      mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
      stop = clock();
      duration = (double)(stop - start) / CLOCKS_PER_SEC;
      faces_reallocation+=duration;

      start = clock();
      mallocQuadrantsBlocks(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
      stop = clock();
      duration = (double)(stop - start) / CLOCKS_PER_SEC;
      quadrants_blocks_reallocation+=duration; 
     }

 
     /* write out solution */
     if (!(i % write_period)) {
        start = clock();
        if(!quadrants_is_fresh) {
          //downloadQuadrantsFromCuda(quads_to_cuda, quadrants, cuda4est->quad_user_data_api);
          quadrants_is_fresh = true;
        }
        stop = clock();
        duration = (double)(stop - start) / CLOCKS_PER_SEC;
        downloading_quads+=duration; 
        step3_write_solution (cuda4est, i);
     }
 
     /* synchronize the ghost data */
     if (!ghost) {
       ghost = p4est_ghost_new (p4est, P4EST_CONNECT_FULL);
       ghost_data = P4EST_ALLOC (step3_data_t, ghost->ghosts.elem_count);
       p4est_ghost_exchange_data (p4est, ghost, ghost_data);
       start=clock();
       //freeMemoryForGhost(malloc_ghost);
       //malloc_ghost = mallocForGhost(p4est, ghost);
       //exchangeGhostDataToCuda(malloc_ghost, ghost);
       //cuda4est->ghost_to_cuda = malloc_ghost;
       stop = clock();
       duration = (double)(stop - start) / CLOCKS_PER_SEC; 
       ghost_allocation+=duration;
       
       start = clock();
       if(revision_after_exchange == revision_before_exchange) {
        //freeMemoryForFacesSides(quads_to_cuda);
       }
       //mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
       //step3_user_data_api_ghost_data->user_data = ghost_data;
       //step3_user_data_api_ghost_data->user_data_elem_count = ghost->ghosts.elem_count;
       stop = clock();
       duration = (double)(stop - start) / CLOCKS_PER_SEC; 
       faces_reallocation+=duration;
     }
 
    // quadrants memory reallocation start
    //freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
    //tree = p4est_tree_array_index (trees, p4est->first_local_tree);
    //quadrants = &(tree->quadrants);
    //quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
    //cuda4est->quads_to_cuda = quads_to_cuda;
    //mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
    // quadrants memory reallocation end
     /* compute du/dt */
     /* *INDENT-OFF* */
    downloadQuadrantsFromCuda(quads_to_cuda, quadrants, cuda4est->quad_user_data_api);
    freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
    tree = p4est_tree_array_index (trees, p4est->first_local_tree);
    quadrants = &(tree->quadrants);
    quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
    cuda4est->quads_to_cuda = quads_to_cuda;
    mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
    mallocQuadrantsBlocks(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);

    //printf("udata_before:\n");
    //for(size_t i = 0; i < quadrants->elem_count; i++) {
    //  p4est_quadrant_t *quad = p4est_quadrant_array_index(quadrants, i);
    //  printf("%d: u: %f\n",i, ((step3_data_t*)quad->p.user_data)->u);
    //}
    start = clock();
    cuda_iterate (cuda4est, ghost,
      (void *) ghost_data, 
      step3_user_data_api_ghost_data,
      step3_quad_divergence,
      step3_cuda_quad_divergence_api,
      step3_upwind_flux,
      step3_cuda_upwind_flux_api,   
#ifdef P4_TO_P8
      NULL,    
#endif
      NULL);
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
    flux_compute_running+=duration;
    downloadQuadrantsFromCuda(quads_to_cuda, quadrants, cuda4est->quad_user_data_api);
    double *test_arr_before = new double[quadrants->elem_count];
    for(size_t i = 0; i < quadrants->elem_count; i++) {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(quadrants, i);
      test_arr_before[i] = ((step3_data_t*)quad->p.user_data)->dudt;
    }
    
    //downloadQuadrantsFromCuda(quads_to_cuda, quadrants, cuda4est->quad_user_data_api);
    //freeMemoryForQuadrantsBlocks(quads_to_cuda);
    //mallocQuadrantsBlocks(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
    start = clock();
    cuda_iterate_new (cuda4est, ghost,
      (void *) ghost_data, 
      step3_user_data_api_ghost_data,
      step3_quad_divergence,
      step3_cuda_quad_divergence_api,
      step3_upwind_flux,
      step3_cuda_upwind_flux_api,
      step3_new_cuda_upwind_flux_api,
      step3_new_cuda_quad_divergence_api,   
#ifdef P4_TO_P8
      NULL,    
#endif
      NULL);
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
    new_flux_compute_running+=duration;

    
    
    double *test_arr_after = new double[quadrants->elem_count];
    for(size_t i = 0; i < quadrants->elem_count; i++) {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(quadrants, i);
      test_arr_after[i] = ((step3_data_t*)quad->p.user_data)->dudt;
    }

    for(size_t i = 0; i < quadrants->elem_count; i++) {
      //printf("%d: %f\n", i, test_arr_before[i] - test_arr_after[i]);
    }
    

    freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
    tree = p4est_tree_array_index (trees, p4est->first_local_tree);
    quadrants = &(tree->quadrants);
    quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
    cuda4est->quads_to_cuda = quads_to_cuda;
    mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
    mallocQuadrantsBlocks(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
    
    //cout << "Time taken by cuda_iterate: "
    //     << duration << " seconds" << endl;  
    
    //download cuda quadrants user data start
    //downloadQuadrantsFromCuda(quads_to_cuda, quadrants, cuda4est->quad_user_data_api);
    // download cuda quadrants user data end
    
    /*
    start = clock();
     p4est_iterate (p4est,                 
                    ghost,                 
                    (void *) ghost_data,   
                    step3_quad_divergence, 
                    step3_upwind_flux,     
 #ifdef P4_TO_P8
                    NULL,                  
 #endif
                    NULL);
                    
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
  
    cout << "Time taken by p4est_iterate: "
         << duration << " seconds" << endl;     
    // quadrants memory reallocation start
    freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
    tree = p4est_tree_array_index (trees, p4est->first_local_tree);
    quadrants = &(tree->quadrants);
    quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
    cuda4est->quads_to_cuda = quads_to_cuda;
    mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
    // quadrants memory reallocation end
    */
    
    start = clock();
    cuda_iterate (cuda4est, NULL,
      (void *) &dt, 
      step3_user_data_api_timestep_update,
      step3_timestep_update,
      step3_cuda_timestep_update_api,
      NULL,
      NULL,   
#ifdef P4_TO_P8
      NULL,    
#endif
      NULL
    );
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
    timestep_update_running+=duration;
    //cout << "Time taken by cuda_timestep_update: "
    //     << duration << " seconds" << endl;
    // download cuda quadrants user data start
    //downloadQuadrantsFromCuda(quads_to_cuda, quadrants, cuda4est->quad_user_data_api);
    // download cuda quadrants user data end

 
     /* update u */
     /*
     start = clock();
     p4est_iterate (p4est, NULL, 
                    (void *) &dt,        
                    step3_timestep_update,       
                    NULL,        
 #ifdef P4_TO_P8
                    NULL,        
 #endif
                    NULL);
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
  
    cout << "Time taken by p4est_timestep_update: "
          << duration << " seconds" << endl;
    */
    // quadrants memory reallocation start
    //freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
    //tree = p4est_tree_array_index (trees, p4est->first_local_tree);
    //quadrants = &(tree->quadrants);
    //quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
    //cuda4est->quads_to_cuda = quads_to_cuda;
    //mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
    // quadrants memory reallocation end

     /* synchronize the ghost data */
     //p4est_ghost_exchange_data (p4est, ghost, ghost_data);
     start=clock();
     //freeMemoryForGhost(malloc_ghost);
     //malloc_ghost = mallocForGhost(p4est, ghost);
     //freeGhostDataFromCuda(malloc_ghost);
     //exchangeGhostDataToCuda(malloc_ghost, ghost);
     //cuda4est->ghost_to_cuda = malloc_ghost;
     //freeMemoryForFacesSides(quads_to_cuda);
     //mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
     step3_user_data_api_ghost_data->user_data = ghost_data;
     step3_user_data_api_ghost_data->user_data_elem_count = ghost->ghosts.elem_count;
     stop = clock();
     duration = (double)(stop - start) / CLOCKS_PER_SEC; 
     ghost_allocation+=duration;

     start = clock();
     cuda_iterate (cuda4est, ghost,
      (void *) ghost_data, 
      step3_user_data_api_ghost_data,
      step3_reset_derivatives,
      step3_cuda_reset_derivatives_api,
      step3_minmod_estimate,
      step3_cuda_minmod_estimate_api,   
  #ifdef P4_TO_P8
      NULL,    
  #endif
      NULL);
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
    reset_derivatives_running+=duration;
    //cout << "Time taken by cuda_reset_derivatives: "
    //      << duration << " seconds" << endl;  
    
      start = clock();
      //downloadQuadrantsFromCuda(quads_to_cuda, quadrants, cuda4est->quad_user_data_api);
      stop = clock();
      duration = (double)(stop - start) / CLOCKS_PER_SEC;
      downloading_quads+=duration; 
    
      //cout << "Time taken by cuda download_quads: "
      //     << duration << " seconds" << endl;  
     /* update du/dx estimate */
     /*
     start = clock();
     p4est_iterate (p4est, ghost, (void *) ghost_data,
                    step3_reset_derivatives,
                    step3_minmod_estimate,
 #ifdef P4_TO_P8
                    NULL,
 #endif
                    NULL);
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
  
    cout << "Time taken by p4est_reset_derivatives: "
          << duration << " seconds" << endl;  
    // quadrants memory reallocation start
    start = clock();
    freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
    tree = p4est_tree_array_index (trees, p4est->first_local_tree);
    quadrants = &(tree->quadrants);
    quads_to_cuda = mallocForQuadrants(cuda4est, quadrants, cuda4est->quad_user_data_api);
    cuda4est->quads_to_cuda = quads_to_cuda;
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
    quadrants_reallocation+=duration;
    
    start = clock();
    mallocFacesSides(cuda4est, quadrants, quads_to_cuda, ghost, malloc_ghost);
    stop = clock();
    duration = (double)(stop - start) / CLOCKS_PER_SEC; 
    faces_reallocation+=duration;
    // quadrants memory reallocation end
    */
    quadrants_is_fresh = false;
   }
 
   P4EST_FREE (ghost_data);
   p4est_ghost_destroy (ghost);
   free(step3_cuda_compute_max_api);
   free(step3_cuda_timestep_update_api);
   start=clock();
   p4est_memory_free(p4est_memory_allocate_info, cuda4est->quad_user_data_api);
   stop = clock();
   duration = (double)(stop - start) / CLOCKS_PER_SEC; 
   p4est_reallocation+=duration;

   start = clock();
   freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
   stop = clock();
   duration = (double)(stop - start) / CLOCKS_PER_SEC; 
   quadrants_reallocation+=duration;

   double summary_time = 
   ghost_allocation + p4est_reallocation + quadrants_reallocation + faces_reallocation + quadrants_blocks_reallocation +
   reset_derivatives_running + compute_max_running + flux_compute_running + new_flux_compute_running + timestep_update_running + 
   downloading_quads;
   printf("summary_time: %f\n", summary_time);
   printf("ghost_allocation: %f, in procent: %f\n", ghost_allocation, ghost_allocation/summary_time);
   printf("p4est_reallocation: %f, in procent: %f\n", p4est_reallocation, p4est_reallocation/summary_time);
   printf("quadrants_reallocation: %f, in procent: %f\n", quadrants_reallocation, quadrants_reallocation/summary_time);
   printf("faces_reallocation: %f, in procent: %f\n", faces_reallocation, faces_reallocation/summary_time);
   printf("quadrants_blocks_reallocation %f in procent: %f\n", quadrants_blocks_reallocation, quadrants_blocks_reallocation/summary_time);
   printf("reset_derivatives_running: %f, in procent: %f\n", reset_derivatives_running, reset_derivatives_running/summary_time);
   printf("compute_max_running: %f, in procent: %f\n", compute_max_running, compute_max_running/summary_time);
   printf("flux_compute_running: %f, in procent: %f\n", flux_compute_running, flux_compute_running/summary_time);
   printf("new_flux_compute_running: %f, in procent: %f\n", new_flux_compute_running, new_flux_compute_running/summary_time);
   printf("timestep_update_running: %f, in procent: %f\n", timestep_update_running, timestep_update_running/summary_time);
   printf("downloading_quads: %f, in procent: %f\n", downloading_quads, downloading_quads/summary_time);
 }
 
 /** The main step 3 program.
  *
  * Setup of the example parameters; create the forest, with the state variable
  * stored in the quadrant data; refine, balance, and partition the forest;
  * timestep; clean up, and exit.
  */
 int
 main (int argc, char **argv)
 {
  auto start = std::chrono::high_resolution_clock::now();
   int                 mpiret;
   int                 recursive, partforcoarsen;
   sc_MPI_Comm         mpicomm;
   p4est_t            *p4est;
   p4est_connectivity_t *conn;
   step3_ctx_t         ctx;
 
   /* Initialize MPI; see sc_mpi.h.
    * If configure --enable-mpi is given these are true MPI calls.
    * Else these are dummy functions that simulate a single-processor run. */

   mpiret = sc_MPI_Init (&argc, &argv);
   SC_CHECK_MPI (mpiret);
   mpicomm = sc_MPI_COMM_WORLD;
 
   /* These functions are optional.  If called they store the MPI rank as a
    * static variable so subsequent global p4est log messages are only issued
    * from processor zero.  Here we turn off most of the logging; see sc.h. */
   sc_init (mpicomm, 1, 1, NULL, SC_LP_ESSENTIAL);
   p4est_init (NULL, SC_LP_PRODUCTION);
   P4EST_GLOBAL_PRODUCTIONF
     ("This is the p4est %dD demo example/steps/%s_step3\n",
      P4EST_DIM, P4EST_STRING);
 
   ctx.bump_width = 0.1;
   ctx.max_err = 2.e-2;
   ctx.center[0] = 0.5;
   ctx.center[1] = 0.5;
 #ifdef P4_TO_P8
   ctx.center[2] = 0.5;
 #endif
 #ifndef P4_TO_P8
   /* randomly chosen advection direction */
   ctx.v[0] = -0.445868402501118;
   ctx.v[1] = -0.895098523991131;
 #else
   ctx.v[0] = 0.485191768970225;
   ctx.v[1] = -0.427996381877778;
   ctx.v[2] = 0.762501176669961;
 #endif
   ctx.refine_period = 2;
   ctx.repartition_period = 4;
   ctx.write_period = 8;


 
   /* Create a forest that consists of just one periodic quadtree/octree. */
 #ifndef P4_TO_P8
   conn = p4est_connectivity_new_periodic ();
 #else
   conn = p8est_connectivity_new_periodic ();
 #endif
 
   /* *INDENT-OFF* */
   p4est = p4est_new_ext (mpicomm, /* communicator */
                          conn,    /* connectivity */
                          0,       /* minimum quadrants per MPI process */
                          6,       /* minimum level of refinement */
                          1,       /* fill uniform */
                          sizeof (step3_data_t),         /* data size */
                          step3_init_initial_condition,  /* initializes data */
                          (void *) (&ctx));              /* context */
  cuda4est_t *cuda4est = (cuda4est_t*) malloc(sizeof(cuda4est_t));
  cuda4est->p4est = p4est;
  
  user_data_for_cuda_t *user_data_api = (user_data_for_cuda_t*) malloc(sizeof(user_data_for_cuda_t));
  user_data_api->user_data = &ctx;
  user_data_api->alloc_cuda_memory = alloc_cuda_memory_step3_ctx;
  user_data_api->free_cuda_memory = free_cuda_memory_step3_ctx;
  user_data_api->get_cuda_allocated_user_data = get_cuda_allocated_user_data_step3_ctx;
  
  cuda4est->user_data_api = user_data_api;

  quad_user_data_api_t *quad_user_data_api = (quad_user_data_api_t*) malloc(sizeof(quad_user_data_api_t));
  quad_user_data_api->alloc_cuda_memory = alloc_cuda_memory_step3_quad_user_data;
  quad_user_data_api->alloc_cuda_memory_for_all_quads = alloc_all_quads_cuda_memory_step3;
  quad_user_data_api->free_cuda_memory = free_cuda_memory_step3_quad_user_data;
  quad_user_data_api->free_cuda_memory_for_all_quads = free_all_quads_cuda_memory_step3;
  quad_user_data_api->get_cuda_allocated_user_data = get_cuda_allocated_user_data_step3_quad_user_data;
  quad_user_data_api->update_quad_cuda_user_data = update_quad_cuda_step3_user_data;
  quad_user_data_api->update_all_quads_cuda_user_data = update_all_quads_cuda_user_data_step3;
  quad_user_data_api->download_quad_cuda_user_data_to_host = download_quad_cuda_user_data_step3_to_host;
  quad_user_data_api->download_all_quads_cuda_user_data_to_host = download_all_quads_cuda_user_data_to_host_t_step3;

  cuda4est->quad_user_data_api = quad_user_data_api;

  gpuErrchk(cudaMemcpyToSymbol(cuda_ctx_ptr, &ctx, sizeof(step3_ctx_t)));
  cuda4est->ctx_size = sizeof(step3_ctx_t);
  gpuErrchk(cudaGetSymbolAddress((void**)&(cuda4est->d_ctx), cuda_ctx_ptr));
  cuda4est->block_quadrants_max_size = 256;
  
   /* *INDENT-ON* */
 
   /* refine and coarsen based on an interpolation error estimate */
   recursive = 1;
   p4est_refine (p4est, recursive, step3_refine_err_estimate,
                 step3_init_initial_condition);
   p4est_coarsen (p4est, recursive, step3_coarsen_initial_condition,
                  step3_init_initial_condition);
 
   /* Partition: The quadrants are redistributed for equal element count.  The
    * partition can optionally be modified such that a family of octants, which
    * are possibly ready for coarsening, are never split between processors. */
   partforcoarsen = 1;
 
   /* If we call the 2:1 balance we ensure that neighbors do not differ in size
    * by more than a factor of 2.  This can optionally include diagonal
    * neighbors across edges or corners as well; see p4est.h. */
   p4est_balance (p4est, P4EST_CONNECT_FACE, step3_init_initial_condition);
   p4est_partition (p4est, partforcoarsen, NULL);
 
   /* time step */
   //step3_timestep (cuda4est, 0.8);
   step3_timestep (cuda4est, 0.0003);
   //step3_timestep(cuda4est, 0.003);
 
   /* Destroy the p4est and the connectivity structure. */
   p4est_destroy (p4est);
   p4est_connectivity_destroy (conn);
 
   /* Verify that allocations internal to p4est and sc do not leak memory.
    * This should be called if sc_init () has been called earlier. */
   sc_finalize ();
 
   /* This is standard MPI programs.  Without --enable-mpi, this is a dummy. */
   mpiret = sc_MPI_Finalize ();
   SC_CHECK_MPI (mpiret);
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
  
  // To get the value of duration use the count() 
  // member function on the duration object 
  std::cout << "time duration: "<< duration.count() << std::endl; 
   return 0;
 }