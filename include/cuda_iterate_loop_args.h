#ifndef CUDA_ITERATE_LOOP_ARGS_H
#define CUDA_ITERATE_LOOP_ARGS_H

#include "cuda_iterate.h"

#define CUDA_ITER_STRIDE (P4EST_CHILDREN + 1)
/* loop arg functions */
typedef struct p4est_iter_loop_args
{
  int                 alloc_size;       /* large enough to accomodate strange
                                           corners/edges between trees */
#ifdef P4_TO_P8
  int8_t              loop_edge;        /* should edge_iterate be run */
#endif
  int8_t              loop_corner;      /* should corner_iterate by run */

  int                 level;
  int                *level_num;        /* an array that keeps track of which
                                           branch we take at each step in the
                                           heirarchical search areas */
  int                *quad_idx2;        /* an indexing variable used in
                                           the iterate functions: passed as an
                                           argument to avoid using alloc/free
                                           on each call */
  sc_array_t        **quadrants;        /* the arrays, two for each side (one
                                           local, one ghost), that contain the
                                           quadrants in each search area */
  size_t            **index;    /* for each sidetype, the indices in quadrants
                                   that form the bounds of the heirarchical
                                   search areas */
  size_t             *first_index;      /* an indexing variable used in the
                                           iterate functions: passed as an
                                           argument to avoid using alloc/free
                                           on each call */
  size_t             *count;    /* a counting variable used in the iterate
                                   functions: passed as an argument to
                                   avoid using alloc/free on each call */
  p4est_quadrant_t  **test;     /* a testing variable used in the iterate
                                   functions: passed as an argument to
                                   avoid using alloc/free on each call */
  int                *test_level;       /* a testing variable used in
                                           the iterate functions:: passed as an
                                           argument to avoid using alloc/free
                                           on each call */
  int8_t             *refine;   /* a testing variable used in the iterate
                                   functions: passed as an argument to avoid
                                   using alloc/free on each call */
  sc_array_t         *tier_rings;
}
p4est_iter_loop_args_t;

/* corner iterate function */
typedef struct p4est_iter_corner_args
{
  int                 num_sides;
  int                *start_idx2;
  int                 remote;
  p4est_iter_loop_args_t *loop_args;
  p4est_iter_corner_info_t info;
}
p4est_iter_corner_args_t;

/* face iterate functions */
typedef struct p4est_iter_face_args
{
  p4est_iter_loop_args_t *loop_args;
  int                 start_idx2[2];
  /* when a search branch is refined,
     num_to_child says which child id
     corresponds to the branch number for
     each side of the face. e.g. Suppose
     face[left] = 1, face[right] = 0, and
     orientation = 0 in 3D. The child ids
     of the descendants of the current
     search area that touch face[left]
     are 1, 3, 5, 7, and given
     face[right] and the orientation, the
     descendants that are opposite them
     are 0, 2, 4, 6, respectively:
     therefore num_to_child =
     { 1, 3, 5, 7, 0, 2, 4, 6} */
  int                 num_to_child[P4EST_CHILDREN];
  int8_t              outside_face;     /* indicates if we are at a tree
                                           boundary without a neighbor across
                                           the face */
#ifdef P4_TO_P8
  p8est_iter_edge_args_t edge_args[2][2];
#endif
  p4est_iter_corner_args_t corner_args;
  p4est_iter_face_info_t info;
  int                 remote;
}
p4est_iter_face_args_t;
/* loop arg functions */
typedef struct cuda_iter_loop_args
{
  int                 alloc_size;       /* large enough to accomodate strange
                                           corners/edges between trees */
#ifdef P4_TO_P8
  int8_t              loop_edge;        /* should edge_iterate be run */
#endif
  int8_t              loop_corner;      /* should corner_iterate by run */

  int                 level;
  int                *level_num;        /* an array that keeps track of which
                                           branch we take at each step in the
                                           heirarchical search areas */
  int                *quad_idx2;        /* an indexing variable used in
                                           the iterate functions: passed as an
                                           argument to avoid using alloc/free
                                           on each call */
  sc_array_t        **quadrants;        /* the arrays, two for each side (one
                                           local, one ghost), that contain the
                                           quadrants in each search area */
  size_t            **index;    /* for each sidetype, the indices in quadrants
                                   that form the bounds of the heirarchical
                                   search areas */
  size_t             *first_index;      /* an indexing variable used in the
                                           iterate functions: passed as an
                                           argument to avoid using alloc/free
                                           on each call */
  size_t             *count;    /* a counting variable used in the iterate
                                   functions: passed as an argument to
                                   avoid using alloc/free on each call */
  p4est_quadrant_t  **test;     /* a testing variable used in the iterate
                                   functions: passed as an argument to
                                   avoid using alloc/free on each call */
  int                *test_level;       /* a testing variable used in
                                           the iterate functions:: passed as an
                                           argument to avoid using alloc/free
                                           on each call */
  int8_t             *refine;   /* a testing variable used in the iterate
                                   functions: passed as an argument to avoid
                                   using alloc/free on each call */
  sc_array_t         *tier_rings;
}
cuda_iter_loop_args_t;

/* corner iterate function */
typedef struct cuda_iter_corner_args
{
  int                 num_sides;
  int                *start_idx2;
  int                 remote;
  cuda_iter_loop_args_t *loop_args;
  cuda_iter_corner_info_t info;
}
cuda_iter_corner_args_t;

/* face iterate functions */
typedef struct cuda_iter_face_args
{
  cuda_iter_loop_args_t *loop_args;
  int                 start_idx2[2];
  /* when a search branch is refined,
     num_to_child says which child id
     corresponds to the branch number for
     each side of the face. e.g. Suppose
     face[left] = 1, face[right] = 0, and
     orientation = 0 in 3D. The child ids
     of the descendants of the current
     search area that touch face[left]
     are 1, 3, 5, 7, and given
     face[right] and the orientation, the
     descendants that are opposite them
     are 0, 2, 4, 6, respectively:
     therefore num_to_child =
     { 1, 3, 5, 7, 0, 2, 4, 6} */
  int                 num_to_child[P4EST_CHILDREN];
  int8_t              outside_face;     /* indicates if we are at a tree
                                           boundary without a neighbor across
                                           the face */
#ifdef P4_TO_P8
  p8est_iter_edge_args_t edge_args[2][2];
#endif
  cuda_iter_corner_args_t corner_args;
  cuda_iter_face_info_t info;
  int                 remote;
}
cuda_iter_face_args_t;

/* volume iterate functions */
typedef struct cuda_iter_volume_args
{
  cuda_iter_loop_args_t *loop_args;
  int                 start_idx2;
  cuda_iter_face_args_t face_args[P4EST_DIM][P4EST_CHILDREN / 2];
#ifdef P4_TO_P8
  p8est_iter_edge_args_t edge_args[P4EST_DIM][2];
#endif
  cuda_iter_corner_args_t corner_args;
  cuda_iter_volume_info_t info;
  int                 remote;
}
cuda_iter_volume_args_t;

/* volume iterate functions */
typedef struct p4est_iter_volume_args
{
  p4est_iter_loop_args_t *loop_args;
  int                 start_idx2;
  p4est_iter_face_args_t face_args[P4EST_DIM][P4EST_CHILDREN / 2];
#ifdef P4_TO_P8
  p8est_iter_edge_args_t edge_args[P4EST_DIM][2];
#endif
  p4est_iter_corner_args_t corner_args;
  p4est_iter_volume_info_t info;
  int                 remote;
}
p4est_iter_volume_args_t;

typedef struct cuda_iter_tier
{
  p4est_quadrant_t   *key;
  size_t              array[CUDA_ITER_STRIDE];
}
cuda_iter_tier_t;

typedef struct cuda_iter_tier_ring
{
  int                 next;
  sc_array_t          tiers;
}
cuda_iter_tier_ring_t;

p4est_iter_loop_args_t *
cuda_iter_loop_args_new (p4est_connectivity_t * conn,
#ifdef P4_TO_P8
                          p8est_iter_edge_t iter_edge,
#endif
                          p4est_iter_corner_t iter_corner,
                          p4est_ghost_t * ghost_layer, int num_procs);
/* initialize volume arguments for a search in a tree */
void
cuda_iter_init_volume (p4est_iter_volume_args_t * args, p4est_t * p4est,
                        p4est_ghost_t * ghost_layer,
                        p4est_iter_loop_args_t * loop_args, p4est_topidx_t t);

sc_array_t  *
cuda_iter_tier_rings_new (int num_procs);
void
cuda_iter_tier_rings_destroy (sc_array_t * tier_rings);

void
cuda_iter_tier_update (sc_array_t * view, int level, size_t * next_tier,
                        size_t shift);

void
cuda_iter_tier_insert (sc_array_t * view, int level, size_t * next_tier,
                        size_t shift, sc_array_t * tier_rings,
                        p4est_quadrant_t * q);

void
cuda_iter_copy_indices (p4est_iter_loop_args_t * loop_args,
                         const int *start_idx2, int old_num, int factor);

int32_t     *
cuda_iter_get_boundaries (p4est_t * p4est, p4est_topidx_t * last_run_tree,
                           int remote);
void
cuda_iter_reset_volume (p4est_iter_volume_args_t * args);

void
cuda_iter_loop_args_destroy (p4est_iter_loop_args_t * loop_args);

#endif