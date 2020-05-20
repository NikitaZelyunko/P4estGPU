#include "p4est_to_cuda.h"
#include "cuda_iterate_loop_args.h"

typedef struct step3_data
{
  double              u;             /**< the state variable */
  double              du[P4EST_DIM]; /**< the spatial derivatives */
  double              dudt;          /**< the time derivative */
}
step3_data_t;

p4est_quadrants_to_cuda_t* mallocForQuadrants(cuda4est_t* cuda4est, sc_array_t *quadrants, quad_user_data_api_t *user_data_api) {
    p4est_quadrants_to_cuda_t *quads_to_cuda = (p4est_quadrants_to_cuda_t*) malloc(sizeof(p4est_quadrants_to_cuda_t));
    sc_array_t *d_quadrants;
    size_t d_quadrants_array_size = quadrants->elem_size * quadrants->elem_count;
    size_t quadrants_length = quadrants->elem_count;
    gpuErrchk(cudaMalloc((void**)&d_quadrants, sizeof(sc_array_t)));
    gpuErrchk(cudaMemcpy(d_quadrants, quadrants, sizeof(sc_array_t), cudaMemcpyHostToDevice));
    quads_to_cuda->d_quadrants = d_quadrants;
    quads_to_cuda->quadrants_length = quadrants_length;
    quads_to_cuda->quadrants_allocated_size = d_quadrants_array_size;

    all_quads_user_data_allocate_info_t * all_quads_user_data_allocate_info = (all_quads_user_data_allocate_info_t*) malloc(sizeof(all_quads_user_data_allocate_info_t));
    user_data_api->alloc_cuda_memory_for_all_quads(all_quads_user_data_allocate_info, quadrants);
    quads_to_cuda->all_quads_user_data_allocate_info = all_quads_user_data_allocate_info;
    char *d_all_quads_user_data = (char*)all_quads_user_data_allocate_info->d_all_quads_user_data;
    
    size_t quad_size = sizeof(p4est_quadrant_t);
    size_t user_data_size = cuda4est->p4est->data_size;
    p4est_quadrant_t *h_quadrants_array_temp = (p4est_quadrant_t*) malloc(quadrants_length * quad_size);
    p4est_quadrant_t *quad_cursor = h_quadrants_array_temp;
    char *user_data_cursor = d_all_quads_user_data;
    for(size_t i = 0; i < quadrants_length; i++, quad_cursor++, user_data_cursor+=user_data_size) {
        p4est_quadrant_t *temp_quad = (p4est_quadrant_t*) malloc(quad_size);
        memcpy(temp_quad, p4est_quadrant_array_index (quadrants, i), quad_size);
        temp_quad->p.user_data = user_data_cursor;
        memcpy(quad_cursor, temp_quad, quad_size);
        free(temp_quad);
    }

    p4est_quadrant_t *d_quadrants_array_temp;
    gpuErrchk(cudaMalloc((void**)&d_quadrants_array_temp, d_quadrants_array_size));
    gpuErrchk(cudaMemcpy(d_quadrants_array_temp, h_quadrants_array_temp, d_quadrants_array_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_quadrants->array), &d_quadrants_array_temp, sizeof(p4est_quadrant_t*), cudaMemcpyHostToDevice));
    free(h_quadrants_array_temp);
    
    quads_to_cuda->d_quadrants_array_temp = d_quadrants_array_temp;
    return quads_to_cuda;
}

void updateQuadrants(cuda4est_t* cuda4est, p4est_quadrants_to_cuda* quads_to_cuda, sc_array_t* quadrants, quad_user_data_api_t *user_data_api) {
    sc_array_t *d_quadrants = quads_to_cuda->d_quadrants;
    size_t d_quadrants_array_size = quads_to_cuda->quadrants_allocated_size;
    size_t quadrants_length = quads_to_cuda->quadrants_length;
    gpuErrchk(cudaMemcpy(d_quadrants, quadrants, sizeof(sc_array_t), cudaMemcpyHostToDevice));

    all_quads_user_data_allocate_info_t * new_quads_user_data_allocate_info = (all_quads_user_data_allocate_info_t*) malloc(sizeof(all_quads_user_data_allocate_info_t));
    void** all_host_quads_user_data = (void**)malloc(quadrants_length * sizeof(void*));
    
    for(size_t i = 0; i < quadrants_length; i++) {
        all_host_quads_user_data[i] = p4est_quadrant_array_index(quadrants, i)->p.user_data;
    }
    
    new_quads_user_data_allocate_info->all_quads_user_data = all_host_quads_user_data;
    user_data_api->update_all_quads_cuda_user_data(quads_to_cuda->all_quads_user_data_allocate_info, new_quads_user_data_allocate_info);
    size_t quad_size = sizeof(p4est_quadrant_t);
    size_t user_data_size = cuda4est->p4est->data_size;

    p4est_quadrant_t *h_quadrants_array_temp = (p4est_quadrant_t*) malloc(quadrants_length * quad_size);

    p4est_quadrant_t *quad_cursor = h_quadrants_array_temp;
    char *user_data_cursor = (char*) new_quads_user_data_allocate_info->d_all_quads_user_data;
    for(size_t i = 0; i < quadrants->elem_count; i++, quad_cursor++, user_data_cursor+=user_data_size) {
        p4est_quadrant_t *temp_quad = (p4est_quadrant_t*) malloc(sizeof(p4est_quadrant_t));
        memcpy(temp_quad, p4est_quadrant_array_index (quadrants, i), sizeof(p4est_quadrant_t));
        temp_quad->p.user_data = user_data_cursor;
        memcpy(quad_cursor, temp_quad, quad_size);
        free(temp_quad);
    }

    free(quads_to_cuda->all_quads_user_data_allocate_info);
    quads_to_cuda->all_quads_user_data_allocate_info = new_quads_user_data_allocate_info;
    p4est_quadrant_t *d_quadrants_array_temp = quads_to_cuda->d_quadrants_array_temp;
    gpuErrchk(cudaMemcpy(d_quadrants_array_temp, h_quadrants_array_temp, d_quadrants_array_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_quadrants->array), &d_quadrants_array_temp, sizeof(p4est_quadrant_t*), cudaMemcpyHostToDevice));
    free(h_quadrants_array_temp);
}

void freeMemoryForQuadrants(p4est_quadrants_to_cuda_t* quads_to_cuda, quad_user_data_api_t *user_data_api) {
  freeMemoryForFacesSides(quads_to_cuda);
  user_data_api->free_cuda_memory_for_all_quads(quads_to_cuda->all_quads_user_data_allocate_info);
  free(quads_to_cuda->all_quads_user_data_allocate_info);
  gpuErrchk(cudaFree(quads_to_cuda->d_quadrants_array_temp));
  gpuErrchk(cudaFree(quads_to_cuda->d_quadrants));
}

void downloadQuadrantsFromCuda(p4est_quadrants_to_cuda* quads_to_cuda, sc_array_t* quadrants, quad_user_data_api_t *user_data_api) {
  user_data_api->download_all_quads_cuda_user_data_to_host(quads_to_cuda->all_quads_user_data_allocate_info, quadrants);
}

p4est_iter_face_side_t* copy_iter_face_side(p4est_iter_face_side_t* source_face_side) {
  size_t face_bytes_size = sizeof(p4est_iter_face_side_t);
  p4est_iter_face_side_t* face_side = (p4est_iter_face_side_t*)malloc(face_bytes_size);
  memcpy(face_side, source_face_side, face_bytes_size);
  return face_side;
}

static void
p4est_face_iterate_without_callbacks (p4est_iter_face_args_t * args, std::vector<p4est_iter_face_side_t*>* all_faces)
{

  const int           left = 0;
  const int           right = 1;
  const int           local = 0;
  const int           ghost = 1;
  const int           ntc_str = P4EST_CHILDREN / 2;

  p4est_iter_loop_args_t *loop_args = args->loop_args;
  int                 start_level = loop_args->level;
  int                *start_idx2 = args->start_idx2;
  int                *level_num = loop_args->level_num;
  sc_array_t        **quadrants = loop_args->quadrants;
  size_t            **zindex = loop_args->index;
  size_t             *first_index = loop_args->first_index;
  int                *num_to_child = args->num_to_child;
  p4est_quadrant_t  **test = loop_args->test;
  size_t             *count = loop_args->count;
  int                *test_level = loop_args->test_level;
  int                *quad_idx2 = loop_args->quad_idx2;
  int8_t             *refine = loop_args->refine;
  int                 limit;

  int                 i;
  int                *Level = &(loop_args->level);
  int                 side;
  int                 type;
  int                 st;
  int                 level_idx2;
  p4est_iter_face_info_t *info = &(args->info);
  p4est_iter_face_side_t *fside;
  p4est_quadrant_t  **quads;
  p4est_locidx_t     *quadids;
  int8_t             *is_ghost;
  int                 child_corner;
  int8_t              has_local;
  sc_array_t          test_view;
  p4est_iter_corner_args_t *corner_args = &(args->corner_args);
  sc_array_t         *tier_rings = loop_args->tier_rings;
#ifdef P4_TO_P8
  int                 dir;
#endif
  int                 stop_refine;
  int                 all_empty;

  /* if we are at an outside face, then there is no right half to our search
   * that needs to be coordinated with the left half */
  limit = args->outside_face ? left : right;

  /* level_idx2 moves us to the correct set of bounds within the index arrays
   * for the level: it is a set of bounds because it includes all children at
   * this level */
  level_idx2 = start_level * CUDA_ITER_STRIDE;

  for (side = left; side <= limit; side++) {

    /* start_idx2 gives the ancestor id at level for the search area on this
     * side, so quad_idx2[side] now gives the correct location in
     * zindex[sidetype] of the bounds of the search area */
    quad_idx2[side] = level_idx2 + start_idx2[side];

    /* get the location in quadrants[sidetype] of the first quadrant in the
     * search area, and the count of quadrants in the search area */
    for (type = local; type <= ghost; type++) {
      st = side * 2 + type;
      first_index[st] = zindex[st][quad_idx2[side]];
      count[st] = (zindex[st][quad_idx2[side] + 1] - first_index[st]);
    }
  }

  /* face_iterate only runs if there is a chance of a local quadrant touching
   * the desired face */
  if (!args->outside_face) {
    if (!count[left * 2 + local] && !count[right * 2 + local]) {
      return;
    }
  }
  else {
    if (!count[left * 2 + local]) {
      return;
    }
  }

  /* we think of the search tree as being rooted at start_level, so we can
   * think the branch number at start_level as 0, even if it actually is not */
  level_num[start_level] = 0;
  for (;;) {
    /* for each sidetype, get the first quadrant in that sidetype search area
     */
    for (side = left; side <= limit; side++) {
      for (type = local; type <= ghost; type++) {
        st = side * 2 + type;
        if (count[st]) {
          test[st] = p4est_quadrant_array_index (quadrants[st],
                                                 first_index[st]);
          test_level[st] = (int) test[st]->level;
        }
        else {
          test[st] = NULL;
          test_level[st] = -1;
        }
      }
    }
    /* initially assume that each side needs to be refined */
    refine[left] = refine[right] = 1;
    stop_refine = 0;
    has_local = 0;

    /* get a candidate from each sidetype */
    for (side = left; side <= limit; side++) {
      for (type = local; type <= ghost; type++) {
        st = side * 2 + type;
        /* if the candidate from sidetype is the same size as the search area,
         * then we do not refine this side */
        if (test_level[st] == *Level) {
          if (!stop_refine) {
            stop_refine = 1;
            /* we are not going to recur on the next level, instead moving to
             * the next branch on this level */
            level_num[*Level]++;
            /* if there is no face callback (i.e., we are just running
             * face_iterate to find edges and corners), then we're done with
             * this branch */
            //if (iter_face == NULL) {
            //  goto change_search_area;
            //}
          }
          P4EST_ASSERT (count[st] == 1);
          P4EST_ASSERT (count[side * 2 + (type ^ 1)] == 0);
          refine[side] = 0;
          fside = cuda_iter_fside_array_index_int (&(info->sides), side);
          fside->is_hanging = 0;
          fside->is.full.quad = test[st];
          fside->is.full.quadid = (p4est_locidx_t) first_index[st];
          has_local = (has_local || (type == local));
          fside->is.full.is_ghost = (type == ghost);
        }
      }
    }
    for (side = left; side <= limit; side++) {
      if (refine[side]) {
        if (stop_refine && count[side * 2 + local] == 0 &&
            count[side * 2 + ghost] == 0) {
          fside = cuda_iter_fside_array_index_int (&info->sides, side);
          fside->is_hanging = 0;
          fside->is.full.quad = NULL;
          fside->is.full.is_ghost = 1;
          fside->is.full.quadid = -1;
          refine[side] = 0;
        }
      }
      if (refine[side]) {
        quad_idx2[side] = level_idx2 + CUDA_ITER_STRIDE;
        for (type = local; type <= ghost; type++) {
          st = side * 2 + type;
          sc_array_init_view (&test_view, quadrants[st],
                              first_index[st], count[st]);
          cuda_iter_tier_insert (&test_view, *Level, zindex[st] +
                                  quad_idx2[side], first_index[st],
                                  tier_rings, test[st]);
        }
        if (stop_refine) {
          fside = cuda_iter_fside_array_index_int (&(info->sides), side);
          fside->is_hanging = 1;
          quads = fside->is.hanging.quad;
          quadids = fside->is.hanging.quadid;
          is_ghost = fside->is.hanging.is_ghost;
          for (i = 0; i < P4EST_CHILDREN / 2; i++) {
            /* fside expects the hanging quadrants listed in z order, not search
             * order */
            child_corner = num_to_child[side * ntc_str + i];
            child_corner =
              p4est_corner_face_corners[child_corner][fside->face];
            quads[child_corner] = NULL;
            quadids[child_corner] = -1;
            is_ghost[child_corner] = 1;
            quad_idx2[side] = level_idx2 + CUDA_ITER_STRIDE +
              num_to_child[side * ntc_str + i];
            for (type = local; type <= ghost; type++) {
              st = side * 2 + type;
              first_index[st] = zindex[st][quad_idx2[side]];
              count[st] = zindex[st][quad_idx2[side] + 1] - first_index[st];
              /* if the search area is non-empty, by the two to one condition
               * it must contain exactly one quadrant: if one of the two types
               * is local, we run iter_face */
              if (count[st]) {
                quads[child_corner] = p4est_quadrant_array_index
                  (quadrants[st], first_index[st]);
                P4EST_ASSERT ((int) quads[child_corner]->level == *Level + 1);
                quadids[child_corner] = (p4est_locidx_t) first_index[st];
                is_ghost[child_corner] = (type == ghost);
                has_local = (has_local || (type == local));
              }
            }
          }
        }
      }
    }
    if (stop_refine) {
      if (has_local) {
        for (side = left; side <= limit; side++) {
          all_faces->push_back(copy_iter_face_side(cuda_iter_fside_array_index_int (&info->sides, side)));
        }
        //iter_face (info, user_data);
      }
    }
    else {
      /* if we refined both sides, we descend to the next level from this branch
       * and continue searching there */
      level_num[++(*Level)] = 0;
      level_idx2 += CUDA_ITER_STRIDE;
    }

  change_search_area:

    for (;;) {
      /* if we tried to advance the search area on start_level, we've completed
       * the search */
      if (level_num[start_level] > 0) {
        P4EST_ASSERT (*Level == start_level);
        return;
      }

      /* if we have tried to advance the search area past the number of
       * descendants, that means that we have completed all of the branches on
       * this level */
      if (level_num[*Level] == P4EST_CHILDREN / 2) {
#ifdef P4_TO_P8
        /* if we have an edge callback, we need to run it on all of the edges
         * between the face branches on this level */
        if (loop_args->loop_edge) {
          for (dir = 0; dir < 2; dir++) {
            for (side = 0; side < 2; side++) {
              P4EST_ASSERT (args->edge_args[dir][side].num_sides ==
                            2 * (limit + 1));
              cuda_iter_copy_indices (loop_args,
                                       args->edge_args[dir][side].start_idx2,
                                       limit + 1, 2);
              p8est_edge_iterate (&(args->edge_args[dir][side]), user_data,
                                  iter_edge, iter_corner);
            }
          }
        }
#endif
        /* if we have a corner callback, we need to run it on the corner between
         * the face branches on this level */
        //if (iter_corner != NULL) {
        //  P4EST_ASSERT (corner_args->num_sides ==
        //                (P4EST_CHILDREN / 2) * (limit + 1));
        //  cuda_iter_copy_indices (loop_args, corner_args->start_idx2,
        //                           limit + 1, P4EST_HALF);
        //  p4est_corner_iterate (corner_args, user_data, iter_corner);
        //}

        /* now that we're done on this level, go up a level and over a branch */
        level_num[--(*Level)]++;
        level_idx2 -= CUDA_ITER_STRIDE;
      }
      else {
        /* at this point, we need to initialize the bounds of the search areas
         * for this new branch */
        all_empty = 1;
        for (side = left; side <= limit; side++) {
          quad_idx2[side] =
            level_idx2 + num_to_child[side * ntc_str + level_num[*Level]];
        }
        for (side = left; side <= limit; side++) {
          for (type = local; type <= ghost; type++) {
            st = side * 2 + type;
            first_index[st] = zindex[st][quad_idx2[side]];
            count[st] = (zindex[st][quad_idx2[side] + 1] - first_index[st]);
            if (type == local && count[st]) {
              all_empty = 0;
            }
          }
        }
        if (all_empty) {
          /* if there are no local quadrants in either of the search areas, we're
           * done with this search area and proceed to the next branch on this
           * level */
          level_num[*Level]++;
        }
        else {
          /* otherwise we are done changing the search area */
          break;
        }
      }
    }
  }
}

static void
cuda_volume_iterate_without_callbacks (p4est_iter_volume_args_t * args, std::vector<p4est_iter_face_side_t*>* all_faces)
{
  const int           local = 0;
  const int           ghost = 1;

  int                 dir, side, type;

  p4est_iter_loop_args_t *loop_args = args->loop_args;
  int                 start_level = loop_args->level;
  int                *Level = &(loop_args->level);
  int                 start_idx2 = args->start_idx2;
  int                *level_num = loop_args->level_num;
  sc_array_t        **quadrants = loop_args->quadrants;
  size_t            **zindex = loop_args->index;
  size_t             *first_index = loop_args->first_index;
  p4est_quadrant_t  **test = loop_args->test;
  size_t             *count = loop_args->count;
  int                *test_level = loop_args->test_level;
  sc_array_t         *tier_rings = loop_args->tier_rings;
  int                 quad_idx2;
  sc_array_t          test_view;
  p4est_iter_volume_info_t *info = &(args->info);
  int                 level_idx2;
  int                 refine;

  /* level_idx2 moves us to the correct set of bounds within the index arrays
   * for the level: it is a set of bounds because it includes all children at
   * this level */
  level_idx2 = start_level * CUDA_ITER_STRIDE;

  /* start_idx2 gives the ancestor id at level for the search area,
   * so quad_idx2 now gives the correct location in
   * index[type] of the bounds of the search area */
  quad_idx2 = level_idx2 + start_idx2;
  for (type = local; type <= ghost; type++) {
    first_index[type] = zindex[type][quad_idx2];
    count[type] = zindex[type][quad_idx2 + 1] - first_index[type];
  }

  /* if ther are no local quadrants, nothing to be done */
  if (!count[local]) {
    return;
  }

  /* we think of the search tree as being rooted at start_level, so we can
   * think the branch number at start_level as 0, even if it actually is not */
  level_num[start_level] = 0;



  for (;;) {

    refine = 1;
    /* for each type, get the first quadrant in the search area */
    for (type = local; type <= ghost; type++) {
      if (count[type]) {
        test[type] = p4est_quadrant_array_index (quadrants[type],
                                                 first_index[type]);
        test_level[type] = (int) test[type]->level;
        /* if the quadrant is the same size as the search area, we're done
         * search */
        if (test_level[type] == *Level) {
          refine = 0;
          P4EST_ASSERT (!count[type ^ 1]);
          /* if the quadrant is local, we run the callback */
          if (type == local) {
            info->quad = test[type];
            info->quadid = (p4est_locidx_t) first_index[type];
            //if (iter_volume != NULL) {
            //  iter_volume (info, user_data);
            //}
          }
          /* proceed to the next search area on this level */
          level_num[*Level]++;
        }
      }
      else {
        test[type] = NULL;
        test_level[type] = -1;
      }
    }

    if (refine) {
      /* we need to refine, we take the search area and split it up, taking the
       * indices for the refined search areas and placing them on the next tier in
       * index[type] */
      quad_idx2 = level_idx2 + CUDA_ITER_STRIDE;
      for (type = local; type <= ghost; type++) {
        sc_array_init_view (&test_view, quadrants[type],
                            first_index[type], count[type]);
        cuda_iter_tier_insert (&test_view, *Level, zindex[type] + quad_idx2,
                                first_index[type], tier_rings, test[type]);
      }

      /* we descend to the first descendant search area and search there */
      level_num[++(*Level)] = 0;
      level_idx2 += CUDA_ITER_STRIDE;
    }

    for (;;) {
      /* if we tried to advance the search area on start_level, we've completed
       * the search */
      if (level_num[start_level] > 0) {
        return;
      }

      /* if we have tried to advance the search area past the number of
       * descendants, that means that we have completed all of the branches on
       * this level. we can now run the face_iterate for all of the faces between
       * search areas on the level*/
      if (level_num[*Level] == P4EST_CHILDREN) {
        /* for each direction */
        for (dir = 0; dir < P4EST_DIM; dir++) {
          for (side = 0; side < P4EST_CHILDREN / 2; side++) {
            cuda_iter_copy_indices (loop_args,
                                     args->face_args[dir][side].start_idx2,
                                     1, 2);
            p4est_face_iterate_without_callbacks (&(args->face_args[dir][side]), all_faces);
          }
        }
#ifdef P4_TO_P8
        /* if there is an edge or a corner callback, we need to use
         * edge_iterate, so we set up the common corners and edge ids
         * for all of the edges between the search areas */
        if (loop_args->loop_edge) {
          for (dir = 0; dir < P4EST_DIM; dir++) {
            for (side = 0; side < 2; side++) {
              cuda_iter_copy_indices (loop_args,
                                       args->edge_args[dir][side].start_idx2,
                                       1, 4);
              p8est_edge_iterate (&(args->edge_args[dir][side]), user_data,
                                  iter_edge, iter_corner);
            }
          }
        }
#endif
        /* we are done at the level, so we go up a level and over a branch */
        level_num[--(*Level)]++;
        level_idx2 -= CUDA_ITER_STRIDE;
      }
      else {
        /* quad_idx now gives the location in index[type] of the bounds
         * of the current search area, from which we get the first quad
         * and the count */
        quad_idx2 = level_idx2 + level_num[*Level];
        for (type = local; type <= ghost; type++) {
          first_index[type] = zindex[type][quad_idx2];
          count[type] = zindex[type][quad_idx2 + 1] - first_index[type];
        }
        if (!count[local]) {
          /* if there are no local quadrants, we are done with this search area,
           * and we advance to the next branch at this level */
          level_num[*Level]++;
        }
        else {
          /* otherwise we are done changing the search area */
          break;
        }
      }
    }
  }
  P4EST_ASSERT (*Level == start_level);
}

void mallocFacesSides(cuda4est_t* cuda4est, sc_array_t* quadrants, p4est_quadrants_to_cuda* quads_to_cuda, p4est_ghost_t* Ghost_layer, p4est_ghost_to_cuda_t* ghost_to_cuda) {
  std::vector<p4est_iter_face_side_t*> all_faces;
  int remote = 0;
  p4est_t *p4est = cuda4est->p4est;
  int                 f, c;
  p4est_topidx_t      t;
  p4est_ghost_t       empty_ghost_layer;
  p4est_ghost_t      *ghost_layer;
  sc_array_t         *trees = p4est->trees;
  p4est_connectivity_t *conn = p4est->connectivity;
  size_t              global_num_trees = trees->elem_count;
  p4est_iter_loop_args_t *loop_args;
  p4est_iter_face_args_t face_args;
#ifdef P4_TO_P8
  int                 e;
  p8est_iter_edge_args_t edge_args;
#endif
  p4est_iter_corner_args_t corner_args;
  p4est_iter_volume_args_t args;
  p4est_topidx_t      first_local_tree = p4est->first_local_tree;
  p4est_topidx_t      last_local_tree = p4est->last_local_tree;
  p4est_topidx_t      last_run_tree;
  int32_t            *owned;
  int32_t             mask, touch;

  p4est_tree_t* first_tree = p4est_tree_array_index(p4est->trees, first_local_tree);

  P4EST_ASSERT (p4est_is_valid (p4est));

  if (p4est->first_local_tree < 0) {
    return;
  }

  if (Ghost_layer == NULL) {
    sc_array_init (&(empty_ghost_layer.ghosts), sizeof (p4est_quadrant_t));
    empty_ghost_layer.tree_offsets = P4EST_ALLOC_ZERO (p4est_locidx_t,
                                                       global_num_trees + 1);
    empty_ghost_layer.proc_offsets = P4EST_ALLOC_ZERO (p4est_locidx_t,
                                                       p4est->mpisize + 1);
    ghost_layer = &empty_ghost_layer;
  }
  else {
    ghost_layer = Ghost_layer;
  }

  /** initialize arrays that keep track of where we are in the search */
  loop_args = cuda_iter_loop_args_new (conn,
#ifdef P4_TO_P8
                                        NULL,
#endif
                                        NULL, ghost_layer,
                                        p4est->mpisize);

  owned = cuda_iter_get_boundaries (p4est, &last_run_tree, remote);
  last_run_tree = (last_run_tree < last_local_tree) ? last_local_tree :
    last_run_tree;

  /* start with the assumption that we only run on entities touches by the
   * local processor's domain */
  args.remote = remote;
  face_args.remote = remote;
#ifdef P4_TO_P8
  edge_args.remote = remote;
#endif
  corner_args.remote = remote;

  /** we have to loop over all trees and not just local trees because of the
   * ghost layer */
  for (t = first_local_tree; t <= last_run_tree; t++) {
    if (t >= first_local_tree && t <= last_local_tree) {
      cuda_iter_init_volume (&args, p4est, ghost_layer, loop_args, t);

      cuda_volume_iterate_without_callbacks (&args, &all_faces);

      cuda_iter_reset_volume (&args);
    }
  }

  if (Ghost_layer == NULL) {
    P4EST_FREE (empty_ghost_layer.tree_offsets);
    P4EST_FREE (empty_ghost_layer.proc_offsets);
  }
  P4EST_FREE (owned);
  cuda_iter_loop_args_destroy (loop_args);

  size_t quads_length = quads_to_cuda->quadrants_length;
  int8_t *quad_is_used = (int8_t*)malloc(sizeof(int8_t) * quads_length);
  std::vector<p4est_iter_face_side_t*>::iterator faces_iter = all_faces.end();
  std::vector<std::vector<p4est_iter_face_side_t*>* > iterations_to_faces;
  size_t faces_count = all_faces.size();
  quads_to_cuda->sides_count = faces_count;

  std::vector<std::vector<p4est_iter_face_side_t*>* >::iterator iteration_faces_iter = iterations_to_faces.begin();
  for(;;) {
    if(faces_iter == all_faces.end()) {
      size_t faces_size = all_faces.size();
      if(!all_faces.size()){
        break;
      }
      faces_iter = all_faces.begin();
      std::vector<p4est_iter_face_side_t*> *new_iteration = new std::vector<p4est_iter_face_side_t*>();
      iterations_to_faces.push_back(new_iteration);
      iteration_faces_iter = iterations_to_faces.end();
      iteration_faces_iter--;
      for(size_t i = 0; i < quads_length; i++){
        quad_is_used[i] = 0;
      }
    }
    p4est_iter_face_side_t *left = *faces_iter;
    faces_iter++;
    p4est_iter_face_side_t *right = *faces_iter;
    faces_iter++;
    bool faces_iter_end_test = faces_iter == all_faces.end();
    if(faces_iter_end_test) {
      faces_iter_end_test = true;
    }
    if(left->is_hanging) {
      p4est_locidx_t *left_quads = left->is.hanging.quadid;
      if(right->is_hanging) {
        p4est_locidx_t *right_quads = right->is.hanging.quadid;
        int8_t facesIsUnused = 1;
        for(size_t i = 0; i < P4EST_HALF; i++) {
          if(quad_is_used[left_quads[i]]) {
            facesIsUnused = 0;
            break;
          }
        }
        if(facesIsUnused) {
          for(size_t i = 0; i < P4EST_HALF; i++) {
            if(quad_is_used[right_quads[i]]) {
              facesIsUnused = 0;
              break;
            }
          }
        }
        if(facesIsUnused) {
          for(size_t i = 0; i < P4EST_HALF; i++) {
            quad_is_used[left_quads[i]] = quad_is_used[right_quads[i]] = 1;
          }
          (*iteration_faces_iter)->push_back(left);
          (*iteration_faces_iter)->push_back(right);
          faces_iter = all_faces.erase(--faces_iter);
          faces_iter = all_faces.erase(--faces_iter);
        }
      } else {
        p4est_locidx_t right_quad = right->is.full.quadid;
        int8_t facesIsUnused = !quad_is_used[right_quad];
        if(facesIsUnused) {
          for(size_t i = 0; i < P4EST_HALF; i++) {
            if(quad_is_used[left_quads[i]]) {
              facesIsUnused = 0;
              break;
            }
          }
          if(facesIsUnused) {
            quad_is_used[right_quad] = 1;
            (*iteration_faces_iter)->push_back(left);
            for(size_t i = 0; i < P4EST_HALF; i++) {
              quad_is_used[left_quads[i]] = 1;
            }
            (*iteration_faces_iter)->push_back(right);
            faces_iter = all_faces.erase(--faces_iter);
            faces_iter = all_faces.erase(--faces_iter);
          }
        }
      }
    } else {
      p4est_locidx_t left_quad = left->is.full.quadid;
      if(right->is_hanging) {
        p4est_locidx_t *right_quads = right->is.hanging.quadid;
        int8_t facesIsUnused = !quad_is_used[left_quad];
        if(facesIsUnused) {
          for(size_t i = 0; i < P4EST_HALF; i++) {
            if(quad_is_used[right_quads[i]]) {
              facesIsUnused = 0;
              break;
            }
          }
          if(facesIsUnused) {
            quad_is_used[left_quad] = 1;
            (*iteration_faces_iter)->push_back(left);
            for(size_t i = 0; i < P4EST_HALF; i++) {
              quad_is_used[right_quads[i]] = 1;
            }
            (*iteration_faces_iter)->push_back(right);
            faces_iter = all_faces.erase(--faces_iter);
            faces_iter = all_faces.erase(--faces_iter);
          }
        }
      } else {
        p4est_locidx_t right_quad = right->is.full.quadid;
        if(!quad_is_used[right_quad] && ! quad_is_used[left_quad]) {
          bool faces_iter_end = faces_iter == all_faces.end();
          size_t facess_size = all_faces.size();
          quad_is_used[right_quad] = quad_is_used[left_quad] = 1;
          (*iteration_faces_iter)->push_back(left);
          (*iteration_faces_iter)->push_back(right);
          faces_iter = all_faces.erase(--faces_iter);
          faces_iter = all_faces.erase(--faces_iter);
        }
      }
    } 
  }

  free(quad_is_used);


  quads_to_cuda->faces_iteration_count = iterations_to_faces.size();
  size_t *faces_per_iter = (size_t*)malloc(sizeof(size_t) * quads_to_cuda->faces_iteration_count);
  size_t *faces_per_iter_cursor = faces_per_iter;
  size_t face_size = sizeof(p4est_iter_face_side_t);
  size_t faces_bytes_alloc = faces_count * face_size;
  p4est_iter_face_side_t* sides_arr = (p4est_iter_face_side_t*)malloc(faces_bytes_alloc);
  p4est_iter_face_side_t* cursor = sides_arr;
  for(std::vector<std::vector<p4est_iter_face_side_t*>* >::iterator iteration_i = iterations_to_faces.begin(); iteration_i != iterations_to_faces.end(); iteration_i++, faces_per_iter_cursor++) {
    *faces_per_iter_cursor = (*iteration_i)->size();
    for(std::vector<p4est_iter_face_side_t*>::iterator face_i = (*iteration_i)->begin(); face_i != (*iteration_i)->end(); face_i++, cursor++) {
      memcpy(cursor, *face_i, face_size);
    }
    delete *iteration_i;
  }
  quads_to_cuda->faces_per_iter = faces_per_iter;

  quads_to_cuda->h_sides = sides_arr;
  p4est_iter_face_side_t* d_faces;
  gpuErrchk(cudaMalloc((void**)&d_faces, faces_bytes_alloc));
  gpuErrchk(cudaMemcpy(d_faces, sides_arr, faces_bytes_alloc, cudaMemcpyHostToDevice));
  quads_to_cuda->d_sides = d_faces;
}

void freeMemoryForFacesSides(p4est_quadrants_to_cuda* quads_to_cuda) {
  gpuErrchk(cudaFree(quads_to_cuda->d_sides));
  free(quads_to_cuda->h_sides);
}

p4est_ghost_to_cuda_t* mallocForGhost(p4est_t* p4est, p4est_ghost_t* ghost_layer) {
    p4est_ghost_to_cuda_t *ghost_to_cuda = (p4est_ghost_to_cuda_t*) malloc(sizeof(p4est_ghost_to_cuda_t));
    sc_array_t *trees = p4est->trees;

    size_t num_trees = trees->elem_count;
    size_t mpisize = p4est->mpisize;

    p4est_ghost_t *d_ghost_layer;
    gpuErrchk(cudaMalloc((void**)&d_ghost_layer, sizeof(p4est_ghost_t)));
    gpuErrchk(cudaMemcpy(d_ghost_layer, ghost_layer, sizeof(p4est_ghost_t), cudaMemcpyHostToDevice));
    ghost_to_cuda->d_ghost_layer = d_ghost_layer;

    char *d_ghosts_array_temp;
    size_t d_ghosts_array_temp_size = ghost_layer->ghosts.elem_size * ghost_layer->ghosts.elem_count;
    p4est_locidx_t *d_tree_offsets_temp, *d_proc_offsets_temp;
    size_t d_tree_offsets_temp_size = (num_trees + 1) * sizeof(p4est_locidx_t);
    size_t d_proc_offsets_temp_size = (mpisize + 1) * sizeof(p4est_locidx_t);

    gpuErrchk(cudaMalloc((void**)&d_ghosts_array_temp, d_ghosts_array_temp_size));
    gpuErrchk(cudaMemcpy(&(d_ghost_layer->ghosts.array), &d_ghosts_array_temp, sizeof(char*), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_ghosts_array_temp, ghost_layer->ghosts.array, d_ghosts_array_temp_size, cudaMemcpyHostToDevice));
    ghost_to_cuda->d_ghosts_array_temp = d_ghosts_array_temp;
    ghost_to_cuda->d_ghosts_array_temp_size = d_ghosts_array_temp_size;

    gpuErrchk(cudaMalloc((void**)&d_tree_offsets_temp, d_tree_offsets_temp_size));
    gpuErrchk(cudaMemcpy(&(d_ghost_layer->tree_offsets),&d_tree_offsets_temp, sizeof(p4est_locidx_t*), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_tree_offsets_temp, ghost_layer->tree_offsets, d_tree_offsets_temp_size, cudaMemcpyHostToDevice));
    ghost_to_cuda->d_tree_offsets_temp = d_tree_offsets_temp;
    ghost_to_cuda->d_tree_offsets_temp_size = d_tree_offsets_temp_size;

    gpuErrchk(cudaMalloc((void**)&d_proc_offsets_temp, d_proc_offsets_temp_size));
    gpuErrchk(cudaMemcpy(&(d_ghost_layer->proc_offsets), &d_proc_offsets_temp, sizeof(p4est_locidx_t*), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_proc_offsets_temp, ghost_layer->proc_offsets, d_proc_offsets_temp_size, cudaMemcpyHostToDevice));
    ghost_to_cuda->d_proc_offsets_temp = d_proc_offsets_temp;
    ghost_to_cuda->d_proc_offsets_temp_size = d_proc_offsets_temp_size;

    return ghost_to_cuda;
}

void freeMemoryForGhost(p4est_ghost_to_cuda_t* ghost_to_cuda) {
  if(ghost_to_cuda->d_ghosts_array_temp) {
    gpuErrchk(cudaFree(ghost_to_cuda->d_ghosts_array_temp));
  }
  if(ghost_to_cuda->d_tree_offsets_temp) {
    gpuErrchk(cudaFree(ghost_to_cuda->d_tree_offsets_temp));
  }
  if(ghost_to_cuda->d_proc_offsets_temp) {
    gpuErrchk(cudaFree(ghost_to_cuda->d_proc_offsets_temp));
  }
  if(ghost_to_cuda->d_ghost_layer) {
    gpuErrchk(cudaFree(ghost_to_cuda->d_ghost_layer));
  }
}


sc_mempool_cuda_memory_allocate_info_t* scMempoolMemoryAllocate(sc_mempool_t* mempool) {
    sc_mempool_cuda_memory_allocate_info_t *mempool_cuda_info = (sc_mempool_cuda_memory_allocate_info_t*) malloc(sizeof(sc_mempool_cuda_memory_allocate_info_t));

    sc_mempool_t *d_mempool;
    gpuErrchk(cudaMalloc((void**)&d_mempool, sizeof(sc_mempool)));
    gpuErrchk(cudaMemcpy(d_mempool, mempool, sizeof(sc_mempool), cudaMemcpyHostToDevice));
    mempool_cuda_info->d_mempool = d_mempool;

    sc_mstamp_cuda_memory_allocate_info_t *mstamp_memory_allocate_info = scMstampMemoryAllocate(&mempool->mstamp);
    gpuErrchk(cudaMemcpy(&(d_mempool->mstamp), &mstamp_memory_allocate_info->d_mstamp, sizeof(sc_mstamp*), cudaMemcpyHostToDevice));

    mempool_cuda_info->mstamp_memory_allocate_info = mstamp_memory_allocate_info;

    sc_array_cuda_memory_allocate_info_t *freed_memory_allocate_info = scArrayMemoryAllocate(&mempool->freed);
    gpuErrchk(cudaMemcpy(&(d_mempool->freed), freed_memory_allocate_info->d_sc_arr, sizeof(sc_array*), cudaMemcpyHostToDevice));
    mempool_cuda_info->freed_memory_allocate_info = freed_memory_allocate_info;
    return mempool_cuda_info;
}

void scMempoolMemoryFree(sc_mempool_cuda_memory_allocate_info_t* mempool_cuda_info) {
    scArrayMemoryFree(mempool_cuda_info->freed_memory_allocate_info);
    scMstampMemoryFree(mempool_cuda_info->mstamp_memory_allocate_info);
    gpuErrchk(cudaFree(mempool_cuda_info->d_mempool));
}


p4est_connectivity_cuda_memory_allocate_info_t* p4est_connectivity_memory_alloc(p4est_connectivity_t* conn) {
    p4est_connectivity_cuda_memory_allocate_info_t *allocate_info = (p4est_connectivity_cuda_memory_allocate_info_t*) malloc(sizeof(p4est_connectivity_cuda_memory_allocate_info_t));
    
    size_t conn_num_trees = conn->num_trees;
    size_t conn_num_vertices = conn->num_vertices;
    size_t conn_num_corners = conn->num_corners;
    
    p4est_connectivity_t *d_connectivity;
    gpuErrchk(cudaMalloc((void**)&d_connectivity, sizeof(p4est_connectivity_t)));
    gpuErrchk(cudaMemcpy(d_connectivity, conn, sizeof(p4est_connectivity_t), cudaMemcpyHostToDevice));
    allocate_info->d_connectivity = d_connectivity;

    double *d_vertices;
    size_t d_vertices_size = conn_num_vertices * sizeof(double);
    arrayPropMemoryAllocate((void**)&d_vertices, d_vertices_size, &(d_connectivity->vertices), sizeof(double*), conn->vertices);
    allocate_info->d_vertices = d_vertices;
    allocate_info->d_vertices_size = d_vertices_size;

    p4est_topidx_t *d_tree_to_vertex;
    size_t d_tree_to_vertex_size = P4EST_CHILDREN * conn_num_trees * sizeof(p4est_topidx_t);
    arrayPropMemoryAllocate((void**)&d_tree_to_vertex, d_tree_to_vertex_size, &(d_connectivity->tree_to_vertex), sizeof(p4est_topidx_t*), conn->tree_to_vertex);
    allocate_info->d_tree_to_vertex = d_tree_to_vertex;
    allocate_info->d_tree_to_vertex_size = d_tree_to_vertex_size;

    char *d_tree_to_attr;
    size_t d_tree_to_attr_size = conn->tree_attr_bytes * sizeof(char) * conn_num_trees;
    arrayPropMemoryAllocate((void**)&d_tree_to_attr, d_tree_to_attr_size, &(d_connectivity->tree_to_attr), sizeof(char*), conn->tree_to_attr);
    allocate_info->d_tree_to_attr = d_tree_to_attr;
    allocate_info->d_tree_to_attr_size = d_tree_to_attr_size;

    p4est_topidx_t *d_tree_to_tree;
    size_t d_tree_to_tree_size = P4EST_FACES * conn_num_trees * sizeof(p4est_topidx_t);
    arrayPropMemoryAllocate((void**)&d_tree_to_tree, d_tree_to_tree_size, &(d_connectivity->tree_to_tree), sizeof(p4est_topidx_t*), conn->tree_to_tree);
    allocate_info->d_tree_to_tree = d_tree_to_tree;
    allocate_info->d_tree_to_tree_size = d_tree_to_tree_size;

    int8_t *d_tree_to_face;
    size_t d_tree_to_face_size = P4EST_FACES * conn_num_trees * sizeof(int8_t);
    arrayPropMemoryAllocate((void**)&d_tree_to_face, d_tree_to_face_size, &(d_connectivity->tree_to_face), sizeof(int8_t*), conn->tree_to_face);
    allocate_info->d_tree_to_face = d_tree_to_face;
    allocate_info->d_tree_to_face_size = d_tree_to_face_size;

    p4est_topidx_t *d_tree_to_corner;
    size_t d_tree_to_corner_size = P4EST_CHILDREN * conn_num_trees * sizeof(p4est_topidx_t);
    arrayPropMemoryAllocate((void**)&d_tree_to_corner, d_tree_to_corner_size, &(d_connectivity->tree_to_corner), sizeof(p4est_topidx_t*), conn->tree_to_corner);
    allocate_info->d_tree_to_corner = d_tree_to_corner;
    allocate_info->d_tree_to_corner_size = d_tree_to_corner_size;

    p4est_topidx_t *d_ctt_offset;
    size_t d_ctt_offset_size = conn_num_corners + 1;
    arrayPropMemoryAllocate((void**)&d_ctt_offset, d_ctt_offset_size, &(d_connectivity->ctt_offset), sizeof(p4est_topidx_t*), conn->ctt_offset);
    allocate_info->d_ctt_offset = d_ctt_offset;
    allocate_info->d_ctt_offset_size = d_ctt_offset_size;

    p4est_topidx_t *d_corner_to_tree;
    size_t d_corner_to_tree_size = conn->ctt_offset[conn_num_corners] * sizeof(p4est_topidx_t);
    arrayPropMemoryAllocate((void**)&d_corner_to_tree, d_corner_to_tree_size, &(d_connectivity->corner_to_tree), sizeof(p4est_topidx_t*), conn->corner_to_tree);
    allocate_info->d_corner_to_tree = d_corner_to_tree;
    allocate_info->d_corner_to_corner_size = d_corner_to_tree_size;

    int8_t *d_corner_to_corner;
    size_t d_corner_to_corner_size = d_corner_to_tree_size;
    arrayPropMemoryAllocate((void**)&d_corner_to_corner, d_corner_to_corner_size, &(d_connectivity->corner_to_corner), sizeof(int8_t), conn->corner_to_corner);
    allocate_info->d_corner_to_corner = d_corner_to_corner;
    allocate_info->d_corner_to_corner_size = d_corner_to_corner_size;

    return allocate_info;
}

void p4est_connectivity_memory_free(p4est_connectivity_cuda_memory_allocate_info_t* allocate_info) {
    gpuErrchk(cudaFree(allocate_info->d_vertices));
    gpuErrchk(cudaFree(allocate_info->d_tree_to_vertex));
    gpuErrchk(cudaFree(allocate_info->d_tree_to_attr));
    gpuErrchk(cudaFree(allocate_info->d_tree_to_tree));
    gpuErrchk(cudaFree(allocate_info->d_tree_to_face));
    gpuErrchk(cudaFree(allocate_info->d_tree_to_corner));
    gpuErrchk(cudaFree(allocate_info->d_ctt_offset));
    gpuErrchk(cudaFree(allocate_info->d_corner_to_tree));
    gpuErrchk(cudaFree(allocate_info->d_corner_to_corner));
    gpuErrchk(cudaFree(allocate_info->d_connectivity));
}

p4est_cuda_memory_allocate_info_t* p4est_memory_alloc(cuda4est_t* cuda4est) {
    p4est_cuda_memory_allocate_info_t *p4est_allocate_info = (p4est_cuda_memory_allocate_info_t*) malloc(sizeof(p4est_cuda_memory_allocate_info_t));

    p4est_t *p4est = cuda4est->p4est;
    sc_array_t *trees = p4est->trees;
    size_t mpisize = p4est->mpisize;
    p4est_connectivity_t *conn = p4est->connectivity;

    sc_mstamp *ms_stamp = (sc_mstamp*)p4est->user_data_pool->mstamp.remember.array;

    p4est_t *d_p4est;
    gpuErrchk(cudaMalloc((void**)&d_p4est, sizeof(p4est_t)));
    gpuErrchk(cudaMemcpy(d_p4est, p4est, sizeof(p4est_t), cudaMemcpyHostToDevice));
    p4est_allocate_info->d_p4est = d_p4est;

    cuda4est->user_data_api->alloc_cuda_memory(cuda4est->user_data_api);
    void *d_user_pointer = cuda4est->user_data_api->get_cuda_allocated_user_data(cuda4est->user_data_api);
    gpuErrchk(cudaMemcpy(&(d_p4est->user_pointer), &d_user_pointer, sizeof(void*), cudaMemcpyHostToDevice));
    p4est_allocate_info->d_user_data_api = cuda4est->user_data_api;

    p4est_gloidx_t *d_global_first_quadrant;
    size_t d_global_first_quadrant_size = (mpisize + 1)* sizeof(p4est_gloidx_t);
    arrayPropMemoryAllocate((void**)&d_global_first_quadrant, d_global_first_quadrant_size, &(d_p4est->global_first_quadrant), sizeof(p4est_gloidx_t*), p4est->global_first_quadrant);
    p4est_allocate_info->d_global_first_quadrant = d_global_first_quadrant;
    p4est_allocate_info->d_global_first_quadrant_size = d_global_first_quadrant_size;

    size_t d_global_first_position_length = mpisize + 1;
    array_allocation_cuda_info_t *d_global_first_position_allocate_info = cuda_quadrants_memory_allocate(p4est->global_first_position, d_global_first_position_length, cuda4est->quad_user_data_api);
    p4est_allocate_info->d_global_first_position_allocate_info = d_global_first_position_allocate_info;
    p4est_quadrant_t *d_global_first_position = d_global_first_position_allocate_info->d_quadrants;
    gpuErrchk(cudaMemcpy(&(d_p4est->global_first_position), &d_global_first_position, sizeof(p4est_quadrant_t*), cudaMemcpyHostToDevice));

    p4est_connectivity_cuda_memory_allocate_info_t *connectivity_allocate_info = p4est_connectivity_memory_alloc(conn);
    gpuErrchk(cudaMemcpy(&(d_p4est->connectivity), &(connectivity_allocate_info->d_connectivity), sizeof(p4est_connectivity_t*), cudaMemcpyHostToDevice));
    p4est_allocate_info->d_connectivity_cuda_allocate_info = connectivity_allocate_info;
    
    sc_array_cuda_memory_allocate_info_t *d_trees_memory_allocate_info = scArrayMemoryAllocate(trees);
    sc_array_t *d_trees = d_trees_memory_allocate_info->d_sc_arr;
    gpuErrchk(cudaMemcpy(&(d_p4est->trees), &d_trees, sizeof(sc_array_t*), cudaMemcpyHostToDevice ));
    p4est_allocate_info->d_trees_cuda_allocate_info = d_trees_memory_allocate_info;

    // not copy user_data_mempool
    sc_mempool_t *d_user_data_pool;
    //sc_mempool_cuda_memory_allocate_info_t *d_user_data_pool_memory_allocate_info = scMempoolMemoryAllocate(p4est->user_data_pool);
    //sc_mempool_t *d_user_data_pool = d_user_data_pool_memory_allocate_info->d_mempool;
    gpuErrchk(cudaMalloc((void**)&d_user_data_pool, sizeof(sc_mempool_t)));
    gpuErrchk(cudaMemcpy(d_user_data_pool, p4est->user_data_pool, sizeof(sc_mempool_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_p4est->user_data_pool), &d_user_data_pool, sizeof(sc_mempool_t*), cudaMemcpyHostToDevice));
    p4est_allocate_info->d_user_data_pool = d_user_data_pool;
    // p4est_allocate_info->d_user_data_pool_cuda_allocate_info = d_user_data_pool_memory_allocate_info;

    sc_mempool_cuda_memory_allocate_info_t *d_quadrant_pool_memory_allocate_info = scMempoolMemoryAllocate(p4est->quadrant_pool);
    sc_mempool_t *d_quadrant_pool = d_quadrant_pool_memory_allocate_info->d_mempool;
    gpuErrchk(cudaMemcpy(&(d_p4est->quadrant_pool), &d_quadrant_pool, sizeof(sc_mempool_t*), cudaMemcpyHostToDevice));
    p4est_allocate_info->d_quadrant_pool_cuda_allocate_info = d_quadrant_pool_memory_allocate_info;

    p4est_inspect_cuda_memory_allocate_info_t *d_p4est_inspect_memory_allocate_info = p4est_inspect_memory_allocate(p4est->inspect);
    p4est_inspect_t *d_inspect = d_p4est_inspect_memory_allocate_info->d_p4est_inspect;
    gpuErrchk(cudaMemcpy(&(d_p4est->inspect), &d_inspect, sizeof(p4est_inspect_t*), cudaMemcpyHostToDevice));
    p4est_allocate_info->d_inspect_cuda_allocate_info = d_p4est_inspect_memory_allocate_info;

    return p4est_allocate_info;
}

void p4est_memory_free(p4est_cuda_memory_allocate_info_t* allocate_info, quad_user_data_api_t *quad_user_data_api) {
    allocate_info->d_user_data_api->free_cuda_memory(allocate_info->d_user_data_api);
    gpuErrchk(cudaFree(allocate_info->d_global_first_quadrant));
    cuda_quadrants_memory_free(allocate_info->d_global_first_position_allocate_info, quad_user_data_api);
    p4est_connectivity_memory_free(allocate_info->d_connectivity_cuda_allocate_info);
    scArrayMemoryFree(allocate_info->d_trees_cuda_allocate_info);
    // scMempoolMemoryFree(allocate_info->d_user_data_pool_cuda_allocate_info);
    gpuErrchk(cudaFree(allocate_info->d_user_data_pool));
    scMempoolMemoryFree(allocate_info->d_quadrant_pool_cuda_allocate_info);
    p4est_inspect_memory_free(allocate_info->d_inspect_cuda_allocate_info);
    gpuErrchk(cudaFree(allocate_info->d_p4est));
}