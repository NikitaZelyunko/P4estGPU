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
  freeMemoryForQuadrantsBlocks(quads_to_cuda);
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

p4est_iter_face_side_t* copy_iter_face_sides(sc_array* sides) {
  size_t face_bytes_size = sizeof(p4est_iter_face_side_t) * 2;
  p4est_iter_face_side_t* face_side = (p4est_iter_face_side_t*)malloc(face_bytes_size);
  memcpy(face_side, sides->array, face_bytes_size);
  return face_side;
}

static void
p4est_face_iterate_without_callbacks (p4est_iter_face_args_t * args, std::vector<p4est_iter_face_side_t>* all_faces)
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
          //all_faces->push_back(copy_iter_face_side(fside));
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
          //all_faces->push_back(copy_iter_face_side(fside));
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
          //all_faces->push_back(copy_iter_face_side(fside));
        }
      }
    }
    if (stop_refine) {
      if (has_local) {
        p4est_iter_face_side_t *faces = copy_iter_face_sides(&info->sides);
        all_faces->push_back(*(faces++));
        all_faces->push_back(*faces);

        //for (side = left; side <= limit; side++) {
          //all_faces->push_back(copy_iter_face_side(cuda_iter_fside_array_index_int (&info->sides, side)));
          //all_faces->push_back(cuda_iter_fside_array_index_int (&info->sides, side)[0]);
        //}
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
cuda_volume_iterate_without_callbacks (p4est_iter_volume_args_t * args, std::vector<p4est_iter_face_side_t>* all_faces)
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
  size_t quads_count = quadrants[0]->elem_count;
  /*
  for(size_t i = 0; i < quads_count; i++) {
    p4est_quadrant* test_quad = p4est_quadrant_array_index (quadrants[0], i);
    printf("quad_level_%d: %d\n", i, test_quad->level);
  }
  */

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
          //if (type == local) {
          //  info->quad = test[type];
          //  info->quadid = (p4est_locidx_t) first_index[type];
            //if (iter_volume != NULL) {
            //  iter_volume (info, user_data);
            //}
          //}
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
  clock_t start = clock();
  clock_t stop = clock();
  double duration = (double)(stop - start) / CLOCKS_PER_SEC;

  std::vector<p4est_iter_face_side_t> all_faces;
  //std::vector<p4est_iter_face_side_t*> all_faces;
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
  start = clock();
  for (t = first_local_tree; t <= last_run_tree; t++) {
    if (t >= first_local_tree && t <= last_local_tree) {
      cuda_iter_init_volume (&args, p4est, ghost_layer, loop_args, t);

      cuda_volume_iterate_without_callbacks (&args, &all_faces);

      cuda_iter_reset_volume (&args);
    }
    touch = owned[t];
    if (!touch) {
      continue;
    }
    mask = 0x00000001;
    /* Now we need to run face_iterate on the faces between trees */
    for (f = 0; f < 2 * P4EST_DIM; f++, mask <<= 1) {
      if ((touch & mask) == 0) {
        continue;
      }
      cuda_iter_init_face (&face_args, p4est, ghost_layer, loop_args, t, f);
      p4est_face_iterate_without_callbacks(&face_args, &all_faces);
      //p4est_face_iterate (&face_args, user_data, iter_face,
      //#ifdef P4_TO_P8
      //                        iter_edge,
      //#endif
      //                        iter_corner);
      cuda_iter_reset_face (&face_args);
    }
  }
  stop = clock();
  duration = (double)(stop - start) / CLOCKS_PER_SEC;

  //cout << "Time taken by create_vector_sides: "
  //      << duration << " seconds" << endl;


  if (Ghost_layer == NULL) {
    P4EST_FREE (empty_ghost_layer.tree_offsets);
    P4EST_FREE (empty_ghost_layer.proc_offsets);
  }
  P4EST_FREE (owned);
  //cuda_iter_loop_args_destroy (loop_args);

  size_t quads_length = quads_to_cuda->quadrants_length;
  unsigned char *quad_is_used = (unsigned char*)malloc(sizeof(unsigned char) * quads_length);
  unsigned char *ghost_quad_is_used = (unsigned char*)malloc(sizeof(unsigned char) * (ghost_layer->ghosts.elem_count + 1));
  ghost_quad_is_used++;
  //std::vector<p4est_iter_face_side_t>::iterator faces_iter = all_faces.end();
  //std::vector<std::vector<p4est_iter_face_side_t>* > iterations_to_faces;
  std::vector<p4est_iter_face_side_t>::iterator faces_iter = all_faces.end();
  //std::vector<std::vector<p4est_iter_face_side_t*>* > iterations_to_faces;
  size_t faces_count = all_faces.size();
  quads_to_cuda->sides_count = faces_count;
  start = clock();
  p4est_iter_face_side_t* sides_array = all_faces.data();
  size_t faces_length = faces_count/2;
  unsigned char *faces_iteration_indexes = (unsigned char*)malloc(faces_length * sizeof(unsigned char));

  for(size_t i = 0; i < quads_length; i++){
    quad_is_used[i] = 0;
  }
  ghost_quad_is_used[-1] = 0;
  for(size_t i = 0; i < ghost_layer->ghosts.elem_count; i++) {
    ghost_quad_is_used[i] = 0;
  }
  size_t *faces_per_iteration = (size_t*)malloc(8 * sizeof(size_t));
  for(int i = 0; i < 8; i++) {
    faces_per_iteration[i] = 0;
  }

  size_t quadrant_size = sizeof(p4est_quadrant_t);
  unsigned char lazy = 0;
  for(size_t face_index = 0, j = 0; face_index < faces_length; face_index++, lazy = 0) {
    unsigned char *a = &(lazy);
    unsigned char *b = &(lazy);
    unsigned char *c = &(lazy);
    p4est_iter_face_side_t* left = &sides_array[j++];
    if(left->is_hanging) {
      p4est_locidx_t * left_ids = left->is.hanging.quadid;
      if(left_ids[0] != -1) {
        if(left->is.hanging.is_ghost[0]) {
          a = &(ghost_quad_is_used[left_ids[0]]);
          left->is.hanging.quad[0] = (p4est_quadrant_t*)(ghost_to_cuda->d_ghosts_array_temp + left_ids[0]*quadrant_size);
        } else {
          a = &(quad_is_used[left_ids[0]]);
          left->is.hanging.quad[0] = quads_to_cuda->d_quadrants_array_temp + left_ids[0];
        }
      }
      if(left_ids[1] != -1) {
        if(left->is.hanging.is_ghost[1]) {
          b = &(ghost_quad_is_used[left_ids[1]]);
          left->is.hanging.quad[1] = (p4est_quadrant_t*)(ghost_to_cuda->d_ghosts_array_temp + left_ids[1]*quadrant_size);
        } else {
          b = &(quad_is_used[left_ids[1]]);
          left->is.hanging.quad[1] = quads_to_cuda->d_quadrants_array_temp + left_ids[1];
        }
      }
    } else {
      p4est_locidx_t left_id = left->is.full.quadid;
      if(left_id != -1) {
        if(left->is.full.is_ghost) {
          a = &(ghost_quad_is_used[left_id]);
          left->is.full.quad = (p4est_quadrant_t*)(ghost_to_cuda->d_ghosts_array_temp + left_id*quadrant_size);
        } else {
          a = &(quad_is_used[left_id]);
          left->is.full.quad = quads_to_cuda->d_quadrants_array_temp + left_id;
        }
      }
    }
    p4est_iter_face_side_t *right = &sides_array[j++];
    if(right->is_hanging) {
      p4est_locidx_t * right_ids = right->is.hanging.quadid;
      if(right_ids[0] != -1) {
        if(right->is.hanging.is_ghost[0]) {
          b = &(ghost_quad_is_used[right_ids[0]]);
          right->is.hanging.quad[0] = (p4est_quadrant_t*)(ghost_to_cuda->d_ghosts_array_temp + right_ids[0]*quadrant_size);
        } else {
          b = &(quad_is_used[right_ids[0]]);
          right->is.hanging.quad[0] = quads_to_cuda->d_quadrants_array_temp + right_ids[0];
        }
      }
      if(right_ids[1] != -1) {
        if(right->is.hanging.is_ghost[1]) {
          c = &(ghost_quad_is_used[right_ids[1]]);
          right->is.hanging.quad[1] = (p4est_quadrant_t*)(ghost_to_cuda->d_ghosts_array_temp + right_ids[1]*quadrant_size);
        } else {
          c = &(quad_is_used[right_ids[1]]);
          right->is.hanging.quad[1] = quads_to_cuda->d_quadrants_array_temp + right_ids[1];
        }
      }
    } else {
      p4est_locidx_t right_id = right->is.full.quadid;
      if(right_id != -1) {
        if(right->is.full.is_ghost) {
          c = &(ghost_quad_is_used[right_id]);
          right->is.full.quad = (p4est_quadrant_t*)(ghost_to_cuda->d_ghosts_array_temp + right_id*quadrant_size);
        } else {
          c = &(quad_is_used[right_id]);
          right->is.full.quad = quads_to_cuda->d_quadrants_array_temp + right_id;
        }
      }
    }
    unsigned char mask = 0x00000001;
    for(int i = 0; i < 8; i++, mask <<= 1) {
      if((*a & mask) != 0x00000000) {
        continue;
      }
      if((*b & mask) != 0x00000000) {
        continue;
      }
      if((*c & mask) != 0x00000000) {
        continue;
      }
      *a |=mask;
      *b |=mask;
      *c |=mask;
      faces_iteration_indexes[face_index] = i;
      faces_per_iteration[i]++;
      break;
    }
  }
  size_t face_side_size = sizeof(p4est_iter_face_side_t);
  size_t sides_bytes_alloc = faces_count * face_side_size;
  p4est_iter_face_side_t* sorted_face_sides = (p4est_iter_face_side_t*)malloc(sides_bytes_alloc);
  size_t *start_indexes = (size_t*)malloc(9 * sizeof(size_t));
  start_indexes[0] = 0;
  size_t faces_iteration_count = 0;
  for(int i = 1; i <= 8; i++) {
    start_indexes[i] = start_indexes[i-1] + (faces_per_iteration[i-1] *= 2);
    if(faces_per_iteration[i-1] != 0) {
      faces_iteration_count++;
    }
  }
  for(size_t i = 0, j = 0; i < faces_length; i++) {
    sorted_face_sides[start_indexes[faces_iteration_indexes[i]]++] = sides_array[j++];
    sorted_face_sides[start_indexes[faces_iteration_indexes[i]]++] = sides_array[j++];
  }
  stop = clock();
  //duration = (double)(stop - start) / CLOCKS_PER_SEC;
  //cout << "Time taken by creating face_iterating_indexes: "
  //      << duration << " seconds" << endl;
  
  /*
  size_t sum_iter_count = 0;
  size_t face_cursor = 0;
  for(size_t i = 0; i < 8; i++) {
    for(size_t i = 0; i < quads_length; i++){
      quad_is_used[i] = 0;
    }
    ghost_quad_is_used[-1] = 0;
    for(size_t i = 0; i < ghost_layer->ghosts.elem_count; i++) {
      ghost_quad_is_used[i] = 0;
    }

    size_t faces_iter_count = faces_per_iteration[i];
    if(faces_iter_count) {
      sum_iter_count+=faces_iter_count;
      printf("%d: faces_iter_count: %d\n", i, faces_iter_count);
    }
    for(size_t j = face_cursor; j < (face_cursor + faces_iter_count); j++) {
      p4est_iter_face_side_t face_side = sorted_face_sides[j];
      if(face_side.is_hanging) {
        if(face_side.is.hanging.is_ghost[0]) {
          if(ghost_quad_is_used[face_side.is.hanging.quadid[0]]) {
            //printf("iter_count: %d, faces_start: %d, face_index_global: %d\n", i, face_cursor, j);
            printf("0 hanging ghost_quad already used: %d\n", face_side.is.hanging.quadid[0]);
          } else {
            ghost_quad_is_used[face_side.is.hanging.quadid[0]] = 1;
          }
        } else {
          if(quad_is_used[face_side.is.hanging.quadid[0]]) {
            //printf("iter_count: %d, faces_start: %d, face_index_global: %d\n", i, face_cursor, j);
            printf("0 hanging quad already used: %d\n", face_side.is.hanging.quadid[0]);
          } else {
            quad_is_used[face_side.is.hanging.quadid[0]] = 1;
          }
        }
        if(face_side.is.hanging.is_ghost[1]) {
          if(ghost_quad_is_used[face_side.is.hanging.quadid[1]]) {
            //printf("iter_count: %d, faces_start: %d, face_index_global: %d\n", i, face_cursor, j);
            printf("1 hanging ghost_quad already used: %d\n", face_side.is.hanging.quadid[1]);
          } else {
            ghost_quad_is_used[face_side.is.hanging.quadid[1]] = 1;
          }
        } else {
          if(quad_is_used[face_side.is.hanging.quadid[1]]) {
            //printf("iter_count: %d, faces_start: %d, face_index_global: %d\n", i, face_cursor, j);
            printf("1 hanging quad already used: %d\n", face_side.is.hanging.quadid[1]);
          } else {
            quad_is_used[face_side.is.hanging.quadid[1]] = 1;
          }
        }
      } else {
        if(face_side.is.full.is_ghost) {
          if(ghost_quad_is_used[face_side.is.full.quadid]) {
            //printf("iter_count: %d, faces_start: %d, face_index_global: %d\n", i, face_cursor, j);
            printf("ghost_quad already used: %d\n", face_side.is.full.quadid);
          } else {
            ghost_quad_is_used[face_side.is.full.quadid] = 1;
          }
        } else {
          if(quad_is_used[face_side.is.full.quadid]) {
            //printf("iter_count: %d, faces_start: %d, face_index_global: %d\n", i, face_cursor, j);
            printf("quad already used: %d\n", face_side.is.full.quadid);
          } else {
            quad_is_used[face_side.is.full.quadid] = 1;
          }
        }
      }
    }
    face_cursor += faces_iter_count;
  }

  if(sum_iter_count == faces_count) {
    printf("count is valid\n");
  } else {
    printf("count is invalid\n");
  }

  if(ghost_layer->ghosts.elem_count) {
    printf("ghosts is available\n");
  }
*/
  quads_to_cuda->h_sides = sorted_face_sides;
  p4est_iter_face_side_t* d_sides;
  gpuErrchk(cudaMalloc((void**)&d_sides, sides_bytes_alloc));
  gpuErrchk(cudaMemcpy(d_sides, sorted_face_sides, sides_bytes_alloc, cudaMemcpyHostToDevice));
  quads_to_cuda->d_sides = d_sides;
  quads_to_cuda->faces_per_iter = faces_per_iteration;
  quads_to_cuda->faces_iteration_count = faces_iteration_count;
  free(start_indexes);
  free(faces_iteration_indexes);
  //quads_to_cuda->faces_iter_indexes = faces_iteration_indexes;
/*
  printf("faces per iteration: \n");
  for(int i = 0; i < 8; i++) {
    printf("%d: %d ", i, faces_per_iteration[i]);
  }
  printf("\n");
  printf("faces_iter_indexes:\n");
  for(size_t i = 0; i < faces_length; i++) {
    printf("%d ", faces_iteration_indexes[i]);
  }
  printf("\n");
  */
  //std::vector<std::vector<p4est_iter_face_side_t>* >::iterator iteration_faces_iter = iterations_to_faces.begin();

  //size_t face_size = sizeof(p4est_iter_face_side_t);
  //size_t faces_bytes_alloc = faces_count * 2 * face_size;
 // size_t faces_bytes_alloc = faces_count * face_size;
  //p4est_iter_face_side_t* sides_arr = (p4est_iter_face_side_t*)malloc(faces_bytes_alloc);
 // p4est_iter_face_side_t* sides_arr_cursor = sides_arr;

  //std::vector<size_t> faces_in_iter;
  /*
  for(size_t j = 0;;) {
    if(faces_iter == all_faces.end()) {
      if(j != 0) {
        faces_in_iter.push_back(j);
        j = 0;
      }
      size_t faces_size = all_faces.size();
      if(!all_faces.size()){
        break;
      }

      faces_iter = all_faces.begin();
      //std::vector<p4est_iter_face_side_t> *new_iteration = new std::vector<p4est_iter_face_side_t>();
      //iterations_to_faces.push_back(new_iteration);
      //iteration_faces_iter = iterations_to_faces.end();
      //iteration_faces_iter--;
      for(size_t i = 0; i < quads_length; i++){
        quad_is_used[i] = 0;
      }
      ghost_quad_is_used[-1] = 0;
      for(size_t i = 0; i < ghost_layer->ghosts.elem_count; i++) {
        ghost_quad_is_used[i] = 0;
      }
    }
    p4est_iter_face_side_t left = (*faces_iter++);
    //faces_iter++;
    p4est_iter_face_side_t right = (*faces_iter++);
    //faces_iter++;

    if(left.is_hanging) {
      p4est_locidx_t *left_quads = left.is.hanging.quadid;
      int8_t *left_is_ghost = left.is.hanging.is_ghost;
      if(right.is_hanging) {
        p4est_locidx_t *right_quads = right.is.hanging.quadid;
        int8_t *right_is_ghost = right.is.hanging.is_ghost;
        
        bool facesIsUsed = 
        (*left_is_ghost++ ? ghost_quad_is_used[*left_quads++] : quad_is_used[*left_quads++]) ||
        (*left_is_ghost ? ghost_quad_is_used[*left_quads] : quad_is_used[*left_quads]) ||
        (*right_is_ghost++ ? ghost_quad_is_used[*right_quads++] : quad_is_used[*right_quads++]) ||
        (*right_is_ghost ? ghost_quad_is_used[*right_quads] : quad_is_used[*right_quads]);

        if(!facesIsUsed) {
          (*left_is_ghost-- ? ghost_quad_is_used[*left_quads--] = 1 : quad_is_used[*left_quads--] = 1) &&
          (*left_is_ghost ? ghost_quad_is_used[*left_quads] = 1 : quad_is_used[*left_quads] = 1) &&
          (*right_is_ghost-- ? ghost_quad_is_used[*right_quads--] = 1 : quad_is_used[*right_quads--] = 1) &&
          (*right_is_ghost ? ghost_quad_is_used[*right_quads] = 1 : quad_is_used[*right_quads] = 1);
          //memcpy(sides_arr_cursor, left, face_size);
          //memcpy(++sides_arr_cursor, right, face_size);
          *sides_arr_cursor = left;
          *(++sides_arr_cursor) = right;
          ++sides_arr_cursor;
          //(*iteration_faces_iter)->push_back(left);
          //(*iteration_faces_iter)->push_back(right);
          faces_iter = all_faces.erase(--faces_iter);
          faces_iter = all_faces.erase(--faces_iter);
          j+=2;
        }
      } else {
        p4est_locidx_t right_quad = right.is.full.quadid;
        int8_t right_is_ghost = right.is.full.is_ghost;
        //int8_t facesIsUnused = right.is.full.is_ghost ? !ghost_quad_is_used[right_quad] : !quad_is_used[right_quad];
        bool facesIsUsed = 
        (*left_is_ghost++ ? ghost_quad_is_used[*left_quads++] : quad_is_used[*left_quads++]) ||
        (*left_is_ghost ? ghost_quad_is_used[*left_quads] : quad_is_used[*left_quads]) ||
        (right_is_ghost ? ghost_quad_is_used[right_quad] : quad_is_used[right_quad]);

        //if(facesIsUsed) {
          if(!facesIsUsed) {
            (*left_is_ghost-- ? ghost_quad_is_used[*left_quads--] = 1 : quad_is_used[*left_quads--] = 1) &&
            (*left_is_ghost ? ghost_quad_is_used[*left_quads] = 1 : quad_is_used[*left_quads] = 1) &&
            (right_is_ghost ? ghost_quad_is_used[right_quad] = 1 : quad_is_used[right_quad] = 1);
            //memcpy(sides_arr_cursor, left, face_size);
            *sides_arr_cursor = left;
            //(*iteration_faces_iter)->push_back(left);
            //memcpy(++sides_arr_cursor, right, face_size);
            *(++sides_arr_cursor) = right;
            ++sides_arr_cursor; 
            //(*iteration_faces_iter)->push_back(right);
            faces_iter = all_faces.erase(--faces_iter);
            faces_iter = all_faces.erase(--faces_iter);
            j+=2;
          }
        //}
      }
    } else {
      p4est_locidx_t left_quad = left.is.full.quadid;
      int8_t left_is_ghost = left.is.full.is_ghost;
      if(right.is_hanging) {
        p4est_locidx_t *right_quads = right.is.hanging.quadid;
        int8_t *right_is_ghost = right.is.hanging.is_ghost;
        //int8_t facesIsUnused = left.is.full.is_ghost ? !ghost_quad_is_used[left_quad] : !quad_is_used[left_quad];
        int8_t facesIsUsed = 
        (left_is_ghost ? ghost_quad_is_used[left_quad] : quad_is_used[left_quad]) ||
        (*right_is_ghost++ ? ghost_quad_is_used[*right_quads++] : quad_is_used[*right_quads++]) ||
        (*right_is_ghost ? ghost_quad_is_used[*right_quads] : quad_is_used[*right_quads]);

        //if(facesIsUnused) {
          if(!facesIsUsed) {
            (left_is_ghost ? ghost_quad_is_used[left_quad] = 1 : quad_is_used[left_quad] = 1) &&
            (*right_is_ghost-- ? ghost_quad_is_used[*right_quads--] = 1 : quad_is_used[*right_quads--] = 1) &&
            (*right_is_ghost ? ghost_quad_is_used[*right_quads] = 1 : quad_is_used[*right_quads] = 1);
            //memcpy(sides_arr_cursor, left, face_size);
            *sides_arr_cursor = left;
            //(*iteration_faces_iter)->push_back(left);
            //memcpy(++sides_arr_cursor, right, face_size);
            *(++sides_arr_cursor) = right;
            ++sides_arr_cursor;
            //(*iteration_faces_iter)->push_back(right);
            faces_iter = all_faces.erase(--faces_iter);
            faces_iter = all_faces.erase(--faces_iter);
            j+=2;
          }
        //}
      } else {
        p4est_locidx_t right_quad = right.is.full.quadid;
        int8_t right_is_ghost = right.is.full.is_ghost;
        //int8_t facesIsUnused = (right.is.full.is_ghost ? !ghost_quad_is_used[right_quad] : !quad_is_used[right_quad]) &&
        //(left.is.full.is_ghost ? !ghost_quad_is_used[left_quad] : !quad_is_used[left_quad]);

        bool facesIsUsed = 
          (left_is_ghost ? ghost_quad_is_used[left_quad] : quad_is_used[left_quad]) ||
          (right_is_ghost ? ghost_quad_is_used[right_quad] : quad_is_used[right_quad]);
        if(!facesIsUsed) {
            (left_is_ghost ? ghost_quad_is_used[left_quad] = 1 : quad_is_used[left_quad] = 1) &&
            (right_is_ghost ? ghost_quad_is_used[right_quad] = 1 : quad_is_used[right_quad] = 1);
          *sides_arr_cursor = left;
          *(++sides_arr_cursor) = right;
          //memcpy(sides_arr_cursor, left, face_size);
          //memcpy(++sides_arr_cursor, right, face_size);
          ++sides_arr_cursor;
          //(*iteration_faces_iter)->push_back(left);
          //(*iteration_faces_iter)->push_back(right);
          faces_iter = all_faces.erase(--faces_iter);
          faces_iter = all_faces.erase(--faces_iter);
          j+=2;
        }
      }
    } 
  }
*/

  free(quad_is_used);
  free(--ghost_quad_is_used);
  //quads_to_cuda->faces_iteration_count = iterations_to_faces.size();
  //quads_to_cuda->faces_iteration_count = faces_in_iter.size();
  //size_t *faces_per_iter = (size_t*)malloc(sizeof(size_t) * quads_to_cuda->faces_iteration_count);
  //size_t *faces_per_iter_cursor = faces_per_iter;
  //size_t face_size = sizeof(p4est_iter_face_side_t);
  //size_t faces_bytes_alloc = faces_count * face_size;
  //p4est_iter_face_side_t* sides_arr = (p4est_iter_face_side_t*)malloc(faces_bytes_alloc);
  //p4est_iter_face_side_t* cursor = sides_arr;
  //for(std::vector<std::vector<p4est_iter_face_side_t>* >::iterator iteration_i = iterations_to_faces.begin(); iteration_i != iterations_to_faces.end(); iteration_i++, faces_per_iter_cursor++) {
  //  *faces_per_iter_cursor = (*iteration_i)->size();
  //  memcpy(cursor, (*iteration_i)->data(), (*faces_per_iter_cursor)*face_size);
  //  cursor+=*faces_per_iter_cursor;
    //for(std::vector<p4est_iter_face_side_t>::iterator face_i = (*iteration_i)->begin(); face_i != (*iteration_i)->end(); face_i++, cursor++) {
    //  memcpy(cursor, &(*face_i), face_size);
    //}
  //  delete *iteration_i;
  //}
  //memcpy(faces_per_iter, faces_in_iter.data(), sizeof(size_t) * faces_in_iter.size());
  //quads_to_cuda->faces_per_iter = faces_per_iter;

  //quads_to_cuda->h_sides = sides_arr;
  //p4est_iter_face_side_t* d_faces;
  //gpuErrchk(cudaMalloc((void**)&d_faces, faces_bytes_alloc));
  //gpuErrchk(cudaMemcpy(d_faces, sides_arr, faces_bytes_alloc, cudaMemcpyHostToDevice));
  //quads_to_cuda->d_sides = d_faces;
  cuda_iter_loop_args_destroy (loop_args);
}

void freeMemoryForFacesSides(p4est_quadrants_to_cuda* quads_to_cuda) {
  gpuErrchk(cudaFree(quads_to_cuda->d_sides));
  free(quads_to_cuda->h_sides);
}

void mallocQuadrantsBlocks(cuda4est_t* cuda4est, sc_array_t* quadrants, p4est_quadrants_to_cuda* quads_to_cuda, p4est_ghost_t* Ghost_layer, p4est_ghost_to_cuda_t* ghost_to_cuda) {
  clock_t start = clock();
  clock_t stop = clock();
  double duration = (double)(stop - start) / CLOCKS_PER_SEC;

  std::vector<p4est_iter_face_side_t> all_faces;
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
  start = clock();
  for (t = first_local_tree; t <= last_run_tree; t++) {
    if (t >= first_local_tree && t <= last_local_tree) {
      cuda_iter_init_volume (&args, p4est, ghost_layer, loop_args, t);

      cuda_volume_iterate_without_callbacks (&args, &all_faces);

      cuda_iter_reset_volume (&args);
    }
    touch = owned[t];
    if (!touch) {
      continue;
    }
    mask = 0x00000001;
    /* Now we need to run face_iterate on the faces between trees */
    for (f = 0; f < 2 * P4EST_DIM; f++, mask <<= 1) {
      if ((touch & mask) == 0) {
        continue;
      }
      cuda_iter_init_face (&face_args, p4est, ghost_layer, loop_args, t, f);
      p4est_face_iterate_without_callbacks(&face_args, &all_faces);
      cuda_iter_reset_face (&face_args);
    }
  }
  stop = clock();
  duration = (double)(stop - start) / CLOCKS_PER_SEC;

  if (Ghost_layer == NULL) {
    P4EST_FREE (empty_ghost_layer.tree_offsets);
    P4EST_FREE (empty_ghost_layer.proc_offsets);
  }
  P4EST_FREE (owned);

  size_t quads_length = quads_to_cuda->quadrants_length;
  size_t sides_count = all_faces.size();
  quads_to_cuda->sides_count = sides_count;
  p4est_iter_face_side_t* sides_array = all_faces.data();
  size_t faces_length = sides_count/2;

  size_t block_max_quads_count = 128;
  const unsigned char sides_max_count = 8;
  const unsigned char faces_max_count = sides_max_count / 2;
  const unsigned char mask_bound[faces_max_count] = {1, 2, 4, 8};

  std::vector<size_t> quad_indexes;
  std::vector<unsigned char> block_lengths;
  std::vector<unsigned char> output_quad_length_per_block;

  size_t ** quad_faces = new size_t*[faces_max_count];
  for(unsigned char i = 0; i < faces_max_count; i++) {
    quad_faces[i] = new size_t[quads_length];
  }

  p4est_iter_face_side_t *sides_cursor = sides_array;
  for(size_t i = 0; i < faces_length; i++) {
    p4est_iter_face_side_t *first_side = sides_cursor++;
    if(first_side->is_hanging) {
      size_t first_quad = first_side->is.hanging.quadid[0];
      if(first_quad != -1) {
        quad_faces[first_side->face][first_quad] = i;
      }
      size_t second_quad = first_side->is.hanging.quadid[1];
      if(second_quad != -1) {
        quad_faces[first_side->face][second_quad] = i;
      }
    } else {
      size_t quad = first_side->is.full.quadid;
      if(quad != -1) {
        quad_faces[first_side->face][quad] = i;
      }
    }

    p4est_iter_face_side_t *second_side = sides_cursor++;
    if(second_side->is_hanging) {
      size_t first_quad = second_side->is.hanging.quadid[0];
      if(first_quad != -1) {
        quad_faces[second_side->face][first_quad] = i;
      }
      size_t second_quad = second_side->is.hanging.quadid[1];
      if(second_quad != -1) {
        quad_faces[second_side->face][second_quad] = i;
      }
    } else {
      size_t quad = second_side->is.full.quadid;
      if(quad != -1) {
        quad_faces[second_side->face][quad] = i;
      }
    }
  }


  unsigned char *quad_is_used_in_block = new unsigned char[quads_length];
  for(size_t i = 0; i < quads_length; i++) {
    quad_is_used_in_block[i] = 0;
  }

  unsigned char max_quads_on_sides = 8;
  size_t *quads_buffer = new size_t[max_quads_on_sides];
  size_t max_faces_per_quads_sides = max_quads_on_sides * faces_max_count;
  size_t *faces_buffer = new size_t[max_faces_per_quads_sides];
  unsigned char k = 0;
  size_t faces_block_count = 0;

  std::vector<size_t> faces_block_indexes;
  std::vector<size_t> faces_block_lengths;

  unsigned char *face_is_used_in_block = new unsigned char[faces_length];
  for(size_t i = 0; i < faces_length; i++) {
    face_is_used_in_block[i] = 0;
  }

  size_t prev_output_quads_block_count = 0;
  unsigned char quads_buffer_count = 0;
  size_t current_quad_index = 0;
  for(; current_quad_index < quads_length;) {
    size_t faces_buffer_count = 0;
    quads_buffer_count = 0;
    for(unsigned char j = 0; j < faces_max_count; j++) {
      size_t face_id = quad_faces[j][current_quad_index];
      if(!face_is_used_in_block[face_id]) {
        p4est_iter_face_side_t* first_side = sides_array + 2 * face_id;
        if(first_side->is_hanging) {
          size_t first_quad = first_side->is.hanging.quadid[0];
          size_t second_quad = first_side->is.hanging.quadid[1];
          if(first_quad != -1) {
            if(!quad_is_used_in_block[first_quad]) {
              quads_buffer[quads_buffer_count++] = first_quad;
              quad_is_used_in_block[first_quad] = 1;
            }
          }
          if(second_quad != -1) {
            if(!quad_is_used_in_block[second_quad]) {
              quads_buffer[quads_buffer_count++] = second_quad;
              quad_is_used_in_block[second_quad] = 1;
            }
          }
          
        } else {
          size_t quad = first_side->is.full.quadid;
          if(quad != -1) {
            if(!quad_is_used_in_block[quad]){
              quads_buffer[quads_buffer_count++] = quad;
              quad_is_used_in_block[quad] = 1;
            }
          }
        }

        p4est_iter_face_side_t* second_side = first_side + 1;
        if(second_side->is_hanging) {
          size_t first_quad = second_side->is.hanging.quadid[0];
          size_t second_quad = second_side->is.hanging.quadid[1];
          if(first_quad != -1) {
            if(!quad_is_used_in_block[first_quad]) {
              quads_buffer[quads_buffer_count++] = first_quad;
              quad_is_used_in_block[first_quad] = 1;
            }
          }
          if(second_quad != -1) {
            if(!quad_is_used_in_block[second_quad]) {
              quads_buffer[quads_buffer_count++] = second_quad;
              quad_is_used_in_block[second_quad] = 1;
            }
          }
        } else {
          size_t quad = second_side->is.full.quadid;
          if(quad != -1) {
            if(!quad_is_used_in_block[quad]){
              quads_buffer[quads_buffer_count++] = quad;
              quad_is_used_in_block[quad] = 1;
            }
          }
        }
        face_is_used_in_block[face_id] = 1;
        faces_buffer[faces_buffer_count++] = face_id;
      }
    }
    if((block_max_quads_count - k) >= quads_buffer_count) {
      for(unsigned char i = 0; i < quads_buffer_count; i++) {
        quad_indexes.push_back(quads_buffer[i]);
      }
      for(unsigned char i = 0; i < faces_buffer_count; i++) {
        faces_block_indexes.push_back(faces_buffer[i]);
      }
      k+=quads_buffer_count;
      faces_block_count+=faces_buffer_count;
      current_quad_index++;
    } else {
      block_lengths.push_back(k);
      output_quad_length_per_block.push_back(current_quad_index - prev_output_quads_block_count);
      faces_block_lengths.push_back(faces_block_count);
      k = 0;
      faces_block_count = 0;
      prev_output_quads_block_count = current_quad_index;
      
      for(size_t l = 0; l < quads_length; l++) {
        quad_is_used_in_block[l] = 0;
      }
      for(size_t i = 0; i < faces_length; i++) {
        face_is_used_in_block[i] = 0;
      }
    }
  }
  if(k != 0) {
    block_lengths.push_back(k);
    output_quad_length_per_block.push_back(current_quad_index - prev_output_quads_block_count);
    faces_block_lengths.push_back(faces_block_count);
  }

  delete face_is_used_in_block;


  size_t block_count = block_lengths.size();

  size_t result_quads_length = 0;
  size_t result_faces_length = 0;
  size_t result_output_quads_length = 0;
  std::vector<size_t>::iterator faces_block_it = faces_block_lengths.begin();
  std::vector<unsigned char>::iterator quadrants_block_it = block_lengths.begin();
  std::vector<unsigned char>::iterator output_quadrants_block_it = output_quad_length_per_block.begin();

  size_t *block_lengths_config = (size_t*)malloc(sizeof(size_t) * block_count * 7);
  size_t *block_lengths_config_cursor = block_lengths_config;
  size_t user_data_size = cuda4est->p4est->data_size;

  size_t max_shared_memory_size = 0;

  size_t test_block_index = 0;
  for(;quadrants_block_it != block_lengths.end(); quadrants_block_it++, faces_block_it++, output_quadrants_block_it++, block_lengths_config_cursor+=7, test_block_index++) {

    size_t quads_start_index = result_quads_length;
    size_t quads_count = *quadrants_block_it;

    size_t start_byte_index = result_quads_length * user_data_size;;
    size_t quads_bytes_count = *quadrants_block_it * user_data_size;
    
    size_t output_quads_count = *output_quadrants_block_it;

    size_t faces_start_index = result_faces_length;
    size_t faces_count = *faces_block_it;

    block_lengths_config_cursor[0] = result_quads_length;
    block_lengths_config_cursor[1] = *quadrants_block_it;
    
    block_lengths_config_cursor[2] = result_quads_length * user_data_size;
    block_lengths_config_cursor[3] = *quadrants_block_it * user_data_size;

    block_lengths_config_cursor[4] = *output_quadrants_block_it;

    block_lengths_config_cursor[5] = result_faces_length;
    block_lengths_config_cursor[6] = *faces_block_it;

    size_t shared_memory_size = *quadrants_block_it * user_data_size + *quadrants_block_it + sizeof(cuda_light_face_side_t) * 2 * *faces_block_it;
    if(max_shared_memory_size < shared_memory_size) {
      max_shared_memory_size = shared_memory_size;
    }

    result_quads_length += *quadrants_block_it;
    result_faces_length += *faces_block_it;
    result_output_quads_length += *output_quadrants_block_it;
  }

  quads_to_cuda->shared_memory_size = max_shared_memory_size;
  printf("max_memory_size: %lu\n", max_shared_memory_size);

  unsigned char *quad_local_indexes = quad_is_used_in_block;
  for(size_t i = 0; i < quads_length; i++) {
    quad_local_indexes[i] = -1;
  }

  unsigned char empty_local_index = -1;

  size_t *quad_indexes_arr = quad_indexes.data();
  unsigned char *block_lengths_arr = block_lengths.data();

  unsigned char *output_quad_length_per_block_arr = output_quad_length_per_block.data();

  size_t *faces_block_indexes_arr = faces_block_indexes.data();
  size_t *faces_block_lengths_arr = faces_block_lengths.data();

  size_t start_output_index = 0;
  size_t end_output_index = 0;

  cuda_light_face_side_t *light_sides_for_cuda = (cuda_light_face_side_t*)malloc(sizeof(cuda_light_face_side_t) * 2 * result_faces_length);
  size_t start_face_index = 0;
  size_t end_face_index = 0;

  
  void *user_data_for_blocks = (void*)malloc(user_data_size * result_quads_length);
  void *user_data_cursor = user_data_for_blocks;
  
  current_quad_index = 0;
  size_t current_not_valid_quad_index = 0;
  size_t *not_valid_block_quad_global_index = (size_t*)malloc(sizeof(size_t) * (result_quads_length - quads_length));
  size_t *valid_quad_index_in_result_quads = (size_t*)malloc(sizeof(size_t) * quads_length);

  unsigned char *quads_levels = new unsigned char[result_quads_length];

  for(size_t block_index = 0; block_index < block_count; block_index++) {
    start_output_index = end_output_index;
    end_output_index += output_quad_length_per_block_arr[block_index];

    start_face_index = end_face_index;
    end_face_index += faces_block_lengths_arr[block_index];

    size_t not_output_quad_index = output_quad_length_per_block_arr[block_index];
    for(size_t quad_index = start_output_index; quad_index < end_output_index; quad_index++, user_data_cursor = user_data_cursor + user_data_size) {
      p4est_quadrant_t *quadrant = p4est_quadrant_array_index(quadrants, quad_index);
      memcpy(user_data_cursor, p4est_quadrant_array_index(quadrants, quad_index)->p.user_data, user_data_size);
      quads_levels[current_quad_index] = quadrant->level;
      valid_quad_index_in_result_quads[quad_index]=current_quad_index++;
    }
    for(size_t face_index = start_face_index; face_index < end_face_index; face_index++) {
      size_t face_id = faces_block_indexes_arr[face_index];
      size_t side_id = face_id * 2;

      size_t base_side_index = face_index * 2;
      
      for(unsigned char side_index = 0; side_index < 2; side_index++, side_id++) {
        p4est_iter_face_side_t *side = sides_array + side_id;
        cuda_light_face_side_t &light_side = light_sides_for_cuda[base_side_index + side_index];
        light_side.face = side->face;
        if(side->is_hanging) {
          light_side.is_hanging = 1;
          size_t first_quad = side->is.hanging.quadid[0];
          size_t second_quad = side->is.hanging.quadid[1];
          if(first_quad != -1) {
            size_t first_local_quad_index = quad_local_indexes[first_quad];
            if(first_local_quad_index != empty_local_index) {
            } else if(first_quad >= start_output_index && first_quad < end_output_index) {
              first_local_quad_index = quad_local_indexes[first_quad] = first_quad - start_output_index;
            } else {
              first_local_quad_index = quad_local_indexes[first_quad] = not_output_quad_index++;
              p4est_quadrant_t *quadrant = p4est_quadrant_array_index(quadrants, first_quad);
              memcpy(user_data_cursor, p4est_quadrant_array_index(quadrants, first_quad)->p.user_data, user_data_size);
              quads_levels[current_quad_index] = quadrant->level;
              user_data_cursor = user_data_cursor + user_data_size;
              not_valid_block_quad_global_index[current_not_valid_quad_index++] = first_quad;
              current_quad_index++;
            }
            light_side.is.hanging.quadid[0] = first_local_quad_index;
          }
          if(second_quad != -1) {
            size_t second_local_quad_index = quad_local_indexes[second_quad];
            if(second_local_quad_index != empty_local_index) {
            } else if(second_quad >= start_output_index && second_quad < end_output_index) {
              second_local_quad_index = quad_local_indexes[second_quad] = second_quad - start_output_index;
            } else {
              second_local_quad_index = quad_local_indexes[second_quad] = not_output_quad_index++;
              p4est_quadrant_t *quadrant = p4est_quadrant_array_index(quadrants, second_quad);
              memcpy(user_data_cursor, p4est_quadrant_array_index(quadrants, second_quad)->p.user_data, user_data_size);
              quads_levels[current_quad_index] = quadrant->level;
              user_data_cursor = user_data_cursor + user_data_size;
              not_valid_block_quad_global_index[current_not_valid_quad_index++] = second_quad;
              current_quad_index++;
            }
            light_side.is.hanging.quadid[1] = second_local_quad_index;
          }
        } else {
          light_side.is_hanging = 0;
          size_t quad = side->is.full.quadid;
          if(quad != -1){
            size_t quad_local_index = quad_local_indexes[quad];
            if(quad_local_index != empty_local_index) {
            } else if(quad >= start_output_index && quad < end_output_index) {
              quad_local_index = quad_local_indexes[quad] = quad - start_output_index;
            } else {
              quad_local_index = quad_local_indexes[quad] = not_output_quad_index++;
              p4est_quadrant_t *quadrant = p4est_quadrant_array_index(quadrants, quad);
              memcpy(user_data_cursor, p4est_quadrant_array_index(quadrants, quad)->p.user_data, user_data_size);
              quads_levels[current_quad_index] = quadrant->level;
              user_data_cursor = user_data_cursor + user_data_size;
              not_valid_block_quad_global_index[current_not_valid_quad_index++] = quad;
              current_quad_index++;
            }
            light_side.is.full.quadid = quad_local_index;
          }
        }
      }
    }
    for(size_t i = 0; i < quads_length; i++) {
      quad_local_indexes[i] = -1;
    }
  }

  for(unsigned char i = 0; i < faces_max_count; i++) {
    delete quad_faces[i];
  }
  delete quad_faces;

  delete quad_is_used_in_block;
  

  quads_to_cuda->block_count = block_count;
  quads_to_cuda->config_blocks = block_lengths_config;

  size_t *d_config_blocks;
  gpuErrchk(cudaMalloc((void**)&d_config_blocks, sizeof(size_t) * 7 * block_count));
  gpuErrchk(cudaMemcpy(d_config_blocks, block_lengths_config, sizeof(size_t) * 7 * block_count, cudaMemcpyHostToDevice));
  quads_to_cuda->d_config_blocks = d_config_blocks;

  quads_to_cuda->not_valid_block_quad_global_index = not_valid_block_quad_global_index;
  quads_to_cuda->valid_quad_index_in_result_quads = valid_quad_index_in_result_quads;

  cuda_light_face_side_t *d_light_sides;
  gpuErrchk(cudaMalloc((void**)&d_light_sides, sizeof(cuda_light_face_side_t) * result_faces_length * 2));
  gpuErrchk(cudaMemcpy(d_light_sides, light_sides_for_cuda, sizeof(cuda_light_face_side_t) * result_faces_length * 2, cudaMemcpyHostToDevice));
  quads_to_cuda->d_light_sides = d_light_sides;

  free(light_sides_for_cuda);


  void* d_blocks_user_data;
  gpuErrchk(cudaMalloc((void**)&d_blocks_user_data, user_data_size * result_quads_length));
  gpuErrchk(cudaMemcpy(d_blocks_user_data, user_data_for_blocks, user_data_size * result_quads_length, cudaMemcpyHostToDevice));
  quads_to_cuda->d_blocks_user_data = d_blocks_user_data;

  unsigned char* d_quads_levels;
  gpuErrchk(cudaMalloc((void**)&d_quads_levels, result_quads_length));
  gpuErrchk(cudaMemcpy(d_quads_levels, quads_levels, result_quads_length, cudaMemcpyHostToDevice));
  quads_to_cuda->d_quads_levels = d_quads_levels;

  free(quads_levels);

  quads_to_cuda->result_quads_length = result_quads_length;

  free(user_data_for_blocks);

  cuda_iter_loop_args_destroy (loop_args);
}

void freeMemoryForQuadrantsBlocks(p4est_quadrants_to_cuda* quads_to_cuda) {
  free(quads_to_cuda->config_blocks);
  free(quads_to_cuda->not_valid_block_quad_global_index);
  free(quads_to_cuda->valid_quad_index_in_result_quads);

  gpuErrchk(cudaFree(quads_to_cuda->d_config_blocks));
  gpuErrchk(cudaFree(quads_to_cuda->d_blocks_user_data));
  gpuErrchk(cudaFree(quads_to_cuda->d_quads_levels));
  gpuErrchk(cudaFree(quads_to_cuda->d_light_sides));
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

    p4est_locidx_t *d_tree_offsets_temp, *d_proc_offsets_temp;
    size_t d_tree_offsets_temp_size = (num_trees + 1) * sizeof(p4est_locidx_t);
    size_t d_proc_offsets_temp_size = (mpisize + 1) * sizeof(p4est_locidx_t);

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

void exchangeGhostDataToCuda(p4est_ghost_to_cuda_t* ghost_to_cuda, p4est_ghost_t* ghost_layer) {
  p4est_ghost_t *d_ghost_layer = ghost_to_cuda->d_ghost_layer;
  char *d_ghosts_array_temp;
  size_t d_ghosts_array_temp_size = ghost_layer->ghosts.elem_size * ghost_layer->ghosts.elem_count;

  gpuErrchk(cudaMalloc((void**)&d_ghosts_array_temp, d_ghosts_array_temp_size));
  gpuErrchk(cudaMemcpy(&(d_ghost_layer->ghosts.array), &d_ghosts_array_temp, sizeof(char*), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_ghosts_array_temp, ghost_layer->ghosts.array, d_ghosts_array_temp_size, cudaMemcpyHostToDevice));
  ghost_to_cuda->d_ghosts_array_temp = d_ghosts_array_temp;
  ghost_to_cuda->d_ghosts_array_temp_size = d_ghosts_array_temp_size;
}

void freeGhostDataFromCuda(p4est_ghost_to_cuda_t* ghost_to_cuda) {
  if(ghost_to_cuda->d_ghosts_array_temp) {
    gpuErrchk(cudaFree(ghost_to_cuda->d_ghosts_array_temp));
  }
}

void freeMemoryForGhost(p4est_ghost_to_cuda_t* ghost_to_cuda) {
  freeGhostDataFromCuda(ghost_to_cuda);
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