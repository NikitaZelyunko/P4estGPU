#include "simple_cuda_iterate.h"
#include "stdio.h"

/** Return a pointer to an array element indexed by a p4est_topidx_t.
 * \param [in] index needs to be in [0]..[elem_count-1].
 */
/*@unused@*/
__device__
static inline p4est_tree_t *
p4est_device_tree_array_index (sc_array_t * array, p4est_topidx_t it)
{
  P4EST_ASSERT (array->elem_size == sizeof (p4est_tree_t));
  P4EST_ASSERT (it >= 0 && (size_t) it < array->elem_count);

  return (p4est_tree_t *) (array->array +
                           sizeof (p4est_tree_t) * (size_t) it);
}

/** Return a pointer to a quadrant array element indexed by a size_t. */
/*@unused@*/
__device__
static inline p4est_quadrant_t *
p4est_device_quadrant_array_index (sc_array_t * array, size_t it)
{
  P4EST_ASSERT (array->elem_size == sizeof (p4est_quadrant_t));
  P4EST_ASSERT (it < array->elem_count);

  return (p4est_quadrant_t *) (array->array + sizeof (p4est_quadrant_t) * it);
}


typedef struct step3_data
{
  double              u;             /**< the state variable */
  double              du[P4EST_DIM]; /**< the spatial derivatives */
  double              dudt;          /**< the time derivative */
}
step3_data_t;

__global__ void
simple_quadrants_iterate(sc_array_t* quadrants, p4est_ghost_t* ghost_layer, p4est_t* p4est, p4est_topidx_t treeId, void* user_data, cuda_iter_volume_t iter_volume)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t quads_length = quadrants->elem_count;

  size_t elems_remaining = quads_length - i;
  size_t thread_elem_count = elems_remaining > 4 ? 4 : elems_remaining;
  for(int j = 0; j < thread_elem_count; j++) {
    iter_volume(
      p4est,
      ghost_layer,
      p4est_device_quadrant_array_index(quadrants, i + j),
      i,
      treeId,
      user_data
    );
  }
}


void simple_volume_cuda_iterate(
    cuda4est_t * cuda4est, 
    p4est_ghost_t * ghost_layer,
    user_data_for_cuda_t *user_data_volume_cuda_api,
    cuda_iter_volume_api_t* iter_volume_api
) {

    p4est_t *p4est = cuda4est->p4est;
    p4est_topidx_t      t;
    p4est_topidx_t      first_local_tree = p4est->first_local_tree;
    p4est_topidx_t      last_local_tree = p4est->last_local_tree;
    sc_array_t         *trees = p4est->trees;
    p4est_tree_t       *tree;
    size_t              si, n_quads;
    sc_array_t         *quadrants;
    cuda_iter_volume_info_t info;

    // p4est memory allocation start
    p4est_cuda_memory_allocate_info_t *p4est_memory_allocate_info = p4est_memory_alloc(cuda4est);
    // p4est memory allocation end

    // ghost_layer memory allocation start
    p4est_ghost_to_cuda_t *ghost_to_cuda = mallocForGhost(p4est, ghost_layer);
    // ghost_layer memory allocation end

    info.p4est = p4est;
    info.ghost_layer = ghost_layer;

    for (t = first_local_tree; t <= last_local_tree; t++) {
        info.treeid = t;
        tree = p4est_tree_array_index (trees, t);
        quadrants = &(tree->quadrants);
        n_quads = quadrants->elem_count;

        // quadrants memory allocation start
        p4est_quadrants_to_cuda_t *quads_to_cuda = mallocForQuadrants(quadrants, cuda4est->quad_user_data_api);
        // quadrants memory allocation end
        user_data_volume_cuda_api->alloc_cuda_memory(user_data_volume_cuda_api);
        void *d_user_data = user_data_volume_cuda_api->get_cuda_allocated_user_data(user_data_volume_cuda_api);

        // constants
        size_t max_block_count = 1024;
        size_t max_treads_per_block = 1024;
        size_t max_block_count_per_process = max_block_count / p4est->mpisize;
        // constants

        // calculate init cuda dimensions
        size_t threads_per_block = 128;

        // calculate init cuda dimensions

        int threads_length = n_quads >= 1024 ? 1024 : n_quads;
        int blocks = n_quads / 64 + 1;
        
        unsigned long long *d_callback, h_callback;
        cudaMalloc(&d_callback, sizeof(unsigned long long));
        iter_volume_api->setup_kernel<<<1,1>>>((cuda_iter_volume_t*)d_callback);
        cudaMemcpy(&h_callback, d_callback, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        simple_quadrants_iterate<<<1, threads_length>>>(quads_to_cuda->d_quadrants, ghost_to_cuda->d_ghost_layer, p4est_memory_allocate_info->d_p4est, t, d_user_data, (cuda_iter_volume_t) h_callback);
        cudaDeviceSynchronize();
        user_data_volume_cuda_api->copy_user_data_from_device(user_data_volume_cuda_api);
        user_data_volume_cuda_api->free_cuda_memory(user_data_volume_cuda_api);
        freeMemoryForQuadrants(quads_to_cuda, cuda4est->quad_user_data_api);
        free(quads_to_cuda);
    }
    freeMemoryForGhost(ghost_to_cuda);
    p4est_memory_free(p4est_memory_allocate_info, cuda4est->quad_user_data_api);
}