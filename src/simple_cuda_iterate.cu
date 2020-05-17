#include "simple_cuda_iterate.h"
#include "stdio.h"




__global__ void
simple_quadrants_iterate(
  sc_array_t* quadrants, p4est_ghost_t* ghost_layer,
  p4est_t* p4est, p4est_topidx_t treeId, 
  void* user_data, cuda_iter_volume_t iter_volume,
  size_t quads_count, size_t quads_per_thread)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < quads_count) {
    size_t elems_remaining;
    if(i >= quads_count - quads_per_thread) {
      elems_remaining = quads_count - i;
    } else {
      elems_remaining = quads_per_thread;
    }
    for(int j = 0; j < elems_remaining; j++) {
      iter_volume(
        p4est,
        ghost_layer,
        p4est_device_quadrant_array_index(quadrants, i + j),
        i+j,
        treeId,
        user_data
      );
    }
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

    // ghost_layer memory allocation start
    p4est_ghost_to_cuda_t *ghost_to_cuda = mallocForGhost(p4est, ghost_layer);
    // ghost_layer memory allocation end

    info.p4est = p4est;
    info.ghost_layer = ghost_layer;

    for (t = first_local_tree; t <= last_local_tree; t++) {
        info.treeid = t;
        n_quads = cuda4est->quads_to_cuda->quadrants_length;

        // quadrants memory allocation start
        // p4est_quadrants_to_cuda_t *quads_to_cuda = mallocForQuadrants(quadrants, cuda4est->quad_user_data_api);
        // quadrants memory allocation end
        user_data_volume_cuda_api->alloc_cuda_memory(user_data_volume_cuda_api);
        void *d_user_data = user_data_volume_cuda_api->get_cuda_allocated_user_data(user_data_volume_cuda_api);

        // constants
        size_t max_block_count = 1024;
        size_t max_treads_per_block = 1024;
        size_t max_block_count_per_process = max_block_count / p4est->mpisize;
        size_t threads_per_block = 128;
        // constants

        // calculate init cuda dimensions
        size_t min_quadrants_per_thread = (double)(n_quads / threads_per_block) / (max_block_count_per_process - 1)  + 1;
        size_t needed_block_count = n_quads / (min_quadrants_per_thread * threads_per_block) + 1;
        // calculate init cuda dimensions
        
        unsigned long long *d_callback, h_callback;
        cudaMalloc(&d_callback, sizeof(unsigned long long));
        iter_volume_api->setup_kernel<<<1,1>>>((cuda_iter_volume_t*)d_callback);
        cudaMemcpy(&h_callback, d_callback, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        simple_quadrants_iterate<<<needed_block_count, threads_per_block>>>(
          cuda4est->quads_to_cuda->d_quadrants, ghost_to_cuda->d_ghost_layer,
          cuda4est->p4est_memory_allocate_info->d_p4est, t,
          d_user_data, (cuda_iter_volume_t) h_callback,
          n_quads, min_quadrants_per_thread
        );
        cudaDeviceSynchronize();
        user_data_volume_cuda_api->copy_user_data_from_device(user_data_volume_cuda_api);
        user_data_volume_cuda_api->free_cuda_memory(user_data_volume_cuda_api);
    }
    freeMemoryForGhost(ghost_to_cuda);
}