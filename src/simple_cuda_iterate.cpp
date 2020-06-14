#include "simple_cuda_iterate.h"


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


    info.p4est = p4est;
    info.ghost_layer = ghost_layer;

    for (t = first_local_tree; t <= last_local_tree; t++) {
        info.treeid = t;
        n_quads = cuda4est->quads_to_cuda->quadrants_length;

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
        gpuErrchk(cudaMalloc((void**)&d_callback, sizeof(unsigned long long)));
        run_setup_kernel_volume_callback(iter_volume_api, (cuda_iter_volume_t*)d_callback);
        gpuErrchk(cudaMemcpy(&h_callback, d_callback, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        run_simple_quadrants_iterate(
          cuda4est->quads_to_cuda->d_quadrants_array_temp, (void*)cuda4est->quads_to_cuda->all_quads_user_data_allocate_info->d_all_quads_user_data,
          cuda4est->p4est->data_size, 
          cuda4est->ghost_to_cuda->d_ghost_layer,
          cuda4est->p4est_memory_allocate_info->d_p4est, t,
          d_user_data, (cuda_iter_volume_t) h_callback,
          n_quads, min_quadrants_per_thread,
          needed_block_count, threads_per_block
        );
        gpuErrchk(cudaDeviceSynchronize());
        user_data_volume_cuda_api->copy_user_data_from_device(user_data_volume_cuda_api);
        user_data_volume_cuda_api->free_cuda_memory(user_data_volume_cuda_api);
    }
}

void simple_face_cuda_iterate(
    cuda4est_t * cuda4est, p4est_ghost_t * ghost_layer,
    user_data_for_cuda_t *user_data_volume_cuda_api,
    cuda_iter_face_api_t* iter_face_api
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


    info.p4est = p4est;
    info.ghost_layer = ghost_layer;

    info.treeid = t;
    size_t faces_count = cuda4est->quads_to_cuda->sides_count;
    size_t faces_per_iter = P4EST_CHILDREN / 2;
    size_t iter_count = faces_count / faces_per_iter;

    user_data_volume_cuda_api->alloc_cuda_memory(user_data_volume_cuda_api);
    void *d_user_data = user_data_volume_cuda_api->get_cuda_allocated_user_data(user_data_volume_cuda_api);

    // constants
    size_t max_block_count = 1024;
    size_t max_treads_per_block = 1024;
    size_t max_block_count_per_process = max_block_count / p4est->mpisize;
    size_t threads_per_block = 128;
    // constants

    
    unsigned long long *d_callback, h_callback;
    gpuErrchk(cudaMalloc((void**)&d_callback, sizeof(unsigned long long)));
    run_setup_kernel_face_callback(iter_face_api, (cuda_iter_face_t*)d_callback);
    gpuErrchk(cudaMemcpy(&h_callback, d_callback, sizeof(unsigned long long), cudaMemcpyDeviceToHost));


    size_t iteration_count = cuda4est->quads_to_cuda->faces_iteration_count;
    size_t *faces_count_per_iter = cuda4est->quads_to_cuda->faces_per_iter;


    for(size_t i = 0, start_index = 0; i < iteration_count; start_index+=faces_count_per_iter[i], i++) {
      size_t iter_faces_count = faces_count_per_iter[i] / 2;
      // calculate init cuda dimensions
      size_t min_faces_per_thread = (double)(iter_faces_count / threads_per_block) / (max_block_count_per_process - 1)  + 1;
      size_t needed_block_count = iter_faces_count / (min_faces_per_thread * threads_per_block) + 1;
      // calculate init cuda dimensions
      run_simple_faces_iterate(
        cuda4est->p4est_memory_allocate_info->d_p4est, cuda4est->ghost_to_cuda->d_ghost_layer,
        cuda4est->quads_to_cuda->d_quadrants,
        cuda4est->quads_to_cuda->d_sides + start_index, iter_faces_count,
        d_user_data, (cuda_iter_face_t) h_callback,
        faces_per_iter, min_faces_per_thread, needed_block_count, threads_per_block
      );
      gpuErrchk(cudaDeviceSynchronize());
    }
    user_data_volume_cuda_api->copy_user_data_from_device(user_data_volume_cuda_api);
    user_data_volume_cuda_api->free_cuda_memory(user_data_volume_cuda_api);
}

void simple_new_face_cuda_iterate(
    cuda4est_t * cuda4est, p4est_ghost_t * ghost_layer,
    user_data_for_cuda_t *user_data_volume_cuda_api,
    cuda_new_iter_face_api_t* new_iter_face_api
) {
  p4est_t *p4est = cuda4est->p4est;
  p4est_topidx_t      t;
  p4est_topidx_t      first_local_tree = p4est->first_local_tree;
  p4est_topidx_t      last_local_tree = p4est->last_local_tree;
  sc_array_t         *trees = p4est->trees;
  p4est_tree_t       *tree;
  size_t              si, n_quads;
  sc_array_t         *quadrants;

  size_t faces_count = cuda4est->quads_to_cuda->sides_count;
  size_t faces_per_iter = P4EST_CHILDREN / 2;
  size_t iter_count = faces_count / faces_per_iter;

  user_data_volume_cuda_api->alloc_cuda_memory(user_data_volume_cuda_api);
  void *d_user_data = user_data_volume_cuda_api->get_cuda_allocated_user_data(user_data_volume_cuda_api);

  // constants
  size_t max_block_count = 1024;
  size_t max_treads_per_block = 1024;
  size_t max_block_count_per_process = max_block_count / p4est->mpisize;
  size_t threads_per_block = 128;
  // constants

  size_t blocks_count = cuda4est->quads_to_cuda->block_count;

  
  // TODO положить callback в константную память
  //unsigned long long *d_callback, h_callback;
  //gpuErrchk(cudaMalloc((void**)&d_callback, sizeof(unsigned long long)));
  //run_setup_new_kernel_face_callback(new_iter_face_api, (cuda_new_iter_face_t*)d_callback);
  //gpuErrchk(cudaMemcpy(&h_callback, d_callback, sizeof(unsigned long long), cudaMemcpyDeviceToHost));


  run_new_simple_faces_iterate(
    cuda4est->p4est_memory_allocate_info->d_p4est,
    cuda4est->quads_to_cuda->block_count,
    cuda4est->quads_to_cuda->d_config_blocks,
    cuda4est->quads_to_cuda->d_blocks_user_data,
    cuda4est->quads_to_cuda->d_quads_levels,
    cuda4est->quads_to_cuda->d_light_sides,
    cuda4est->quads_to_cuda->shared_memory_size,
    //d_user_data, (cuda_new_iter_face_t)h_callback
    d_user_data, new_iter_face_api->callback
  );
  gpuErrchk(cudaDeviceSynchronize());

  // TODO сделать обновление d_blocks_user_data

  user_data_volume_cuda_api->copy_user_data_from_device(user_data_volume_cuda_api);
  user_data_volume_cuda_api->free_cuda_memory(user_data_volume_cuda_api);

  // TODO удалить callback из памяти
}