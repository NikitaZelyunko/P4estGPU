#ifndef CUDA_SIMPLE_ITERATE_H
#define CUDA_SIMPLE_ITERATE_H

#include <p4est.h>
#include <p4est_ghost.h>
#include <cuda_iterate.h>
#include "p4est_to_cuda.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "stdio.h"


extern "C" {
    void simple_volume_cuda_iterate(
        cuda4est_t * cuda4est, p4est_ghost_t * ghost_layer,
        user_data_for_cuda_t *user_data_volume_cuda_api,
        cuda_iter_volume_api_t* iter_volume_api
    );

    void simple_face_cuda_iterate(
        cuda4est_t * cuda4est, p4est_ghost_t * ghost_layer,
        user_data_for_cuda_t *user_data_volume_cuda_api,
        cuda_iter_face_api_t* iter_face_api
    );

    void simple_new_face_cuda_iterate(
        cuda4est_t * cuda4est, p4est_ghost_t * ghost_layer,
        user_data_for_cuda_t *user_data_volume_cuda_api,
        cuda_new_iter_face_api_t* iter_face_api,
        cuda_new_iter_quad_api_t* iter_quad_api
    );

    void run_setup_kernel_volume_callback(cuda_iter_volume_api_t* iter_volume_api, cuda_iter_volume_t* d_callback);
    void run_setup_kernel_face_callback(cuda_iter_face_api_t* iter_face_api, cuda_iter_face_t* d_callback);
    void run_setup_new_kernel_face_callback(cuda_new_iter_face_api_t* new_iter_face_api, cuda_new_iter_face_t* d_callback);

    void run_simple_quadrants_iterate(p4est_quadrant_t* quadrants, void* quads_data, size_t quad_data_size,
        p4est_ghost_t* ghost_layer,
        p4est_t* p4est, p4est_topidx_t treeId, 
        void* user_data, cuda_iter_volume_t iter_volume,
        size_t quads_count, size_t quads_per_thread,
        size_t needed_block_count, size_t threads_per_block);
    
    void run_simple_faces_iterate(p4est_t* p4est, p4est_ghost_t* ghost_layer,
        sc_array_t* quadrants,
        p4est_iter_face_side_t* faces, size_t faces_count,
        void* user_data, cuda_iter_face_t iter_face,
        size_t faces_per_iter, size_t faces_per_thread, size_t needed_block_count, size_t threads_per_block);
    
    void run_new_simple_faces_iterate(cuda4est_t* cuda4est, char* ctx,
        size_t block_count,
        size_t *block_config,
        void *blocks_user_data,
        unsigned char* quads_levels,
        cuda_light_face_side_t* sides,
        size_t shared_memory_size,
        void* user_data, 
        cuda_new_iter_face_api_t *new_iter_face_api,
        cuda_new_iter_quad_api_t *new_iter_quad_api
    );
}

#endif