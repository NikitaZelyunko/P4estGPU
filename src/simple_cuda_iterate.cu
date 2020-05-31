#include "simple_cuda_iterate.h"

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

void run_setup_kernel_volume_callback(cuda_iter_volume_api_t* iter_volume_api, cuda_iter_volume_t* d_callback) {
    iter_volume_api->setup_kernel<<<1,1>>>(d_callback);
}

void run_setup_kernel_face_callback(cuda_iter_face_api_t* iter_face_api, cuda_iter_face_t* d_callback) {
    iter_face_api->setup_kernel<<<1,1>>>(d_callback);
}

void run_simple_quadrants_iterate(sc_array_t* quadrants, p4est_ghost_t* ghost_layer,
    p4est_t* p4est, p4est_topidx_t treeId, 
    void* user_data, cuda_iter_volume_t iter_volume,
    size_t quads_count, size_t quads_per_thread,
    size_t needed_block_count, size_t threads_per_block
) {

    simple_quadrants_iterate<<<needed_block_count, threads_per_block>>>(
        quadrants, ghost_layer,
        p4est, treeId,
        user_data, iter_volume,
        quads_count, quads_per_thread
    );
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void
simple_faces_iterate(
    p4est_t* p4est, p4est_ghost_t* ghost_layer,
    sc_array_t* quadrants, 
    p4est_iter_face_side_t* faces, size_t faces_count,
    void* user_data, cuda_iter_face_t iter_face,
    size_t faces_per_iter, size_t faces_per_thread)
{
  sc_array_t *ghost_quadrants = &(ghost_layer->ghosts);
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < faces_count) {
    size_t elems_remaining;
    if(i >= faces_count - faces_per_thread) {
      elems_remaining = faces_count - i;
    } else {
      elems_remaining = faces_per_thread;
    }
    p4est_iter_face_side_t* cursor = faces + i * faces_per_iter;
    for(int j = 0; j < elems_remaining; j++, cursor+=faces_per_iter) {
        p4est_iter_face_side_t* current_face = cursor;
        if(current_face->is_hanging) {
          p4est_quadrant_t** quads = current_face->is.hanging.quad;
          p4est_locidx_t *quadid = current_face->is.hanging.quadid;
          if(current_face->is.hanging.is_ghost[0]){
            quads[0] = p4est_device_quadrant_array_index(ghost_quadrants, quadid[0]);
          } else {
            quads[0] = p4est_device_quadrant_array_index(quadrants, quadid[0]);
          }

          if(current_face->is.hanging.is_ghost[1]) {
            quads[1] = p4est_device_quadrant_array_index(ghost_quadrants, quadid[1]);
          } else {
            quads[1] = p4est_device_quadrant_array_index(quadrants, quadid[1]);
          }
          
        } else {
          if(current_face->is.full.is_ghost){
            current_face->is.full.quad = p4est_device_quadrant_array_index(ghost_quadrants, current_face->is.full.quadid);
          } else {
            current_face->is.full.quad = p4est_device_quadrant_array_index(quadrants, current_face->is.full.quadid);
          }
        }
        current_face++;
        if(current_face->is_hanging) {
          p4est_quadrant_t** quads = current_face->is.hanging.quad;
          p4est_locidx_t *quadid = current_face->is.hanging.quadid;
          
          if(current_face->is.hanging.is_ghost[0]){
            quads[0] = p4est_device_quadrant_array_index(ghost_quadrants, quadid[0]);
          } else {
            quads[0] = p4est_device_quadrant_array_index(quadrants, quadid[0]);
          }

          if(current_face->is.hanging.is_ghost[1]) {
            quads[1] = p4est_device_quadrant_array_index(ghost_quadrants, quadid[1]);
          } else {
            quads[1] = p4est_device_quadrant_array_index(quadrants, quadid[1]);
          }
        } else {
          if(current_face->is.full.is_ghost){
            current_face->is.full.quad = p4est_device_quadrant_array_index(ghost_quadrants, current_face->is.full.quadid);
          } else {
            current_face->is.full.quad = p4est_device_quadrant_array_index(quadrants, current_face->is.full.quadid);
          }
        }
        iter_face(
            p4est,
            ghost_layer,
            cursor,
            user_data
        );
    }
  }
}

void run_simple_faces_iterate(p4est_t* p4est, p4est_ghost_t* ghost_layer,
    sc_array_t* quadrants,
    p4est_iter_face_side_t* faces, size_t faces_count,
    void* user_data, cuda_iter_face_t iter_face,
    size_t faces_per_iter, size_t faces_per_thread, size_t needed_block_count, size_t threads_per_block) {
    
    simple_faces_iterate<<<needed_block_count, threads_per_block>>>(
        p4est, ghost_layer,
        quadrants,
        faces, faces_count,
        user_data, iter_face,
        faces_per_iter, faces_per_thread
    );
    gpuErrchk(cudaDeviceSynchronize());
}
