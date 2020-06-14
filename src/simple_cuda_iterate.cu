#include "simple_cuda_iterate.h"

__global__ void
simple_quadrants_iterate(
  p4est_quadrant_t* quadrants, void* quads_data, size_t quad_data_size, p4est_ghost_t* ghost_layer,
  p4est_t* p4est, p4est_topidx_t treeId, 
  void* user_data, cuda_iter_volume_t iter_volume,
  size_t quads_count, size_t quads_per_thread)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t cursor = i * quads_per_thread; 
  if(cursor < quads_count) {
    size_t elems_remaining;
    if(cursor >= quads_count - quads_per_thread) {
      elems_remaining = quads_count - cursor;
    } else {
      elems_remaining = quads_per_thread;
    }
    p4est_quadrant_t *quad_cursor = quadrants + cursor;
    void* quad_data_cursor = quads_data + cursor * quad_data_size;
    for(int j = 0; j < elems_remaining; j++, quad_data_cursor=quad_data_cursor + quad_data_size) {
      iter_volume(
        p4est,
        ghost_layer,
        quad_cursor++,
        quad_data_cursor,
        cursor++,
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

void run_setup_new_kernel_face_callback(cuda_new_iter_face_api_t* new_iter_face_api, cuda_new_iter_face_t* d_callback) {
  new_iter_face_api->setup_kernel<<<1,1>>>(d_callback);
}

void run_simple_quadrants_iterate(p4est_quadrant_t* quadrants, void* quads_data, size_t quad_data_size,
    p4est_ghost_t* ghost_layer,
    p4est_t* p4est, p4est_topidx_t treeId, 
    void* user_data, cuda_iter_volume_t iter_volume,
    size_t quads_count, size_t quads_per_thread,
    size_t needed_block_count, size_t threads_per_block
) {
    simple_quadrants_iterate<<<needed_block_count, threads_per_block>>>(
        quadrants, quads_data, quad_data_size,
        ghost_layer,
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
//  sc_array_t *ghost_quadrants = &(ghost_layer->ghosts);
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < faces_count) {
    size_t elems_remaining;
    if(i >= faces_count - faces_per_thread) {
      elems_remaining = faces_count - i;
    } else {
      elems_remaining = faces_per_thread;
    }
    p4est_iter_face_side_t* cursor = faces + i * faces_per_thread * faces_per_iter;
    for(int j = 0; j < elems_remaining; j++, cursor+=faces_per_iter) {
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

__global__ void
simple_new_faces_iterate(
    p4est_t* p4est,
    size_t *block_configs,
    void *blocks_user_data,
    unsigned char* quads_levels,
    cuda_light_face_side_t* sides,
    void* user_data, cuda_new_iter_face_t new_iter_face)
{ 
  extern __shared__ char array[];

  size_t *global_block_config = block_configs + blockIdx.x * 7;

  size_t global_block_quads_start_index = global_block_config[0];
  size_t global_block_quads_count = global_block_config[1];

  size_t global_block_start_byte_index = global_block_config[2];
  size_t global_block_quads_bytes_count = global_block_config[3];
  
  size_t global_block_output_quads_count = global_block_config[4];

  size_t global_block_faces_start_index = global_block_config[5];
  size_t global_block_faces_count = global_block_config[6];
  
  
  char* block_user_data = (char*)array;
  unsigned char* block_quads_levels = (unsigned char*)(global_block_quads_bytes_count + array); 
  cuda_light_face_side_t *face_sides = (cuda_light_face_side_t*)(global_block_quads_bytes_count + global_block_quads_count + array);

  /*
  if(threadIdx.x == 0){
    printf("[cuda] %d-block  %lu, %lu, %lu, %lu, %lu, %lu, %lu\n",
      blockIdx.x,
      global_block_quads_start_index,
      global_block_quads_count,
      global_block_start_byte_index,
      global_block_quads_bytes_count,
      global_block_output_quads_count,
      global_block_faces_start_index,
      global_block_faces_count
    );
  }
  */
  
  
  size_t faces_count = global_block_faces_count;
  size_t faces_per_thread;
  if(faces_count % blockDim.x) {
    faces_per_thread = faces_count / blockDim.x + 1;
  } else {
    faces_per_thread = faces_count / blockDim.x;
  }
  
  size_t quads_count = global_block_quads_count;
  size_t quads_per_thread;
  if(quads_count % blockDim.x) { 
    quads_per_thread = quads_count / blockDim.x + 1;
  } else {
    quads_per_thread = quads_count / blockDim.x;
  }

  /*
  if(threadIdx.x == 0){
    printf("[cuda] %d-block faces_count: %lu, faces_per_thread: %lu\n",
      blockIdx.x,
      faces_count,
      faces_per_thread
    );
  }
  */

  /*
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    printf("0 - memory_size: %d\n", global_block_quads_bytes_count + global_block_quads_count + faces_count * sizeof(cuda_light_face_side_t) * 2);
  }

  if(threadIdx.x == 0 && blockIdx.x == 1) {
    printf("1 - memory_size: %d\n", global_block_quads_bytes_count + global_block_quads_count + faces_count * sizeof(cuda_light_face_side_t) * 2);
  }
  */
  
  
  
  int i = threadIdx.x;
  size_t faces_remaining = 0;
  size_t face_start_index = faces_per_thread * i;
  if(face_start_index < faces_count){
    faces_remaining = faces_count - face_start_index;
    if(faces_remaining > faces_per_thread) {
      faces_remaining = faces_per_thread;
    }
  }
  size_t sides_start_index = (global_block_faces_start_index + face_start_index) * 2;
  size_t local_start_index = face_start_index * 2;

  for(size_t j = 0; j < faces_remaining; j++, sides_start_index++, local_start_index++) {
    face_sides[local_start_index] = sides[sides_start_index];
    sides_start_index++;
    local_start_index++;
    face_sides[local_start_index] = sides[sides_start_index];
  }
  
  size_t quads_remaining = 0;
  size_t quad_start_index = quads_per_thread * i;
  if(quad_start_index < quads_count){
    quads_remaining = quads_count - quad_start_index;
    if(quads_remaining > quads_per_thread) {
      quads_remaining = quads_per_thread;
    }
  }
  
  size_t quad_start_byte = global_block_faces_start_index;
  size_t user_data_size = global_block_quads_bytes_count / global_block_quads_count;

  for(size_t j = 0; j < quads_remaining; j++, quad_start_byte+=user_data_size) {
    size_t user_data_byte_end = quad_start_byte + user_data_size;
    for(size_t byte_index = quad_start_byte; byte_index < user_data_byte_end; byte_index++) {
      block_user_data[byte_index] = (char)(blocks_user_data + byte_index);
    }
    size_t current_quad_index = quad_start_index + j;
    block_quads_levels[current_quad_index] = quads_levels[current_quad_index];
  }
  __shared__ size_t output_quads_count;
  output_quads_count = global_block_output_quads_count;
  
  __syncthreads();

  size_t face_side_start_index = face_start_index * 2;
  size_t sides_remaining = faces_remaining * 2;
  for(size_t j = 0; j < sides_remaining; j+=2) {
    new_iter_face(
      p4est,
      output_quads_count,
      block_user_data,
      block_quads_levels,
      face_sides + face_side_start_index + j
    );
  }
}

void run_new_simple_faces_iterate(p4est_t* p4est,
  size_t block_count,
  size_t *block_configs,
  void *blocks_user_data,
  unsigned char* quads_levels,
  cuda_light_face_side_t* sides,
  size_t shared_memory_size,
  void* user_data, cuda_new_iter_face_t new_iter_face
){
  simple_new_faces_iterate<<<block_count,128, shared_memory_size>>>(p4est, block_configs, blocks_user_data, quads_levels, sides, user_data, new_iter_face);
  gpuErrchk(cudaDeviceSynchronize());
}
