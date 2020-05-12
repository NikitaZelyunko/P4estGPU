#ifndef P4EST_TO_CUDA
#define P4EST_TO_CUDA

#include "cuda_utils.cuh"
#include <p4est.h>
#include <p4est_ghost.h>
#include <p4est_extended.h>

typedef struct user_data_for_cuda user_data_for_cuda_t;

typedef void (*alloc_cuda_memory_t)(user_data_for_cuda_t* user_data_api);
typedef void (*free_cuda_memory_t)(user_data_for_cuda_t* user_data_api);
typedef void* (*get_cuda_allocated_user_data_t)(user_data_for_cuda_t* user_data_api);
typedef void (*copy_user_data_from_device_t)(user_data_for_cuda_t* user_data_api);

struct user_data_for_cuda {
    void* user_data;
    void* cuda_memory_allocating_info;
    alloc_cuda_memory_t alloc_cuda_memory;
    free_cuda_memory_t free_cuda_memory;
    get_cuda_allocated_user_data_t get_cuda_allocated_user_data;
    copy_user_data_from_device_t copy_user_data_from_device;
};

typedef struct quad_user_data_allocate_info quad_user_data_allocate_info_t;

typedef void (*alloc_quad_cuda_memory_t)(quad_user_data_allocate_info_t* user_data_api);
typedef void (*free_quad_cuda_memory_t)(quad_user_data_allocate_info_t* user_data_api);
typedef void* (*get_quad_cuda_allocated_user_data_t)(quad_user_data_allocate_info_t* user_data_api);

struct quad_user_data_allocate_info {
    void* user_data;
    void* cuda_memory_allocating_info;
};

typedef struct quad_user_data_api {
    alloc_quad_cuda_memory_t alloc_cuda_memory;
    free_quad_cuda_memory_t free_cuda_memory;
    get_quad_cuda_allocated_user_data_t get_cuda_allocated_user_data;
}quad_user_data_api_t;

typedef struct cuda4est {
    p4est_t *p4est;
    user_data_for_cuda_t *user_data_api;
    quad_user_data_api_t *quad_user_data_api;
} cuda4est_t;

typedef struct p4est_quadrant_data_to_cuda {
    quad_user_data_allocate_info_t *d_user_data;
}p4est_quadrant_data_to_cuda_t;

typedef struct p4est_quadrant_to_cuda {
    p4est_quadrant *d_p4est_quadrant;
    p4est_quadrant_data_to_cuda_t *d_p4est_quadrant_data;
}p4est_quadrant_to_cuda_t;

inline p4est_quadrant_to_cuda_t* p4est_quadrant_allocate_cuda_memory(p4est_quadrant_t *quadrant, quad_user_data_api_t *user_data_api){
    p4est_quadrant_to_cuda_t *allocate_info = (p4est_quadrant_to_cuda_t*) malloc(sizeof(p4est_quadrant_to_cuda_t));
    
    quad_user_data_allocate_info_t *quad_user_data_allocate_info = (quad_user_data_allocate_info_t*) malloc(sizeof(quad_user_data_allocate_info_t));

    p4est_quadrant_t *d_quadrant;
    gpuErrchk(cudaMalloc((void**)&d_quadrant, sizeof(p4est_quadrant_t)));
    gpuErrchk(cudaMemcpy(d_quadrant, quadrant, sizeof(p4est_quadrant_t), cudaMemcpyHostToDevice));
    allocate_info->d_p4est_quadrant = d_quadrant;

    quad_user_data_allocate_info->user_data = quadrant->p.user_data;
    user_data_api->alloc_cuda_memory(quad_user_data_allocate_info);
    void *user_data = user_data_api->get_cuda_allocated_user_data(quad_user_data_allocate_info);
    gpuErrchk(cudaMemcpy(&(d_quadrant->p.user_data), &user_data, sizeof(void*), cudaMemcpyHostToDevice));
    p4est_quadrant_data_to_cuda_t *quadrant_data_info = (p4est_quadrant_data_to_cuda_t*) malloc(sizeof(p4est_quadrant_data_to_cuda_t));
    quadrant_data_info->d_user_data = quad_user_data_allocate_info;
    allocate_info->d_p4est_quadrant_data = quadrant_data_info;

    return allocate_info; 
}

inline void p4est_quadrant_free_cuda_memory(p4est_quadrant_to_cuda_t* allocate_info, quad_user_data_api_t *user_data_api) {
    quad_user_data_allocate_info_t *user_data_allocate_info = allocate_info->d_p4est_quadrant_data->d_user_data;
    user_data_api->free_cuda_memory(user_data_allocate_info);
    gpuErrchk(cudaFree(allocate_info->d_p4est_quadrant));
}

typedef struct array_allocation_cuda_info {
    p4est_quadrant_t *h_quadrants;
    size_t quadrants_length;
    p4est_quadrant_t *d_quadrants;
    quad_user_data_allocate_info_t *d_quads_data_allocate_info;

}array_allocation_cuda_info_t;

inline array_allocation_cuda_info_t* cuda_quadrants_memory_allocate(p4est_quadrant_t* quadrants, size_t quadrants_length, quad_user_data_api_t *user_data_api){
    array_allocation_cuda_info_t *allocation_info = (array_allocation_cuda_info_t*) malloc(sizeof(array_allocation_cuda_info_t));
    allocation_info->h_quadrants = quadrants;
    allocation_info->quadrants_length = quadrants_length;
    quad_user_data_allocate_info_t *quads_user_data_allocate_info = (quad_user_data_allocate_info_t*)malloc(quadrants_length * sizeof(quad_user_data_allocate_info_t));
    allocation_info->d_quads_data_allocate_info = quads_user_data_allocate_info;

    size_t quadrants_size = quadrants_length * sizeof(p4est_quadrant);
    p4est_quadrant_t *quadrants_temp = (p4est_quadrant_t*)malloc(quadrants_size);

    for (size_t i = 0; i < quadrants_length; i++)
    {
        quad_user_data_allocate_info_t *quad_user_data_allocate_info = quads_user_data_allocate_info + i;

        p4est_quadrant_t *h_quadrant = quadrants + i;
        quad_user_data_allocate_info->user_data = h_quadrant->p.user_data;
        user_data_api->alloc_cuda_memory(quad_user_data_allocate_info);
        void *user_data = user_data_api->get_cuda_allocated_user_data(quad_user_data_allocate_info);
        memcpy(&(quadrants_temp[i]),h_quadrant, sizeof(p4est_quadrant_t));
        quadrants_temp[i].p.user_data = user_data;
    }
    
    p4est_quadrant_t *d_quadrants;
    gpuErrchk(cudaMalloc((void**)&d_quadrants, quadrants_size));
    gpuErrchk(cudaMemcpy(d_quadrants, quadrants_temp, quadrants_size, cudaMemcpyHostToDevice));
    free(quadrants_temp);

    allocation_info->d_quadrants = d_quadrants;

    return allocation_info;
}

inline void cuda_quadrants_memory_free(array_allocation_cuda_info_t* allocation_info, quad_user_data_api_t *user_data_api) {
    for(size_t i = 0; i < allocation_info->quadrants_length; i++) {
        quad_user_data_allocate_info_t *quad_user_data_allocate_info = allocation_info->d_quads_data_allocate_info + i;
        user_data_api->free_cuda_memory(quad_user_data_allocate_info);
    }
    free(allocation_info->d_quads_data_allocate_info);
    gpuErrchk(cudaFree(allocation_info->d_quadrants));
}

typedef struct p4est_quadrants_to_cuda
{
    sc_array_t *d_quadrants;
    char *d_quadrants_array_temp;
    size_t quadrants_length;
    quad_user_data_allocate_info_t *quads_user_data_allocate_info;
}
p4est_quadrants_to_cuda_t;

p4est_quadrants_to_cuda* mallocForQuadrants(sc_array_t* quadrants, quad_user_data_api_t *user_data_api);
void freeMemoryForQuadrants(p4est_quadrants_to_cuda* quads_to_cuda, quad_user_data_api_t *user_data_api);


typedef struct p4est_ghost_to_cuda
{
    p4est_ghost_t *d_ghost_layer;

    char *d_ghosts_array_temp;
    size_t d_ghosts_array_temp_size;

    p4est_locidx_t *d_tree_offsets_temp;
    size_t d_tree_offsets_temp_size;

    p4est_locidx_t *d_proc_offsets_temp;
    size_t d_proc_offsets_temp_size;

}
p4est_ghost_to_cuda_t;

p4est_ghost_to_cuda_t* mallocForGhost(p4est_t* p4est, p4est_ghost_t* ghost_layer);
void freeMemoryForGhost(p4est_ghost_to_cuda_t* ghost_to_cuda);

typedef struct sc_array_cuda_memory_allocate_info {
    sc_array_t *d_sc_arr;
    char *d_sc_arr_temp;
    size_t d_sc_arr_temp_size;
}sc_array_cuda_memory_allocate_info_t;

inline sc_array_cuda_memory_allocate_info_t* scArrayMemoryAllocate(sc_array *arr) {
    sc_array_cuda_memory_allocate_info_t *sc_arr_info = (sc_array_cuda_memory_allocate_info_t*) malloc(sizeof(sc_array_cuda_memory_allocate_info_t));
    
    sc_array_t *d_sc_arr;
    char *d_sc_arr_temp;
    size_t d_sc_arr_temp_size = arr->elem_size * arr->elem_count;
    gpuErrchk(cudaMalloc((void**)&d_sc_arr, sizeof(sc_array_t)));
    gpuErrchk(cudaMemcpy(d_sc_arr, arr, sizeof(sc_array_t), cudaMemcpyHostToDevice));
    sc_arr_info->d_sc_arr = d_sc_arr;
    
    gpuErrchk(cudaMalloc((void**)&d_sc_arr_temp, d_sc_arr_temp_size));
    gpuErrchk(cudaMemcpy(&(d_sc_arr->array), &d_sc_arr_temp, sizeof(char*), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_sc_arr_temp, arr->array, d_sc_arr_temp_size, cudaMemcpyHostToDevice));

    sc_arr_info->d_sc_arr_temp = d_sc_arr_temp;
    sc_arr_info->d_sc_arr_temp_size = d_sc_arr_temp_size;
    return sc_arr_info;
}

inline void scArrayMemoryFree(sc_array_cuda_memory_allocate_info_t* sc_array_memory_allocate_info) {
    gpuErrchk(cudaFree(sc_array_memory_allocate_info->d_sc_arr_temp));
    gpuErrchk(cudaFree(sc_array_memory_allocate_info->d_sc_arr));
}

typedef struct sc_mstamp_cuda_memory_allocate_info {
    sc_mstamp *d_mstamp;
    char* d_current_temp;
    size_t d_current_temp_size;
    sc_array_cuda_memory_allocate_info_t *remember_memory_allocate_info;

}sc_mstamp_cuda_memory_allocate_info_t;

inline sc_mstamp_cuda_memory_allocate_info_t* scMstampMemoryAllocate(sc_mstamp *mstamp) {
    sc_mstamp_cuda_memory_allocate_info_t *mstamp_cuda_info = (sc_mstamp_cuda_memory_allocate_info_t*) malloc(sizeof(sc_mstamp_cuda_memory_allocate_info_t));

    sc_mstamp *d_mstamp;
    gpuErrchk(cudaMalloc((void**)&d_mstamp, sizeof(sc_mstamp)));
    gpuErrchk(cudaMemcpy(d_mstamp, mstamp, sizeof(sc_mstamp), cudaMemcpyHostToDevice));
    mstamp_cuda_info->d_mstamp = d_mstamp;

    char *d_current_temp;
    size_t d_current_temp_size = mstamp->stamp_size;
    
    arrayPropMemoryAllocate((void**)&d_current_temp, d_current_temp_size, &(d_mstamp->current), sizeof(sc_mstamp*), mstamp->current);
    mstamp_cuda_info->d_current_temp = d_current_temp;
    mstamp_cuda_info->d_current_temp_size = d_current_temp_size;

    mstamp_cuda_info->remember_memory_allocate_info = scArrayMemoryAllocate(&mstamp->remember);

    return mstamp_cuda_info;
}

inline void scMstampMemoryFree(sc_mstamp_cuda_memory_allocate_info_t* mstamp_cuda_info) {
    scArrayMemoryFree(mstamp_cuda_info->remember_memory_allocate_info);
    gpuErrchk(cudaFree(mstamp_cuda_info->d_current_temp));
    gpuErrchk(cudaFree(mstamp_cuda_info->d_mstamp));
}

typedef struct sc_mempool_cuda_memory_allocate_info {
    sc_mempool_t *d_mempool;

    #ifdef SC_MEMPOOL_MSTAMP
        sc_mstamp_cuda_memory_allocate_info_t *mstamp_memory_allocate_info;
    #else
    // todo I don't know what is this
    // struct obstackobstack;
    #endif

    sc_array_cuda_memory_allocate_info_t *freed_memory_allocate_info;

}sc_mempool_cuda_memory_allocate_info_t;

sc_mempool_cuda_memory_allocate_info_t* scMempoolMemoryAllocate(sc_mempool_t* mempool);
void scMempoolMemoryFree(sc_mempool_cuda_memory_allocate_info_t* mempool_cuda_info);

typedef struct p4est_inspect_cuda_memory_allocate_info {
    p4est_inspect_t *d_p4est_inspect;
    size_t *d_balance_zero_sends;
    size_t d_balance_zero_sends_size;
    size_t *d_balance_zero_receives;
    size_t d_balance_zero_receives_size;
} p4est_inspect_cuda_memory_allocate_info_t;

inline p4est_inspect_cuda_memory_allocate_info_t* p4est_inspect_memory_allocate(p4est_inspect_t* p4est_inspect) {
    p4est_inspect_cuda_memory_allocate_info_t *p4est_inspect_cuda_info = (p4est_inspect_cuda_memory_allocate_info_t*) malloc(sizeof(p4est_inspect_cuda_memory_allocate_info_t));

    p4est_inspect_t *d_p4est_inspect;
    gpuErrchk(cudaMalloc((void**)&d_p4est_inspect, sizeof(p4est_inspect_t)));
    if(p4est_inspect != NULL) {
        gpuErrchk(cudaMemcpy(d_p4est_inspect, p4est_inspect, sizeof(p4est_inspect_t), cudaMemcpyHostToDevice));
    }
    p4est_inspect_cuda_info->d_p4est_inspect = d_p4est_inspect;

    size_t *d_balance_zero_sends;
    size_t d_balance_zero_sends_size = 2;
    if(p4est_inspect != NULL) {
        arrayPropMemoryAllocate((void**)&d_balance_zero_sends, d_balance_zero_sends_size, &(d_p4est_inspect->balance_zero_sends), sizeof(size_t*), p4est_inspect->balance_zero_sends);
    } else {
        arrayPropMemoryAllocate((void**)&d_balance_zero_sends, d_balance_zero_sends_size, &(d_p4est_inspect->balance_zero_sends), sizeof(size_t*));
    }
    
    p4est_inspect_cuda_info->d_balance_zero_sends = d_balance_zero_sends;
    p4est_inspect_cuda_info->d_balance_zero_sends_size = d_balance_zero_sends_size;

    size_t *d_balance_zero_receives;
    size_t d_balance_zero_receives_size = 2;
    if(p4est_inspect != NULL) {
        arrayPropMemoryAllocate((void**)&d_balance_zero_receives, d_balance_zero_receives_size, &(d_p4est_inspect->balance_zero_receives), sizeof(size_t*), p4est_inspect->balance_zero_receives);
    } else {
        arrayPropMemoryAllocate((void**)&d_balance_zero_receives, d_balance_zero_receives_size, &(d_p4est_inspect->balance_zero_receives), sizeof(size_t*));
    }
    p4est_inspect_cuda_info->d_balance_zero_receives = d_balance_zero_receives;
    p4est_inspect_cuda_info->d_balance_zero_receives_size = d_balance_zero_receives_size;

    return p4est_inspect_cuda_info;
}

inline void p4est_inspect_memory_free(p4est_inspect_cuda_memory_allocate_info_t* p4est_inspect_cuda_info) {
    gpuErrchk(cudaFree(p4est_inspect_cuda_info->d_balance_zero_receives));
    gpuErrchk(cudaFree(p4est_inspect_cuda_info->d_balance_zero_sends));
    gpuErrchk(cudaFree(p4est_inspect_cuda_info->d_p4est_inspect));
}

typedef struct p4est_connectivity_cuda_memory_allocate_info {
    p4est_connectivity_t *d_connectivity;

    double *d_vertices;
    size_t d_vertices_size;

    p4est_topidx_t *d_tree_to_vertex;
    size_t d_tree_to_vertex_size;

    char *d_tree_to_attr;
    size_t d_tree_to_attr_size;

    p4est_topidx_t *d_tree_to_tree;
    size_t d_tree_to_tree_size;

    int8_t *d_tree_to_face;
    size_t d_tree_to_face_size;

    p4est_topidx_t *d_tree_to_corner;
    size_t d_tree_to_corner_size;

    p4est_topidx_t *d_ctt_offset;
    size_t d_ctt_offset_size;

    p4est_topidx_t *d_corner_to_tree;
    size_t d_corner_to_tree_size;

    int8_t *d_corner_to_corner;
    size_t d_corner_to_corner_size;

}p4est_connectivity_cuda_memory_allocate_info_t;

p4est_connectivity_cuda_memory_allocate_info_t* p4est_connectivity_memory_alloc(p4est_connectivity_t* p4est_connectivity);
void p4est_connectivity_memory_free(p4est_connectivity_cuda_memory_allocate_info_t* allocate_info);

typedef struct p4est_cuda_memory_allocate_info {
    p4est_t *d_p4est;
    user_data_for_cuda_t *d_user_data_api;

    p4est_gloidx_t *d_global_first_quadrant;
    size_t d_global_first_quadrant_size;

    array_allocation_cuda_info_t *d_global_first_position_allocate_info;

    p4est_connectivity_cuda_memory_allocate_info_t *d_connectivity_cuda_allocate_info;
    sc_array_cuda_memory_allocate_info_t *d_trees_cuda_allocate_info;
    // sc_mempool_cuda_memory_allocate_info_t *d_user_data_pool_cuda_allocate_info;
    sc_mempool_t *d_user_data_pool;
    sc_mempool_cuda_memory_allocate_info_t *d_quadrant_pool_cuda_allocate_info;
    p4est_inspect_cuda_memory_allocate_info_t *d_inspect_cuda_allocate_info;
}p4est_cuda_memory_allocate_info_t;

p4est_cuda_memory_allocate_info_t* p4est_memory_alloc(cuda4est_t* cuda4est);
void p4est_memory_free(p4est_cuda_memory_allocate_info_t* allocate_info, quad_user_data_api_t* user_data_api);

#endif