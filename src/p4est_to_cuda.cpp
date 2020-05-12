#include "p4est_to_cuda.h"

p4est_quadrants_to_cuda_t* mallocForQuadrants(sc_array_t *quadrants, quad_user_data_api_t *user_data_api) {
    p4est_quadrants_to_cuda_t *quads_to_cuda = (p4est_quadrants_to_cuda_t*) malloc(sizeof(p4est_quadrants_to_cuda_t));
    sc_array_t *d_quadrants;
    size_t d_quadrants_array_size = quadrants->elem_size * quadrants->elem_count;
    gpuErrchk(cudaMalloc((void**)&d_quadrants, sizeof(sc_array_t)));
    gpuErrchk(cudaMemcpy(d_quadrants, quadrants, sizeof(sc_array_t), cudaMemcpyHostToDevice));
    quads_to_cuda->d_quadrants = d_quadrants;
    quads_to_cuda->quadrants_length = quadrants->elem_count;

    char *h_quadrants_array_temp = (char*) malloc(d_quadrants_array_size * sizeof(char));
    quad_user_data_allocate_info_t *quads_user_data_allocate_info = (quad_user_data_allocate_info_t*)malloc(sizeof(quad_user_data_allocate_info_t) * quadrants->elem_count);
    size_t quad_size = sizeof(p4est_quadrant_t);
    char *cursor = h_quadrants_array_temp;
    for(size_t i = 0; i < quadrants->elem_count; i++, cursor+=quad_size) {
        p4est_quadrant_t *temp_quad = (p4est_quadrant_t*) malloc(sizeof(p4est_quadrant_t));
        memcpy(temp_quad, p4est_quadrant_array_index (quadrants, i), sizeof(p4est_quadrant_t));
        quad_user_data_allocate_info_t * temp_quad_user_data = quads_user_data_allocate_info + i;
        temp_quad_user_data->user_data = temp_quad->p.user_data;
        user_data_api->alloc_cuda_memory(temp_quad_user_data);
        temp_quad->p.user_data = user_data_api->get_cuda_allocated_user_data(temp_quad_user_data);
        memcpy(cursor, temp_quad, quad_size);
    }
    quads_to_cuda->quads_user_data_allocate_info = quads_user_data_allocate_info;
    char *d_quadrants_array_temp;
    gpuErrchk(cudaMalloc((void**)&d_quadrants_array_temp, d_quadrants_array_size));
    gpuErrchk(cudaMemcpy(d_quadrants_array_temp, h_quadrants_array_temp, d_quadrants_array_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_quadrants->array), &d_quadrants_array_temp, sizeof(char*), cudaMemcpyHostToDevice));
    free(h_quadrants_array_temp);

    quads_to_cuda->d_quadrants_array_temp = d_quadrants_array_temp;

    return quads_to_cuda;
}

void freeMemoryForQuadrants(p4est_quadrants_to_cuda_t* quads_to_cuda, quad_user_data_api_t *user_data_api) {
    for(size_t i = 0; i < quads_to_cuda->quadrants_length; i++) {
        quad_user_data_allocate_info_t *quad_user_data_allocate_info = &(quads_to_cuda->quads_user_data_allocate_info[i]);
        user_data_api->free_cuda_memory(quad_user_data_allocate_info);
    }
    free(quads_to_cuda->quads_user_data_allocate_info);
    gpuErrchk(cudaFree(quads_to_cuda->d_quadrants_array_temp));
    gpuErrchk(cudaFree(quads_to_cuda->d_quadrants));
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
    gpuErrchk(cudaFree(ghost_to_cuda->d_ghosts_array_temp));
    gpuErrchk(cudaFree(ghost_to_cuda->d_tree_offsets_temp));
    gpuErrchk(cudaFree(ghost_to_cuda->d_proc_offsets_temp));
    gpuErrchk(cudaFree(ghost_to_cuda->d_ghost_layer));
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