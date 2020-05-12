#ifndef CUDA_SIMPLE_ITERATE_H
#define CUDA_SIMPLE_ITERATE_H

#include <p4est.h>
#include <p4est_ghost.h>
#include <cuda_iterate.h>
#include "p4est_to_cuda.h"

extern "C" {
    void simple_volume_cuda_iterate(
        cuda4est_t * cuda4est, p4est_ghost_t * ghost_layer,
        user_data_for_cuda_t *user_data_volume_cuda_api,
        cuda_iter_volume_api_t* iter_volume_api
    );
}

#endif