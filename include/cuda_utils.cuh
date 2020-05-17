#ifndef CUDA_UTILS
#define CUDA_UTILS

#include "stdio.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline void arrayPropMemoryAllocate(
    void** d_arr, size_t arr_size,
    void* parent_ptr, size_t ptr_size,
    void* source_arr
    ) {
    gpuErrchk(cudaMalloc(d_arr, arr_size));
    gpuErrchk(cudaMemcpy(parent_ptr, d_arr, ptr_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(*d_arr, source_arr, arr_size, cudaMemcpyHostToDevice));
}

inline void arrayPropMemoryAllocate(
    void** d_arr, size_t arr_size,
    void* parent_ptr, size_t ptr_size
) {
    gpuErrchk(cudaMalloc(d_arr, arr_size));
    gpuErrchk(cudaMemcpy(parent_ptr, d_arr, ptr_size, cudaMemcpyHostToDevice));
}

inline void arrayPropMemoryUpdate(
    void** d_arr, size_t arr_size,
    void* source_arr
) {
    gpuErrchk(cudaMemcpy(*d_arr, source_arr, arr_size, cudaMemcpyHostToDevice));
}

#endif