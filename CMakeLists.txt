cmake_minimum_required(VERSION 3.16)
project(P4estGpu LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_C_COMPILER "mpicc")
set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_CUDA_COMPILER "nvcc")

set(MPI_INCLUDE "/usr/include/mpich")

set(CUDA_INCLUDE "/usr/local/cuda/include")

set(P4EST_INSTALL "/home/nikita/Diplom/P4estInstall")
set(P4EST_INCLUDE "${P4EST_INSTALL}/include" "${P4EST_INSTALL}/include/sc_builtin")
set(P4EST_LIB "${P4EST_INSTALL}/lib")


set(MAIN_INCLUDE ${CMAKE_HOME_DIRECTORY}/include)
set(MAIN_SRC ${CMAKE_HOME_DIRECTORY}/src)

include_directories(${MPI_INCLUDE} ${P4EST_INCLUDE} ${CUDA_INCLUDE} ${MAIN_INCLUDE})
link_directories(${P4EST_LIB})

set_source_files_properties(${MAIN_SRC}/main.cu ${MAIN_SRC}/simple_cuda_iterate.cu ${MAIN_INCLUDE}/main_cuda_callbacks.cu PROPERTIES LANGUAGE "CUDA")

add_executable(P4estGpu
        ${MAIN_SRC}/cuda_iterate.cpp 
        ${MAIN_SRC}/p4est_to_cuda.cpp
        ${MAIN_SRC}/overlap_quads_generate.cpp
        ${MAIN_INCLUDE}/p4est_to_cuda.h
        ${MAIN_INCLUDE}/cuda_iterate_loop_args.h

        ${MAIN_SRC}/simple_cuda_iterate.cpp
        ${MAIN_SRC}/simple_cuda_iterate.cu
        ${MAIN_SRC}/main.cu
)

target_link_libraries(P4estGpu p4est sc m)

set_target_properties(
        P4estGpu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)