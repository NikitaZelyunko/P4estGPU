cmake_minimum_required(VERSION 3.16)
project(P4estGpu CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(P4estGpu main.cu)

set_target_properties(
        P4estGpu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)