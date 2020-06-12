#ifndef OVERLAP_QUADS_GENERATE
#define OVERLAP_QUADS_GENERATE

#include <iostream>
#include <cmath>
#include <cstdlib>

#include "p4est_to_cuda.h"

using namespace std;

typedef struct quad {
	unsigned char level;
} test_quad_t;

typedef struct cuda_bound {
    size_t indexes[2];
}cuda_bound_t;


void generate_overlap_p4est_quadrants(sc_array *quadrants, int8_t max_level, size_t *&start_indexes, unsigned char* &boundaries, sc_array *res_quadrants);
//void generate_overlap_p4est_quadrants(quad_t *quadrants, int8_t max_level, size_t *start_indexes, unsigned char* boundaries);
void test_quads();

#endif