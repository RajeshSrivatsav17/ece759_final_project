#ifndef CGA_CUH
#define CGA_CUH

#include "parameters.h"

// Conjugate Gradient Algorithm kernel for solving pressure using shared memory
__global__ void cg_pressure_solver_flat(
    float* p, const float* b, float dx, int N, int maxIters, float tolerance);

#endif // CGA_CUH