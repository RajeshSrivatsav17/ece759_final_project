#ifndef ADVECT_H
#define ADVECT_H

#include "parameters.h"

// Declaration of the semi-Lagrangian advection kernel using shared memory
__global__ void semi_lagrangian_advection_kernel(float *dst, const float *src, const float *u, const float *v, const float *w, float dt);

#endif // ADVECT_H