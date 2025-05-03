#ifndef DIVERGENCE_H
#define DIVERGENCE_H

#include "parameters.h"

// Declaration of the divergence computation kernel
__global__ void computeDivergence_kernel(const float *u, const float *v, const float *w, float *divergence);

#endif // DIVERGENCE_H