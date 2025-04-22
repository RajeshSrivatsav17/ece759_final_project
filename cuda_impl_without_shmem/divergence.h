#include <iostream>
#include "parameters.h"

__global__ void computeDivergence_kernel(const float *u, const float *v, const float *w, float *divergence);
