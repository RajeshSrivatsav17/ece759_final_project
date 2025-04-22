#include "parameters.h"

__global__ void semi_lagrangian_advection_kernel(float *dst, const float *src, const float *u, const float *v, const float *w, float dt);