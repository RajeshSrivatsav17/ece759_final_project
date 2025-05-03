#ifndef BUOYANTFORCE_H
#define BUOYANTFORCE_H

#include "parameters.h"

__global__ void buoyantforce_kernel(const float* rho, const float* T, float* v);

#endif // BUOYANTFORCE_H