#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include "parameters.h"

__global__ void applyBoundaryConditionsX(float* u);
__global__ void applyBoundaryConditionsY(float* v);
__global__ void applyBoundaryConditionsZ(float* w);

void applyBoundaryConditions_kernel(float* u, float* v, float* w);

#endif