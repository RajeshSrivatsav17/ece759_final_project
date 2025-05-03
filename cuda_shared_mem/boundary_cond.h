#ifndef BOUNDARY_COND_H
#define BOUNDARY_COND_H

#include "parameters.h"

// Kernel declarations for applying boundary conditions
__global__ void applyBoundaryConditionsX(float* u);
__global__ void applyBoundaryConditionsY(float* v);
__global__ void applyBoundaryConditionsZ(float* w);

#endif // BOUNDARY_COND_H