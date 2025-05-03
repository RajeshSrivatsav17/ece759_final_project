#ifndef VELOCITY_CORRECTION_H
#define VELOCITY_CORRECTION_H

#include "parameters.h"

// Declaration of the velocity correction kernel
__global__ void velocityCorrection_kernel(float* u, float* v, float* w, const float* p);

#endif // VELOCITY_CORRECTION_H