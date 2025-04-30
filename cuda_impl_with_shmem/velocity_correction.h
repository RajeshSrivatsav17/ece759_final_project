#include <iostream>
#include "cuda.h"
#include "Utilities.h"

__global__ void velocityCorrection_kernel(float* u, float* v, float* w, float* p);