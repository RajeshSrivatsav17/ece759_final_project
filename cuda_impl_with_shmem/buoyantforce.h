#include <iostream>
#include "parameters.h"

__global__ void buoyantforce_kernel(const float* rho, const float* T, float* v);