#include <iostream>
#include "buoyantforce.h"

__global__ void buoyantforce_kernel(const float* rho, const float* T, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = XDIM * YDIM * ZDIM;

    if (idx < totalSize) {
        float buoy_force = alpha * rho[idx] * beta * (T[idx] - T_ambient);
        v[idx] += buoy_force * dt;
    }
}