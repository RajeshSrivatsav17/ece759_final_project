#include <iostream>
#include "parameters.h"

__global__ void buoyantforce_kernel(const float* rho, const float* T, float* v) {
    extern __shared__ float shared_mem[]; // Shared memory allocation
    float* shared_rho = shared_mem;       // Shared memory for rho
    float* shared_T = &shared_mem[blockDim.x]; // Shared memory for T

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = XDIM * YDIM * ZDIM;

    // Load data into shared memory
    if (idx < totalSize) {
        shared_rho[threadIdx.x] = rho[idx];
        shared_T[threadIdx.x] = T[idx];
    }
    __syncthreads();

    // Compute buoyant force using shared memory
    if (idx < totalSize) {
        float buoy_force = alpha * shared_rho[threadIdx.x] * beta * (shared_T[threadIdx.x] - T_ambient);
        v[idx] += buoy_force * dt;
    }
}