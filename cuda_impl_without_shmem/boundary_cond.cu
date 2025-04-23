#include "boundary_cond.h"
#include <cuda_runtime.h>
#include <iostream>

// Kernel to apply boundary conditions on the X boundaries
__global__ void applyBoundaryConditionsX(float* u) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < YDIM && k < ZDIM) {
        int idxStart = 0 * YDIM * ZDIM + j * ZDIM + k;       // Index for x = 0
        int idxEnd = (XDIM - 1) * YDIM * ZDIM + j * ZDIM + k; // Index for x = XDIM-1
        u[idxStart] = 0.0f;
        u[idxEnd] = 0.0f;
    }
}

// Kernel to apply boundary conditions on the Y boundaries
__global__ void applyBoundaryConditionsY(float* v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < XDIM && k < ZDIM) {
        int idxStart = i * YDIM * ZDIM + 0 * ZDIM + k;       // Index for y = 0
        int idxEnd = i * YDIM * ZDIM + (YDIM - 1) * ZDIM + k; // Index for y = YDIM-1
        v[idxStart] = 0.0f;
        v[idxEnd] = 0.0f;
    }
}

// Kernel to apply boundary conditions on the Z boundaries
__global__ void applyBoundaryConditionsZ(float* w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < XDIM && j < YDIM) {
        int idxStart = i * YDIM * ZDIM + j * ZDIM + 0;       // Index for z = 0
        int idxEnd = i * YDIM * ZDIM + j * ZDIM + (ZDIM - 1); // Index for z = ZDIM-1
        w[idxStart] = 0.0f;
        w[idxEnd] = 0.0f;
    }
}

// Host function to call the CUDA kernels
void applyBoundaryConditions(float* u, float* v, float* w) {
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDimX((YDIM + blockDim.x - 1) / blockDim.x, (ZDIM + blockDim.y - 1) / blockDim.y);
    dim3 gridDimY((XDIM + blockDim.x - 1) / blockDim.x, (ZDIM + blockDim.y - 1) / blockDim.y);
    dim3 gridDimZ((XDIM + blockDim.x - 1) / blockDim.x, (YDIM + blockDim.y - 1) / blockDim.y);

    applyBoundaryConditionsX<<<gridDimX, blockDim>>>(u);
    cudaDeviceSynchronize();

    applyBoundaryConditionsY<<<gridDimY, blockDim>>>(v);
    cudaDeviceSynchronize();

    applyBoundaryConditionsZ<<<gridDimZ, blockDim>>>(w);
    cudaDeviceSynchronize();
}