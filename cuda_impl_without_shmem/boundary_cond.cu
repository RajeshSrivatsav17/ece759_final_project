#include "boundary_cond.h"
#include <cuda_runtime.h>
#include <iostream>

// Kernel to apply boundary conditions on the X boundaries
__global__ void applyBoundaryConditionsX(float* u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = idx / ZDIM;
    int k = idx % ZDIM;

    if (j < YDIM && k < ZDIM) {
        int idxStart = 0 * YDIM * ZDIM + j * ZDIM + k;
        int idxEnd = (XDIM - 1) * YDIM * ZDIM + j * ZDIM + k;
        u[idxStart] = 0.0f;
        u[idxEnd] = 0.0f;
    }
}

// Kernel to apply boundary conditions on the Y boundaries
__global__ void applyBoundaryConditionsY(float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / ZDIM;
    int k = idx % ZDIM;

    if (i < XDIM && k < ZDIM) {
        int idxStart = i * YDIM * ZDIM + 0 * ZDIM + k;
        int idxEnd = i * YDIM * ZDIM + (YDIM - 1) * ZDIM + k;
        v[idxStart] = 0.0f;
        v[idxEnd] = 0.0f;
    }
}

// Kernel to apply boundary conditions on the Z boundaries
__global__ void applyBoundaryConditionsZ(float* w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / YDIM;
    int j = idx % YDIM;

    if (i < XDIM && j < YDIM) {
        int idxStart = i * YDIM * ZDIM + j * ZDIM + 0;
        int idxEnd = i * YDIM * ZDIM + j * ZDIM + (ZDIM - 1);
        w[idxStart] = 0.0f;
        w[idxEnd] = 0.0f;
    }
}

// Host function to call the CUDA kernels
void applyBoundaryConditions_kernel(float* u, float* v, float* w) {
    int blockSize = 256;

    // For X boundary (j, k): total threads = YDIM * ZDIM
    int gridSizeX = (YDIM * ZDIM + blockSize - 1) / blockSize;
    applyBoundaryConditionsX<<<gridSizeX, blockSize>>>(u);
    cudaDeviceSynchronize();

    // For Y boundary (i, k): total threads = XDIM * ZDIM
    int gridSizeY = (XDIM * ZDIM + blockSize - 1) / blockSize;
    applyBoundaryConditionsY<<<gridSizeY, blockSize>>>(v);
    cudaDeviceSynchronize();

    // For Z boundary (i, j): total threads = XDIM * YDIM
    int gridSizeZ = (XDIM * YDIM + blockSize - 1) / blockSize;
    applyBoundaryConditionsZ<<<gridSizeZ, blockSize>>>(w);
    cudaDeviceSynchronize();
}