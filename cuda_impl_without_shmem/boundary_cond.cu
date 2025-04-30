#include "boundary_cond.h"
#include <cuda_runtime.h>
#include <iostream>

// Kernel to apply boundary conditions on the X boundaries
__global__ void applyBoundaryConditionsX(float* u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = idx / ZDIM;
    int k = idx % ZDIM;

    if (j < YDIM && k < ZDIM) {
        int iStart = 0;
        int iEnd = XDIM - 1;
        int idxStart = iStart * YDIM * ZDIM + j * ZDIM + k;
        int idxEnd   = iEnd   * YDIM * ZDIM + j * ZDIM + k;
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
        int jStart = 0;
        int jEnd = YDIM - 1;
        int idxStart = i * YDIM * ZDIM + jStart * ZDIM + k;
        int idxEnd   = i * YDIM * ZDIM + jEnd   * ZDIM + k;
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
        int kStart = 0;
        int kEnd = ZDIM - 1;
        int idxStart = i * YDIM * ZDIM + j * ZDIM + kStart;
        int idxEnd   = i * YDIM * ZDIM + j * ZDIM + kEnd;
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