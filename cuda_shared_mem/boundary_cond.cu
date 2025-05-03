#include "boundary_cond.h"
#include <cuda_runtime.h>
#include <iostream>

// Z-major indexing: z + y * ZDIM + x * YDIM * ZDIM

// Kernel for X boundaries (j, k loop flattened)
__global__ void applyBoundaryConditionsX(float* u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = YDIM * ZDIM;

    if (idx < total) {
        int j = idx / ZDIM;
        int k = idx % ZDIM;

        int idxStart = k + j * ZDIM + 0 * YDIM * ZDIM;
        int idxEnd   = k + j * ZDIM + (XDIM - 1) * YDIM * ZDIM;

        u[idxStart] = 0.0f;
        u[idxEnd]   = 0.0f;
    }
}

// Kernel for Y boundaries (i, k loop flattened)
__global__ void applyBoundaryConditionsY(float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = XDIM * ZDIM;

    if (idx < total) {
        int i = idx / ZDIM;
        int k = idx % ZDIM;

        int idxStart = k + 0 * ZDIM + i * YDIM * ZDIM;
        int idxEnd   = k + (YDIM - 1) * ZDIM + i * YDIM * ZDIM;

        v[idxStart] = 0.0f;
        v[idxEnd]   = 0.0f;
    }
}

// Kernel for Z boundaries (i, j loop flattened)
__global__ void applyBoundaryConditionsZ(float* w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = XDIM * YDIM;

    if (idx < total) {
        int i = idx / YDIM;
        int j = idx % YDIM;

        int idxStart = 0 + j * ZDIM + i * YDIM * ZDIM;
        int idxEnd   = (ZDIM - 1) + j * ZDIM + i * YDIM * ZDIM;

        w[idxStart] = 0.0f;
        w[idxEnd]   = 0.0f;
    }
}