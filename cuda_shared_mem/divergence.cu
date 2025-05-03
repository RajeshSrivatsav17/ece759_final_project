#include "divergence.h"

__global__ void computeDivergence_kernel(const float *u, const float *v, const float *w, float *divergence) {
    extern __shared__ float shared_mem[]; // Shared memory allocation
    float *shared_u = shared_mem;         // Shared memory for u
    float *shared_v = &shared_mem[blockDim.x * blockDim.y * blockDim.z]; // Shared memory for v
    float *shared_w = &shared_v[blockDim.x * blockDim.y * blockDim.z];  // Shared memory for w

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int local_z = threadIdx.z;

    int shared_dim = blockDim.x; // Assuming cubic blocks

    // Load data into shared memory
    if (x < XDIM && y < YDIM && z < ZDIM) {
        int global_idx = z + y * ZDIM + x * YDIM * ZDIM;
        int local_idx = local_z + local_y * shared_dim + local_x * shared_dim * shared_dim;

        shared_u[local_idx] = u[global_idx];
        shared_v[local_idx] = v[global_idx];
        shared_w[local_idx] = w[global_idx];
    }
    __syncthreads();

    // Compute divergence
    if (x >= 1 && x < XDIM - 1 &&
        y >= 1 && y < YDIM - 1 &&
        z >= 1 && z < ZDIM - 1) {
        int local_idx = local_z + local_y * shared_dim + local_x * shared_dim * shared_dim;

        float du_dx = (shared_u[local_idx + shared_dim * shared_dim] - shared_u[local_idx - shared_dim * shared_dim]) / (2.0f * dx);
        float dv_dy = (shared_v[local_idx + shared_dim] - shared_v[local_idx - shared_dim]) / (2.0f * dy);
        float dw_dz = (shared_w[local_idx + 1] - shared_w[local_idx - 1]) / (2.0f * dz);

        int global_idx = z + y * ZDIM + x * YDIM * ZDIM;
        divergence[global_idx] = du_dx + dv_dy + dw_dz;
    }
}