#include "velocity_correction.h"

#define IDX(i, j, k) ((k) + (j) * ZDIM + (i) * YDIM * ZDIM)
#define LOCAL_IDX(i, j, k) ((k) + (j) * (BLOCK_SIZE + 2) + (i) * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2))

__global__ void velocityCorrection_kernel(float* u, float* v, float* w, const float* p) {
    extern __shared__ float shared_p[]; // Shared memory for pressure

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    int global_idx = IDX(i, j, k);
    int local_idx = LOCAL_IDX(tx, ty, tz);

    // Load pressure into shared memory
    if (i < XDIM && j < YDIM && k < ZDIM) {
        shared_p[local_idx] = p[global_idx];
    } else {
        shared_p[local_idx] = 0.0f; // Handle out-of-bound threads
    }

    // Load halo regions
    if (tx == 1 && i > 0) {
        shared_p[LOCAL_IDX(tx - 1, ty, tz)] = p[IDX(i - 1, j, k)];
    }
    if (tx == blockDim.x && i < XDIM - 1) {
        shared_p[LOCAL_IDX(tx + 1, ty, tz)] = p[IDX(i + 1, j, k)];
    }
    if (ty == 1 && j > 0) {
        shared_p[LOCAL_IDX(tx, ty - 1, tz)] = p[IDX(i, j - 1, k)];
    }
    if (ty == blockDim.y && j < YDIM - 1) {
        shared_p[LOCAL_IDX(tx, ty + 1, tz)] = p[IDX(i, j + 1, k)];
    }
    if (tz == 1 && k > 0) {
        shared_p[LOCAL_IDX(tx, ty, tz - 1)] = p[IDX(i, j, k - 1)];
    }
    if (tz == blockDim.z && k < ZDIM - 1) {
        shared_p[LOCAL_IDX(tx, ty, tz + 1)] = p[IDX(i, j, k + 1)];
    }
    __syncthreads();

    // Ensure we are not on boundary cells
    if (i > 0 && i < XDIM - 1 &&
        j > 0 && j < YDIM - 1 &&
        k > 0 && k < ZDIM - 1) {
        int idx_center = LOCAL_IDX(tx, ty, tz);
        int idx_ip = LOCAL_IDX(tx + 1, ty, tz);
        int idx_jp = LOCAL_IDX(tx, ty + 1, tz);
        int idx_kp = LOCAL_IDX(tx, ty, tz + 1);

        // Correct velocity using pressure gradient
        u[global_idx] -= (shared_p[idx_ip] - shared_p[idx_center]) / dx;
        v[global_idx] -= (shared_p[idx_jp] - shared_p[idx_center]) / dy;
        w[global_idx] -= (shared_p[idx_kp] - shared_p[idx_center]) / dz;
    }
}