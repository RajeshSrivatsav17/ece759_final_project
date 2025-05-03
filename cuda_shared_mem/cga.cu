#include "cga.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "parameters.h"

// Updated IDX macro for Z-major ordering
#define IDX(i, j, k) ((k) + (j) * ZDIM + (i) * YDIM * ZDIM)
#define LOCAL_IDX(i, j, k) ((k) + (j) * (BLOCK_SIZE + 2) + (i) * (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2))

__device__ float laplacian_shared(const float* shared_p, int tx, int ty, int tz, int shared_dim, float dx) {
    float center = shared_p[LOCAL_IDX(tx, ty, tz)];
    float sum = shared_p[LOCAL_IDX(tx - 1, ty, tz)] +
                shared_p[LOCAL_IDX(tx + 1, ty, tz)] +
                shared_p[LOCAL_IDX(tx, ty - 1, tz)] +
                shared_p[LOCAL_IDX(tx, ty + 1, tz)] +
                shared_p[LOCAL_IDX(tx, ty, tz - 1)] +
                shared_p[LOCAL_IDX(tx, ty, tz + 1)];
    return (sum - 6.0f * center) / (dx * dx);
}

__global__ void cg_pressure_solver_flat(
    float* p, const float* b, float dx, int N, int maxIters, float tolerance) {
    constexpr int X = BLOCK_SIZE;
    constexpr int Y = BLOCK_SIZE;
    constexpr int Z = BLOCK_SIZE;

    __shared__ float shared_p[(X + 2) * (Y + 2) * (Z + 2)];
    __shared__ float shared_r[(X + 2) * (Y + 2) * (Z + 2)];
    __shared__ float shared_d[(X + 2) * (Y + 2) * (Z + 2)];
    __shared__ float shared_q[(X + 2) * (Y + 2) * (Z + 2)];

    int i = blockIdx.x * X + threadIdx.x;
    int j = blockIdx.y * Y + threadIdx.y;
    int k = blockIdx.z * Z + threadIdx.z;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    int global_idx = IDX(i, j, k);
    int local_idx = LOCAL_IDX(tx, ty, tz);

    float r = 0.0f, d = 0.0f, q = 0.0f;
    float alpha = 0.0f, beta = 0.0f;
    float delta_new = 0.0f, delta_old = 0.0f;
    float dq = 0.0f;

    // Initialize shared memory and global memory
    if (i < XDIM && j < YDIM && k < ZDIM) {
        shared_p[local_idx] = p[global_idx];
        shared_r[local_idx] = b[global_idx] - laplacian_shared(shared_p, tx, ty, tz, BLOCK_SIZE + 2, dx);
        shared_d[local_idx] = shared_r[local_idx];
    } else {
        shared_p[local_idx] = 0.0f;
        shared_r[local_idx] = 0.0f;
        shared_d[local_idx] = 0.0f;
    }
    __syncthreads();

    // Compute initial delta_new
    if (i < XDIM && j < YDIM && k < ZDIM) {
        float r_local = shared_r[local_idx];
        atomicAdd(&delta_new, r_local * r_local);
    }
    __syncthreads();

    for (int iter = 0; iter < maxIters && sqrtf(delta_new) > tolerance; ++iter) {
        // Compute q = A * d using shared memory
        if (i < XDIM && j < YDIM && k < ZDIM) {
            shared_q[local_idx] = laplacian_shared(shared_d, tx, ty, tz, BLOCK_SIZE + 2, dx);
        }
        __syncthreads();

        // Compute dot product d.q
        if (i < XDIM && j < YDIM && k < ZDIM) {
            float d_local = shared_d[local_idx];
            float q_local = shared_q[local_idx];
            atomicAdd(&dq, d_local * q_local);
        }
        __syncthreads();

        // Compute alpha = delta_new / d.q
        if (dq != 0.0f) {
            alpha = delta_new / dq;
        }
        __syncthreads();

        // Update p and r
        if (i < XDIM && j < YDIM && k < ZDIM) {
            shared_p[local_idx] += alpha * shared_d[local_idx];
            shared_r[local_idx] -= alpha * shared_q[local_idx];
        }
        __syncthreads();

        // Compute new delta_new
        delta_old = delta_new;
        delta_new = 0.0f;
        if (i < XDIM && j < YDIM && k < ZDIM) {
            float r_local = shared_r[local_idx];
            atomicAdd(&delta_new, r_local * r_local);
        }
        __syncthreads();

        // Compute beta = delta_new / delta_old
        if (delta_old != 0.0f) {
            beta = delta_new / delta_old;
        }
        __syncthreads();

        // Update d
        if (i < XDIM && j < YDIM && k < ZDIM) {
            shared_d[local_idx] = shared_r[local_idx] + beta * shared_d[local_idx];
        }
        __syncthreads();
    }

    // Write back results to global memory
    if (i < XDIM && j < YDIM && k < ZDIM) {
        p[global_idx] = shared_p[local_idx];
    }
}