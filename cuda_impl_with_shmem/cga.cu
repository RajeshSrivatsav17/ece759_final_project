#include <cuda_runtime.h>
#include <iostream>
#include "parameters.h"

#define IDX(i, j, k) ((k) + (j) * ZDIM + (i) * YDIM * ZDIM)
#define LOCAL_IDX(i, j, k) ((k) + (j) * (Z + 2) + (i) * (Y + 2) * (Z + 2))

__global__ void solve_pressure_cg_flat(float* p, const float* b, float dx) {
    constexpr int X = 8;  // tile sizes
    constexpr int Y = 8;
    constexpr int Z = 8;

    __shared__ float s_d[(X + 2) * (Y + 2) * (Z + 2)];

    int i = blockIdx.x * X + threadIdx.x;
    int j = blockIdx.y * Y + threadIdx.y;
    int k = blockIdx.z * Z + threadIdx.z;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    int N = XDIM * YDIM * ZDIM;

    int index = IDX(i, j, k);
    int local_index = LOCAL_IDX(tx, ty, tz);

    float r = 0.0f, d = 0.0f, q = 0.0f;
    float alpha = 0.0f, beta = 0.0f;
    float delta_new = 0.0f, delta_old = 0.0f;
    float dq = 0.0f;

    if (i < XDIM && j < YDIM && k < ZDIM) {
        p[index] = 0.0f;
        r = b[index];
        d = r;
    }

    for (int iter = 0; iter < 100; ++iter) {
        // --- Load with halo into shared memory
        if (i < XDIM && j < YDIM && k < ZDIM)
            s_d[local_index] = d;

        // halo boundaries
        if (threadIdx.x == 0 && i > 0)       s_d[LOCAL_IDX(0, ty, tz)]     = d;  // -x
        if (threadIdx.x == X - 1 && i < XDIM - 1) s_d[LOCAL_IDX(X + 1, ty, tz)] = d;  // +x
        if (threadIdx.y == 0 && j > 0)       s_d[LOCAL_IDX(tx, 0, tz)]     = d;  // -y
        if (threadIdx.y == Y - 1 && j < YDIM - 1) s_d[LOCAL_IDX(tx, Y + 1, tz)] = d;  // +y
        if (threadIdx.z == 0 && k > 0)       s_d[LOCAL_IDX(tx, ty, 0)]     = d;  // -z
        if (threadIdx.z == Z - 1 && k < ZDIM - 1) s_d[LOCAL_IDX(tx, ty, Z + 1)] = d;  // +z

        __syncthreads();

        // --- Laplacian
        if (i < XDIM && j < YDIM && k < ZDIM) {
            float center = s_d[local_index];
            float sum =
                s_d[LOCAL_IDX(tx - 1, ty, tz)] +
                s_d[LOCAL_IDX(tx + 1, ty, tz)] +
                s_d[LOCAL_IDX(tx, ty - 1, tz)] +
                s_d[LOCAL_IDX(tx, ty + 1, tz)] +
                s_d[LOCAL_IDX(tx, ty, tz - 1)] +
                s_d[LOCAL_IDX(tx, ty, tz + 1)];

            q = (sum - 6.0f * center) / (dx * dx);
        }

        __syncthreads();

        // --- Dot products (simplified, single block only)
        float local_rr = r * r;
        float local_dq = d * q;

        atomicAdd(&delta_new, local_rr);
        atomicAdd(&dq, local_dq);

        __syncthreads();

        alpha = delta_new / dq;

        if (i < XDIM && j < YDIM && k < ZDIM) {
            p[index] += alpha * d;
            r -= alpha * q;
        }

        delta_old = delta_new;
        delta_new = 0.0f;

        float r2 = r * r;
        atomicAdd(&delta_new, r2);

        __syncthreads();
        beta = delta_new / delta_old;

        if (i < XDIM && j < YDIM && k < ZDIM) {
            d = r + beta * d;
        }

        if (sqrtf(delta_new) < 1e-5f)
            break;

        __syncthreads();
    }

    if (i < XDIM && j < YDIM && k < ZDIM)
        p[index] = p[index]; // store back result
}

void solvePressureCG(float* d_p, float* d_b, float dx) {
    int N = XDIM * YDIM * ZDIM;
    int maxIters = 100;
    float tolerance = 1e-5f;

    dim3 blockSize(8, 8, 8);  // Matches kernel's tile size
    dim3 gridSize((XDIM + blockSize.x - 1) / blockSize.x, 
                  (YDIM + blockSize.y - 1) / blockSize.y, 
                  (ZDIM + blockSize.z - 1) / blockSize.z);

    // Shared memory size based on block dimensions
    size_t sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * (blockSize.z + 2) * sizeof(float);

    // Kernel arguments
    void* args[] = { &d_p, &d_b, &dx };

    // Launch the kernel
    solve_pressure_cg_flat<<<gridSize, blockSize, sharedMemSize>>>(d_p, d_b, dx);

    // Check for errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
}