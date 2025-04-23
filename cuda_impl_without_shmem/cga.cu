#include "cga.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "parameters.h"

// Kernel to compute q = A * d (Laplacian operation)
__global__ void laplacian_cuda_kernel(float* q, const float* d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < XDIM - 1 && j > 0 && j < YDIM - 1 && k > 0 && k < ZDIM - 1) {
        int idx = i * YDIM * ZDIM + j * ZDIM + k;
        q[idx] = d[(i + 1) * YDIM * ZDIM + j * ZDIM + k] +
                 d[(i - 1) * YDIM * ZDIM + j * ZDIM + k] +
                 d[i * YDIM * ZDIM + (j + 1) * ZDIM + k] +
                 d[i * YDIM * ZDIM + (j - 1) * ZDIM + k] +
                 d[i * YDIM * ZDIM + j * ZDIM + (k + 1)] +
                 d[i * YDIM * ZDIM + j * ZDIM + (k - 1)] -
                 6.0f * d[idx];
    }
}

// Kernel to initialize pressure field
__global__ void initialize_pressure(float* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < XDIM && j < YDIM && k < ZDIM) {
        int idx = i * YDIM * ZDIM + j * ZDIM + k;
        p[idx] = 0.0f;
    }
}

// Kernel to compute the residual r = b - A*p
__global__ void compute_residual_kernel(float* r, const float* b, const float* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < XDIM && j < YDIM && k < ZDIM) {
        int idx = i * YDIM * ZDIM + j * ZDIM + k;
        r[idx] = b[idx];  // Since p is initialized to 0, r = b - A*p simplifies to r = b
    }
}

// Kernel to update pressure and residual
__global__ void update_pressure_and_residual_kernel(
    float* p, float* r, const float* d, const float* q, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < XDIM && j < YDIM && k < ZDIM) {
        int idx = i * YDIM * ZDIM + j * ZDIM + k;
        p[idx] += alpha * d[idx];
        r[idx] -= alpha * q[idx];
    }
}

// Kernel to update the search direction d = r + beta * d
__global__ void update_search_direction_kernel(
    float* d, const float* r, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < XDIM && j < YDIM && k < ZDIM) {
        int idx = i * YDIM * ZDIM + j * ZDIM + k;
        d[idx] = r[idx] + beta * d[idx];
    }
}

// Reduction kernel to compute dot product
__global__ void dot_product_kernel(const float* a, const float* b, float* result, int size) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    while (tid < size) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    // Reduce within the block
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Write block result to global memory
    if (cacheIdx == 0) {
        atomicAdd(result, cache[0]);
    }
}

// Main CG solver function
void solvePressureCG(
    float* d_p, float* d_b) {
    int totalSize = XDIM * YDIM * ZDIM;

    // Allocate device memory
    float *d_r, *d_d, *d_q, *d_delta_new, *d_delta_old, *d_dq;
    // cudaMalloc(&d_p, totalSize * sizeof(float));
    // cudaMalloc(&d_b, totalSize * sizeof(float));
    cudaMalloc(&d_r, totalSize * sizeof(float));
    cudaMalloc(&d_d, totalSize * sizeof(float));
    cudaMalloc(&d_q, totalSize * sizeof(float));
    cudaMalloc(&d_delta_new, sizeof(float));
    cudaMalloc(&d_delta_old, sizeof(float));
    cudaMalloc(&d_dq, sizeof(float));

    // cudaMemcpy(d_b, b, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_p, 0, totalSize * sizeof(float));  // Initialize p to 0

    dim3 blockDim(8, 8, 8);
    dim3 gridDim((XDIM + blockDim.x - 1) / blockDim.x,
                 (YDIM + blockDim.y - 1) / blockDim.y,
                 (ZDIM + blockDim.z - 1) / blockDim.z);

    // Initialize residual r = b
    compute_residual_kernel<<<gridDim, blockDim>>>(d_r, d_b, d_p);
    cudaMemcpy(d_d, d_r, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);

    float delta_new = 0.0f, delta_old = 0.0f;

    // Compute initial delta_new = dot(r, r)
    cudaMemset(d_delta_new, 0, sizeof(float));
    dot_product_kernel<<<gridDim.x, 256>>>(d_r, d_r, d_delta_new, totalSize);
    cudaMemcpy(&delta_new, d_delta_new, sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int iter = 0; iter < cg_max_iterations && delta_new > cg_tolerance * cg_tolerance; ++iter) {
        delta_old = delta_new;
        // Compute q = A * d (Laplacian operation)
        laplacian_cuda_kernel<<<gridDim, blockDim>>>(d_q, d_d);

        // Compute dq = dot(d, q)
        cudaMemset(d_dq, 0, sizeof(float));
        dot_product_kernel<<<gridDim.x, 256>>>(d_d, d_q, d_dq, totalSize);
        float dq;
        cudaMemcpy(&dq, d_dq, sizeof(float), cudaMemcpyDeviceToHost);

        // Compute alpha = delta_new / dq
        float alpha = delta_new / dq;

        // Update p and r
        update_pressure_and_residual_kernel<<<gridDim, blockDim>>>(d_p, d_r, d_d, d_q, alpha);

        // Compute new delta_new = dot(r, r)
        cudaMemset(d_delta_new, 0, sizeof(float));
        dot_product_kernel<<<gridDim.x, 256>>>(d_r, d_r, d_delta_new, totalSize);
        cudaMemcpy(&delta_new, d_delta_new, sizeof(float), cudaMemcpyDeviceToHost);

        // Check for convergence
        if (delta_new < cg_tolerance * cg_tolerance) 
            break;

        float beta = delta_new / delta_old;

        // Update search direction d
        update_search_direction_kernel<<<gridDim, blockDim>>>(d_d, d_r, beta);

        // delta_old = delta_new;
    }

    // Copy result back to host
    // cudaMemcpy(p, d_p, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    // cudaFree(d_p);
    // cudaFree(d_b);
    cudaFree(d_r);
    cudaFree(d_d);
    cudaFree(d_q);
    cudaFree(d_delta_new);
    cudaFree(d_delta_old);
    cudaFree(d_dq);
}