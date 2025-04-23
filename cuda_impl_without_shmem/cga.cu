#include "cga.cuh"
#include "utilities.cuh"
#include <cuda_runtime.h>
#include <iostream>


// Kernel to initialize pressure field
__global__ void initialize_pressure(float* p, int xdim, int ydim, int zdim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < xdim && j < ydim && k < zdim) {
        int idx = i * ydim * zdim + j * zdim + k;
        p[idx] = 0.0f;
    }
}

// Kernel to compute the residual r = b - A*p
__global__ void compute_residual(float* r, const float* b, const float* p, int xdim, int ydim, int zdim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < xdim && j < ydim && k < zdim) {
        int idx = i * ydim * zdim + j * zdim + k;
        r[idx] = b[idx];  // Since p is initialized to 0, r = b - A*p simplifies to r = b
    }
}

// Kernel to update pressure and residual
__global__ void update_pressure_and_residual(
    float* p, float* r, const float* d, const float* q, float alpha, int xdim, int ydim, int zdim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < xdim && j < ydim && k < zdim) {
        int idx = i * ydim * zdim + j * zdim + k;
        p[idx] += alpha * d[idx];
        r[idx] -= alpha * q[idx];
    }
}

// Kernel to update the search direction d = r + beta * d
__global__ void update_search_direction(
    float* d, const float* r, float beta, int xdim, int ydim, int zdim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < xdim && j < ydim && k < zdim) {
        int idx = i * ydim * zdim + j * zdim + k;
        d[idx] = r[idx] + beta * d[idx];
    }
}

// Reduction kernel to compute dot product
__global__ void dot_product(const float* a, const float* b, float* result, int size) {
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
    int totalSize = xdim * ydim * zdim;

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
    dim3 gridDim((xdim + blockDim.x - 1) / blockDim.x,
                 (ydim + blockDim.y - 1) / blockDim.y,
                 (zdim + blockDim.z - 1) / blockDim.z);

    // Initialize residual r = b
    compute_residual<<<gridDim, blockDim>>>(d_r, d_b, d_p, xdim, ydim, zdim);
    cudaMemcpy(d_d, d_r, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);

    float delta_new = 0.0f, delta_old = 0.0f;

    // Compute initial delta_new = dot(r, r)
    cudaMemset(d_delta_new, 0, sizeof(float));
    dot_product<<<gridDim.x, 256>>>(d_r, d_r, d_delta_new, totalSize);
    cudaMemcpy(&delta_new, d_delta_new, sizeof(float), cudaMemcpyDeviceToHost);

    for (int iter = 0; iter < maxIterations && delta_new > tolerance * tolerance; ++iter) {
        // Compute q = A * d (Laplacian operation)
        laplacian_cuda<<<gridDim, blockDim>>>(d_q, d_d, xdim, ydim, zdim);

        // Compute dq = dot(d, q)
        cudaMemset(d_dq, 0, sizeof(float));
        dot_product<<<gridDim.x, 256>>>(d_d, d_q, d_dq, totalSize);
        float dq;
        cudaMemcpy(&dq, d_dq, sizeof(float), cudaMemcpyDeviceToHost);

        // Compute alpha = delta_new / dq
        float alpha = delta_new / dq;

        // Update p and r
        update_pressure_and_residual<<<gridDim, blockDim>>>(d_p, d_r, d_d, d_q, alpha, xdim, ydim, zdim);

        // Compute new delta_new = dot(r, r)
        cudaMemset(d_delta_new, 0, sizeof(float));
        dot_product<<<gridDim.x, 256>>>(d_r, d_r, d_delta_new, totalSize);
        cudaMemcpy(&delta_new, d_delta_new, sizeof(float), cudaMemcpyDeviceToHost);

        // Check for convergence
        if (delta_new < tolerance * tolerance) 
            break;

        float beta = delta_new / delta_old;

        // Update search direction d
        update_search_direction<<<gridDim, blockDim>>>(d_d, d_r, beta, xdim, ydim, zdim);

        delta_old = delta_new;
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