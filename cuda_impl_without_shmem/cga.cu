#include "cga.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "parameters.h"

#define IDX(i, j, k) ((i) * YDIM * ZDIM + (j) * ZDIM + (k))

__device__ float laplacian(const float* p, int i, int j, int k, float dx) {
    float center = p[IDX(i, j, k)];
    float sum = 0.0f;
    if (i > 0)        sum += p[IDX(i - 1, j, k)];
    if (i < XDIM - 1) sum += p[IDX(i + 1, j, k)];
    if (j > 0)        sum += p[IDX(i, j - 1, k)];
    if (j < YDIM - 1) sum += p[IDX(i, j + 1, k)];
    if (k > 0)        sum += p[IDX(i, j, k - 1)];
    if (k < ZDIM - 1) sum += p[IDX(i, j, k + 1)];
    return (sum - 6.0f * center) / (dx * dx);
}

__global__ void initialize_kernel(float* p, float* r, float* d, const float* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = XDIM * YDIM * ZDIM;
    if (idx < N) {
        p[idx] = 0.0f;
        r[idx] = b[idx];
        d[idx] = r[idx];
    }
}

__global__ void dot_product_kernel(const float* a, const float* b, float* result) {
    __shared__ float cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int N = XDIM * YDIM * ZDIM;
    cache[tid] = (idx < N) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, cache[0]);
}

__global__ void compute_laplacian_kernel(float* q, const float* d, float dx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = XDIM * YDIM * ZDIM;
    if (idx < N) {
        int i = idx / (YDIM * ZDIM);
        int j = (idx / ZDIM) % YDIM;
        int k = idx % ZDIM;
        q[idx] = laplacian(d, i, j, k, dx);
    }
}

__global__ void update_p_r_kernel(float* p, float* r, const float* d, const float* q, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = XDIM * YDIM * ZDIM;
    if (idx < N) {
        p[idx] += alpha * d[idx];
        r[idx] -= alpha * q[idx];
    }
}

__global__ void update_d_kernel(float* d, const float* r, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = XDIM * YDIM * ZDIM;
    if (idx < N) {
        d[idx] = r[idx] + beta * d[idx];
    }
}

void solvePressureCG(float* d_p, float* d_b) {
    float *d_r, *d_d, *d_q;
    float *d_dot_new, *d_dq;
    float alpha, beta, delta_new, delta_old;
    int maxIterations = 100;
    float tolerance = 1e-5f;
    int N = XDIM * YDIM * ZDIM;
    cudaMalloc(&d_r, N * sizeof(float));
    cudaMalloc(&d_d, N * sizeof(float));
    cudaMalloc(&d_q, N * sizeof(float));
    cudaMalloc(&d_dot_new, sizeof(float));
    cudaMalloc(&d_dq, sizeof(float));

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    initialize_kernel<<<gridSize, blockSize>>>(d_p, d_r, d_d, d_b);
    cudaMemset(d_dot_new, 0, sizeof(float));
    dot_product_kernel<<<gridSize, blockSize>>>(d_r, d_r, d_dot_new);
    cudaMemcpy(&delta_new, d_dot_new, sizeof(float), cudaMemcpyDeviceToHost);

    for (int iter = 0; iter < maxIterations && delta_new > tolerance * tolerance; ++iter) {
        compute_laplacian_kernel<<<gridSize, blockSize>>>(d_q, d_d, dx);
        cudaMemset(d_dq, 0, sizeof(float));
        dot_product_kernel<<<gridSize, blockSize>>>(d_d, d_q, d_dq);
        float dq;
        cudaMemcpy(&dq, d_dq, sizeof(float), cudaMemcpyDeviceToHost);
        alpha = delta_new / dq;

        update_p_r_kernel<<<gridSize, blockSize>>>(d_p, d_r, d_d, d_q, alpha);

        delta_old = delta_new;
        cudaMemset(d_dot_new, 0, sizeof(float));
        dot_product_kernel<<<gridSize, blockSize>>>(d_r, d_r, d_dot_new);
        cudaMemcpy(&delta_new, d_dot_new, sizeof(float), cudaMemcpyDeviceToHost);

        beta = delta_new / delta_old;
        update_d_kernel<<<gridSize, blockSize>>>(d_d, d_r, beta);
    }

    cudaFree(d_r);
    cudaFree(d_d);
    cudaFree(d_q);
    cudaFree(d_dot_new);
    cudaFree(d_dq);
}

