#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include "parameters.h"
#include "Utilities.h"
#include "cga.cuh"
#include "boundary_cond.h"
#include "cuda.h"
#include "velocity_correction.h"

__device__ float trilinear_sample(const float* field, float x, float y, float z) {
    int i = static_cast<int>(floor(x));
    int j = static_cast<int>(floor(y));
    int k = static_cast<int>(floor(z));

    float tx = x - i;
    float ty = y - j;
    float tz = z - k;

    auto clamp = [](int v, int minv, int maxv) { return max(minv, min(v, maxv)); };

    i = clamp(i, 0, XDIM - 2);
    j = clamp(j, 0, YDIM - 2);
    k = clamp(k, 0, ZDIM - 2);

    float c000 = field[i + j * XDIM + k * XDIM * YDIM];
    float c100 = field[(i + 1) + j * XDIM + k * XDIM * YDIM];
    float c010 = field[i + (j + 1) * XDIM + k * XDIM * YDIM];
    float c110 = field[(i + 1) + (j + 1) * XDIM + k * XDIM * YDIM];
    float c001 = field[i + j * XDIM + (k + 1) * XDIM * YDIM];
    float c101 = field[(i + 1) + j * XDIM + (k + 1) * XDIM * YDIM];
    float c011 = field[i + (j + 1) * XDIM + (k + 1) * XDIM * YDIM];
    float c111 = field[(i + 1) + (j + 1) * XDIM + (k + 1) * XDIM * YDIM];

    float c00 = c000 * (1 - tx) + c100 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}

__global__ void clear_kernel(float* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = 0.0f;
    }
}

__global__ void initialize_kernel(float* rho, float* T, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = XDIM * YDIM * ZDIM;
    if (idx >= totalSize) return;
 
    int i = idx % XDIM;
    int j = (idx / XDIM) % YDIM;
    int k = idx / (XDIM * YDIM);
 
    if (k >= ceilf(0.06f * ZDIM) && k < ceilf(0.12f * ZDIM) &&
        i >= XDIM/2 -16 && i < XDIM/2 +16 &&
        j >= YDIM/2 -16 && j < YDIM/2 +16) {
        rho[idx] = 1.0f;
        T[idx] = 300.0f;
        v[idx] = 2.0f;
    }
}

__global__ void applyBoundaryConditions_kernel(float* u, float* v, float* w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < XDIM * YDIM * ZDIM) {
        // Apply boundary conditions (example)
        if (idx < XDIM) u[idx] = 0.0f;
        if (idx < YDIM) v[idx] = 0.0f;
        if (idx < ZDIM) w[idx] = 0.0f;
    }
}

void dumpToCSV(float* rho_d, float* u_d, float* v_d, float* w_d, float* T_d, int totalSize, int t) {
    // Allocate host memory
    float* rho_h = new float[totalSize];
    float* u_h = new float[totalSize];
    float* v_h = new float[totalSize];
    float* w_h = new float[totalSize];
    float* T_h = new float[totalSize];

    // Copy data from device to host
    cudaMemcpy(rho_h, rho_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_h, u_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h, v_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_h, w_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(T_h, T_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Write data to CSV file
    std::ofstream file("output_" + std::to_string(t) + ".csv");
    file << "i,j,k,rho\n"; // Add header for clarity
    for (int idx = 0; idx < totalSize; ++idx) {
        int i = idx % XDIM;
        int j = (idx / XDIM) % YDIM;
        int k = idx / (XDIM * YDIM);

        file << i << "," << j << "," << k << "," 
             << rho_h[idx]<< "\n";
    }
    file.close();

    // Free host memory
    delete[] rho_h;
    delete[] u_h;
    delete[] v_h;
    delete[] w_h;
    delete[] T_h;
}


__global__ void unified_kernel(
    float* u, float* v, float* w,
    float* rho, float* T,
    float* u_star, float* v_star, float* w_star,
    float* rho_next, float* T_next,
    float* divergence,
    float dt, float dx
) {
    // Define shared memory arrays with halo regions
    extern __shared__ float shared_mem[];
    float* s_u = shared_mem; // Shared memory for u
    float* s_v = s_u + blockDim.x * blockDim.y * blockDim.z; // Offset for v
    float* s_w = s_v + blockDim.x * blockDim.y * blockDim.z; // Offset for w
    float* s_rho = s_w + blockDim.x * blockDim.y * blockDim.z; // Offset for rho
    float* s_T = s_rho + blockDim.x * blockDim.y * blockDim.z; // Offset for T
    float* s_u_star = s_T + blockDim.x * blockDim.y * blockDim.z; // Offset for u_star
    float* s_v_star = s_u_star + blockDim.x * blockDim.y * blockDim.z; // Offset for v_star
    float* s_w_star = s_v_star + blockDim.x * blockDim.y * blockDim.z; // Offset for w_star
    float* s_rho_next = s_w_star + blockDim.x * blockDim.y * blockDim.z; // Offset for rho_next
    float* s_T_next = s_rho_next + blockDim.x * blockDim.y * blockDim.z; // Offset for T_next

    // Compute global and local thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int local_i = threadIdx.x;
    int local_j = threadIdx.y;
    int local_k = threadIdx.z;

    int idx = i + j * XDIM + k * XDIM * YDIM; // Global index
    int local_idx = local_i + local_j * blockDim.x + local_k * blockDim.x * blockDim.y; // Local index

    // Load data into shared memory
    if (i < XDIM && j < YDIM && k < ZDIM) {
        s_u[local_idx] = u[idx];
        s_v[local_idx] = v[idx];
        s_w[local_idx] = w[idx];
        s_rho[local_idx] = rho[idx];
        s_T[local_idx] = T[idx];
        s_u_star[local_idx] = 0.0f;
        s_v_star[local_idx] = 0.0f;
        s_w_star[local_idx] = 0.0f;
        s_rho_next[local_idx] = 0.0f;
        s_T_next[local_idx] = 0.0f;
    } else {
        s_u[local_idx] = 0.0f;
        s_v[local_idx] = 0.0f;
        s_w[local_idx] = 0.0f;
        s_rho[local_idx] = 0.0f;
        s_T[local_idx] = 0.0f;
        s_u_star[local_idx] = 0.0f;
        s_v_star[local_idx] = 0.0f;
        s_w_star[local_idx] = 0.0f;
        s_rho_next[local_idx] = 0.0f;
        s_T_next[local_idx] = 0.0f;
    }

    __syncthreads();

    // Step 1: Apply buoyant force
    if (i < XDIM && j < YDIM && k < ZDIM) {
        float buoy_force = alpha * s_rho[local_idx] * beta * (s_T[local_idx] - T_ambient);
        s_v[local_idx] += buoy_force * dt;
    }

    __syncthreads();

    // Step 2: Semi-Lagrangian advection for velocity components
    if (i < XDIM && j < YDIM && k < ZDIM) {
        float x = i - dt * s_u[local_idx];
        float y = j - dt * s_v[local_idx];
        float z = k - dt * s_w[local_idx];

        x = fmaxf(0.0f, fminf((float)(XDIM - 1.001f), x));
        y = fmaxf(0.0f, fminf((float)(YDIM - 1.001f), y));
        z = fmaxf(0.0f, fminf((float)(ZDIM - 1.001f), z));

        s_u_star[local_idx] = trilinear_sample(s_u, x, y, z);
        s_v_star[local_idx] = trilinear_sample(s_v, x, y, z);
        s_w_star[local_idx] = trilinear_sample(s_w, x, y, z);
    }

    __syncthreads();

    // Step 2: Semi-Lagrangian advection for density and temperature
    if (i < XDIM && j < YDIM && k < ZDIM) {
        float x = i - dt * s_u[local_idx];
        float y = j - dt * s_v[local_idx];
        float z = k - dt * s_w[local_idx];

        x = fmaxf(0.0f, fminf((float)(XDIM - 1.001f), x));
        y = fmaxf(0.0f, fminf((float)(YDIM - 1.001f), y));
        z = fmaxf(0.0f, fminf((float)(ZDIM - 1.001f), z));

        s_rho_next[local_idx] = trilinear_sample(s_rho, x, y, z);
        s_T_next[local_idx] = trilinear_sample(s_T, x, y, z);
    }

    __syncthreads();

    // Step 3: Compute divergence
    if (i > 0 && i < XDIM - 1 && j > 0 && j < YDIM - 1 && k > 0 && k < ZDIM - 1) {
        float du_dx = (s_u_star[local_idx + 1] - s_u_star[local_idx - 1]) / (2.0f * dx);
        float dv_dy = (s_v_star[local_idx + blockDim.x] - s_v_star[local_idx - blockDim.x]) / (2.0f * dx);
        float dw_dz = (s_w_star[local_idx + blockDim.x * blockDim.y] - s_w_star[local_idx - blockDim.x * blockDim.y]) / (2.0f * dx);

        divergence[idx] = du_dx + dv_dy + dw_dz;
    } else {
        divergence[idx] = 0.0f;
    }

    __syncthreads();

    // Write back shared memory _star variables to global memory
    if (i < XDIM && j < YDIM && k < ZDIM) {
        u[idx] = s_u_star[local_idx];
        v[idx] = s_v_star[local_idx];
        w[idx] = s_w_star[local_idx];
        rho[idx] = s_rho_next[local_idx];
        T[idx] = s_T_next[local_idx];
    }
}

int main() {
    cudaEvent_t startEvent, stopEvent;
    cudaEvent_t startEvent_totalSteps, stopEvent_totalSteps;
    float elapsedTime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&startEvent_totalSteps);
    cudaEventCreate(&stopEvent_totalSteps);

    size_t totalSize = XDIM * YDIM * ZDIM;

    int totalSteps = 500;

    float *u_d, *v_d, *w_d;             // Velocity components
    float *rho_d, *T_d;                 // Density and Temperature
    float *divergence_d, *p_d;          // Divergence and Pressure

    int threadsPerBlock = 512;
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate on device
    cudaMalloc((void**)&u_d, totalSize * sizeof(float));
    cudaMalloc((void**)&v_d, totalSize * sizeof(float));
    cudaMalloc((void**)&w_d, totalSize * sizeof(float));
    cudaMalloc((void**)&rho_d, totalSize * sizeof(float));
    cudaMalloc((void**)&T_d, totalSize * sizeof(float));
    cudaMalloc((void**)&divergence_d, totalSize * sizeof(float));
    cudaMalloc((void**)&p_d, totalSize * sizeof(float));

    float *u_star_d, *v_star_d, *w_star_d;
    float *rho_next_d, *T_next_d;

    cudaMalloc((void**)&u_star_d, totalSize * sizeof(float));
    cudaMalloc((void**)&v_star_d, totalSize * sizeof(float));
    cudaMalloc((void**)&w_star_d, totalSize * sizeof(float));
    cudaMalloc((void**)&rho_next_d, totalSize * sizeof(float));
    cudaMalloc((void**)&T_next_d, totalSize * sizeof(float));

    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(u_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(v_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(w_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(divergence_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(p_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(rho_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(T_d, totalSize);

    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(u_star_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(v_star_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(w_star_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(T_next_d, totalSize);
    clear_kernel<<<blocksPerGrid, threadsPerBlock>>>(rho_next_d, totalSize);

    initialize_kernel<<<blocksPerGrid, threadsPerBlock>>>(rho_d, T_d, v_d);

    cudaEventRecord(startEvent_totalSteps, 0);

    for (int t = 0; t < totalSteps; ++t) {
        cudaEventRecord(startEvent, 0);

        // Unified kernel for steps up to divergence
        unified_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            u_d, v_d, w_d,
            rho_d, T_d,
            u_star_d, v_star_d, w_star_d,
            rho_next_d, T_next_d,
            divergence_d,
            dt, dx
        ); // Ensure the arguments match the kernel definition
        cudaDeviceSynchronize();

        // std::swap(u_d, u_star_d);
	    // std::swap(v_d, v_star_d);
        // std::swap(w_d, w_star_d);
	    // std::swap(rho_d, rho_next_d);
	    // std::swap(T_d, T_next_d);

        // Step 4: Solve pressure using CGA
        solvePressureCG(p_d, divergence_d); // Ensure this is a device kernel
        cudaDeviceSynchronize();

        // // Step 5: Velocity correction
        velocityCorrection_kernel<<<blocksPerGrid, threadsPerBlock>>>(u_d, v_d, w_d, p_d);
        cudaDeviceSynchronize();

        // // Step 6: Apply boundary conditions
        applyBoundaryConditions_kernel<<<blocksPerGrid, threadsPerBlock>>>(u_d, v_d, w_d);
        cudaDeviceSynchronize();

        if(t%10 ==0) {
            // Dump data to CSV every 10 steps
            dumpToCSV(rho_d, u_d, v_d, w_d, T_d, totalSize, t);
        }

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        std::cout << "CUDA Event compute time for one frame # " << t << " = "
                  << elapsedTime / 1000.0f << " sec\n";
    }

    cudaEventRecord(stopEvent_totalSteps, 0);
    cudaEventSynchronize(stopEvent_totalSteps);

    cudaEventElapsedTime(&elapsedTime, startEvent_totalSteps, stopEvent_totalSteps);
    std::cout << "\n\nCUDA Event compute time for " << totalSteps << " frames = "
              << elapsedTime / 1000.0f << " sec\n";

    // Free device memory
    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(w_d);
    cudaFree(rho_d);
    cudaFree(T_d);
    cudaFree(divergence_d);
    cudaFree(p_d);
    cudaFree(u_star_d);
    cudaFree(v_star_d);
    cudaFree(w_star_d);
    cudaFree(rho_next_d);
    cudaFree(T_next_d);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent_totalSteps);
    cudaEventDestroy(stopEvent_totalSteps);

    cudaDeviceReset();

    return 0;
}
