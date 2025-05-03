#include <iostream>
#include <vector>
#include <cmath>
#include "parameters.h"
#include "utilities.cu"
#include "buoyantforce.h"
#include "advect.h"
#include "divergence.h"
#include "cga.cuh"
#include "boundary_cond.h"
#include "velocity_correction.h"

int main() {
    // CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    size_t totalSize = XDIM * YDIM * ZDIM;

    // Host memory allocation
    float *uRaw = new float[totalSize]; // Velocity in x direction
    float *vRaw = new float[totalSize]; // Velocity in y direction
    float *wRaw = new float[totalSize]; // Velocity in z direction
    float *rhoRaw = new float[totalSize]; // Density
    float *TRaw = new float[totalSize]; // Temperature
    float *divergenceRaw = new float[totalSize]; // Divergence
    float *pRaw = new float[totalSize]; // Pressure

    // Device memory allocation
    float *uRaw_d, *vRaw_d, *wRaw_d, *rhoRaw_d, *TRaw_d, *divergenceRaw_d, *pRaw_d;
    cudaMalloc((void**)&uRaw_d, totalSize * sizeof(float));
    cudaMalloc((void**)&vRaw_d, totalSize * sizeof(float));
    cudaMalloc((void**)&wRaw_d, totalSize * sizeof(float));
    cudaMalloc((void**)&rhoRaw_d, totalSize * sizeof(float));
    cudaMalloc((void**)&TRaw_d, totalSize * sizeof(float));
    cudaMalloc((void**)&divergenceRaw_d, totalSize * sizeof(float));
    cudaMalloc((void**)&pRaw_d, totalSize * sizeof(float));

    // Initialize fields
    Clear(uRaw, totalSize);
    Clear(vRaw, totalSize);
    Clear(wRaw, totalSize);
    Clear(rhoRaw, totalSize);
    Clear(TRaw, totalSize);
    InitializeProblem(rhoRaw, TRaw, vRaw);

    // Copy initialized fields to device
    cudaMemcpy(uRaw_d, uRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vRaw_d, vRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wRaw_d, wRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rhoRaw_d, rhoRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(TRaw_d, TRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);

    // Grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((XDIM + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (YDIM + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (ZDIM + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t sharedMemSize = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

    // Main simulation loop
    for (int t = 0; t < TOTAL_STEPS; ++t) {
        cudaEventRecord(startEvent, 0);

        // Step 1: Apply buoyant force
        buoyantforce_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(rhoRaw_d, TRaw_d, vRaw_d);

        // Step 2: Advect velocity
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(uRaw_d, uRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(vRaw_d, vRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(wRaw_d, wRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);

        // Step 3: Advect smoke density and temperature
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(rhoRaw_d, rhoRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(TRaw_d, TRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);

        // Step 4: Compute divergence
        computeDivergence_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(uRaw_d, vRaw_d, wRaw_d, divergenceRaw_d);

        // Step 5: Solve pressure using Conjugate Gradient
        cg_pressure_solver_flat<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(pRaw_d, divergenceRaw_d, dx, totalSize, cg_max_iterations, cg_tolerance);

        // Step 6: Correct velocity
        velocityCorrection_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(uRaw_d, vRaw_d, wRaw_d, pRaw_d);

        // Step 7: Apply boundary conditions
        applyBoundaryConditionsX<<<(YDIM * ZDIM + 255) / 256, 256>>>(uRaw_d);
        applyBoundaryConditionsY<<<(XDIM * ZDIM + 255) / 256, 256>>>(vRaw_d);
        applyBoundaryConditionsZ<<<(XDIM * YDIM + 255) / 256, 256>>>(wRaw_d);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);

        // Optional: Output results every 10 steps
        if (t % 10 == 0 && RESULT) {
            cudaMemcpy(rhoRaw, rhoRaw_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
            writetoCSV(rhoRaw, "density_frame_" + std::to_string(t) + ".csv", "density");
        }
    }

    // Cleanup
    delete[] uRaw;
    delete[] vRaw;
    delete[] wRaw;
    delete[] rhoRaw;
    delete[] TRaw;
    delete[] divergenceRaw;
    delete[] pRaw;

    cudaFree(uRaw_d);
    cudaFree(vRaw_d);
    cudaFree(wRaw_d);
    cudaFree(rhoRaw_d);
    cudaFree(TRaw_d);
    cudaFree(divergenceRaw_d);
    cudaFree(pRaw_d);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaDeviceReset();
    return 0;
}