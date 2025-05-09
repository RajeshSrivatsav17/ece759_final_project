#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include "parameters.h"
#include "Utilities.h"
#include "buoyantforce.h"
#include "advect.h"
#include "divergence.h"
#include "cga.cuh"
#include "boundary_cond.h"
#include "cuda.h"
#include "velocity_correction.h"

int main(){
    
    cudaEvent_t startEvent, stopEvent;
    cudaEvent_t startEvent_totalSteps, stopEvent_totalSteps;
    float elapsedTime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&startEvent_totalSteps);
    cudaEventCreate(&stopEvent_totalSteps);

    size_t totalSize = XDIM * YDIM * ZDIM;

    using array_t = float (&) [XDIM][YDIM][ZDIM];
    float *uRaw = new float [totalSize]; //Velocity in x direction
    float *vRaw = new float [totalSize]; //Velocity in y direction
    float *wRaw = new float [totalSize]; //Velocity in z direction
    float *rhoRaw = new float [totalSize]; //Density
    float *TRaw = new float [totalSize]; //Temperature
    float *divergenceRaw = new float [totalSize]; //Divergence
    float *pRaw = new float [totalSize]; //Pressure
    float *compare = new float [totalSize];
    float *rhoRaw_for_GPU = new float [totalSize];

    float* uRaw_star = new float[totalSize];
    float* vRaw_star = new float[totalSize];
    float* wRaw_star = new float[totalSize];
    float* rhoRaw_next = new float[totalSize];
    float* TRaw_next = new float[totalSize];
    int totalSteps = TOTAL_STEPS;

    float *uRaw_d, *vRaw_d, *wRaw_d;             // Velocity components
    float *rhoRaw_d, *TRaw_d;                    // Density and Temperature
    float *divergenceRaw_d, *pRaw_d;             // Divergence and Pressure
    
    // Allocate on device
    cudaMalloc((void**)&uRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&vRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&wRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&rhoRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&TRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&divergenceRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&pRaw_d, totalSize*sizeof(float));

    float *uRaw_star_d, *vRaw_star_d, *wRaw_star_d;
    float *rhoRaw_next_d, *TRaw_next_d;
    
    cudaMalloc((void**)&uRaw_star_d, totalSize * sizeof(float));
    cudaMalloc((void**)&vRaw_star_d, totalSize * sizeof(float));
    cudaMalloc((void**)&wRaw_star_d, totalSize * sizeof(float));
    cudaMalloc((void**)&rhoRaw_next_d, totalSize * sizeof(float));
    cudaMalloc((void**)&TRaw_next_d, totalSize * sizeof(float));
    
    //Velocity//
    array_t u = reinterpret_cast<array_t>(*uRaw); //Velocity in x direction
    array_t v = reinterpret_cast<array_t>(*vRaw); //Velocity in y direction
    array_t w = reinterpret_cast<array_t>(*wRaw); //Velocity in z direction
    //Density//
    array_t rho = reinterpret_cast<array_t>(*rhoRaw);
    //Temperature//
    array_t T = reinterpret_cast<array_t>(*TRaw);
    //Divergence// 
    array_t divergence = reinterpret_cast<array_t>(*divergenceRaw);
    //Pressure//
    array_t p = reinterpret_cast<array_t>(*pRaw);

    array_t rho_for_GPU = reinterpret_cast<array_t>(*rhoRaw_for_GPU);
    //Advection Velocity//
    array_t u_star = reinterpret_cast<array_t>(*uRaw_star); //Velocity in x direction
    array_t v_star = reinterpret_cast<array_t>(*vRaw_star); //Velocity in y direction
    array_t w_star = reinterpret_cast<array_t>(*wRaw_star); //Velocity in z direction
    //Advection Density//
    array_t rho_star = reinterpret_cast<array_t>(*rhoRaw_next);
    //Advection Temperature//
    array_t T_star = reinterpret_cast<array_t>(*TRaw_next);

    Clear(u);Clear(w);Clear(divergence);Clear(p);
    Clear(u_star);Clear(v_star);Clear(w_star);
    
    InitializeProblem(rho,T,v);

    cudaMemcpy(uRaw_d, uRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vRaw_d, vRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wRaw_d, wRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rhoRaw_d, rhoRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(TRaw_d, TRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(divergenceRaw_d, divergenceRaw , totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pRaw_d, pRaw, totalSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(uRaw_star_d, uRaw_star, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vRaw_star_d, vRaw_star, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wRaw_star_d, wRaw_star, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rhoRaw_next_d, rhoRaw_next, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(TRaw_next_d, TRaw_next, totalSize * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE; // 8*8*8 tile
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;    

    cudaEventRecord(startEvent_totalSteps, 0);

    for (int t = 0; t < totalSteps; ++t) {
        cudaEventRecord(startEvent, 0);

        // Step 1
        buoyantforce_kernel<<<blocksPerGrid, threadsPerBlock>>>(rhoRaw_d,TRaw_d,vRaw_d); //applying buoyant force on pressure and temperature of smoke from vertical velocity compoenent
        
        // Step 2: Advect velocity (u*, v*, w*)
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(uRaw_star_d, uRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(vRaw_star_d, vRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(wRaw_star_d, wRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        // Step 2: Advect smoke density and temperature
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(rhoRaw_next_d, rhoRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(TRaw_next_d, TRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        
        // Swap buffers for next timestep
        std::swap(uRaw_d, uRaw_star_d);
	    std::swap(vRaw_d, vRaw_star_d);
        std::swap(wRaw_d, wRaw_star_d);
        std::swap(rhoRaw_d, rhoRaw_next_d);
        std::swap(TRaw_d, TRaw_next_d);
	           
        // Step 3: Divergence of velocity
        computeDivergence_kernel<<<blocksPerGrid,threadsPerBlock>>>(uRaw_d, vRaw_d, wRaw_d, divergenceRaw_d);
        
        // Step 4: Iterative solver
        
        solvePressureCG(pRaw_d, divergenceRaw_d);
        
        // Step 5: Velocity correction
        velocityCorrection_kernel<<<blocksPerGrid,threadsPerBlock>>>(uRaw_d, vRaw_d, wRaw_d, pRaw_d);
    
        applyBoundaryConditions(uRaw_d,vRaw_d,wRaw_d);
    
        cudaMemcpy(rhoRaw_for_GPU, rhoRaw_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	
        if ((t % 10 == 0) && RESULT){
                writetoCSV(rho_for_GPU, "density_frame_" + std::to_string(t) + ".csv","density");
        }
	    cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        std::cout << "CUDA Event compute time for one frame #  " << t << " = " 
                  << elapsedTime / 1000.0f << " sec\n";        
    }

    cudaEventRecord(stopEvent_totalSteps,0);
    cudaEventSynchronize(stopEvent_totalSteps);
    cudaEventElapsedTime(&elapsedTime, startEvent_totalSteps, stopEvent_totalSteps);
    std::cout << "\n\nCUDA Event compute time for " << totalSteps << " frames  " << elapsedTime / 1000.0f << " sec\n";  
    
    delete[] uRaw;
    delete[] vRaw;
    delete[] wRaw;
    delete[] rhoRaw;
    delete[] TRaw;
    delete[] divergenceRaw;
    delete[] pRaw;

    // Free device memory
    cudaFree(uRaw_d);
    cudaFree(vRaw_d);
    cudaFree(wRaw_d);
    cudaFree(rhoRaw_d);
    cudaFree(TRaw_d);
    cudaFree(divergenceRaw_d);
    cudaFree(pRaw_d);
    cudaFree(uRaw_star_d);
    cudaFree(vRaw_star_d);
    cudaFree(wRaw_star_d);
    cudaFree(rhoRaw_next_d);
    cudaFree(TRaw_next_d);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent_totalSteps);
    cudaEventDestroy(stopEvent_totalSteps);

    cudaDeviceReset(); 

    return 0;
}
