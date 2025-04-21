#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "parameters.h"
#include "Utilities.h"
#include "buoyantforce.h"
#include "advect.h"
#include "divergence.h"
#include "cga.h"
#include "boundary_cond.h"
#include "cuda.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using elapsed_time_t = std::chrono::duration<double, std::milli>;

int main(){
    
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    size_t totalSize = XDIM * YDIM * ZDIM;

    using array_t = float (&) [XDIM][YDIM][ZDIM];
    float *uRaw = new float [totalSize]; //Velocity in x direction
    float *vRaw = new float [totalSize]; //Velocity in y direction
    float *wRaw = new float [totalSize]; //Velocity in z direction
    float *rhoRaw = new float [totalSize]; //Density
    float *TRaw = new float [totalSize]; //Temperature
    float *divergenceRaw = new float [totalSize]; //Divergence
    float *PRaw = new float [totalSize]; //Pressure

    float* uRaw_star = new float[totalSize];
    float* vRaw_star = new float[totalSize];
    float* wRaw_star = new float[totalSize];
    float* rhoRaw_next = new float[totalSize];
    float* TRaw_next = new float[totalSize];
    int totalSteps = 500;

    float *uRaw_d, *vRaw_d, *wRaw_d;             // Velocity components
    float *rhoRaw_d, *TRaw_d;                    // Density and Temperature
    float *divergenceRaw_d, *PRaw_d;             // Divergence and Pressure
    
    // Allocate on device
    cudaMalloc((void**)&uRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&vRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&wRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&rhoRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&TRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&divergenceRaw_d, totalSize*sizeof(float));
    cudaMalloc((void**)&PRaw_d, totalSize*sizeof(float));
    
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
    array_t p = reinterpret_cast<array_t>(*PRaw);

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
    
    start = high_resolution_clock::now();
    InitializeProblem(rho,T,v);

    cudaMemcpy(uRaw_d, uRaw, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(vRaw_d, vRaw, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(wRaw_d, wRaw, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(rhoRaw_d, rhoRaw, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(TRaw_d, TRaw, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(divergenceRaw_d, divergenceRaw, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(PRaw_d, PRaw, totalSize, cudaMemcpyHostToDevice);

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout<<"Time for initializing Density Temperature and Vertical Velocity: "<<duration_sec.count()<<" ms\n";
    
    duration_sec = elapsed_time_t::zero();
    start = high_resolution_clock::now();

    int totalSize = XDIM * YDIM * ZDIM;
    int threadsPerBlock = 512; // 8*8*8 tile
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int t = 0; t < totalSteps; ++t) {
        // Step 1
        //std::cout<<"Calling buoyantforce()\n";
        buoyantforce_kernel<<<blocksPerGrid, threadsPerBlock>>>(rho,T,v); //applying buoyant force on pressure and temperature of smoke from vertical velocity compoenent
        cudaDeviceSynchronize();
        //std::cout<<"Returned from buoyantforce()\n";
        // Step 2: Advect velocity (u*, v*, w*)
        //std::cout<<"Calling semi_lag_adv() for u\n";
        semi_lagrangian_advection(u_star, u, u, v, w, dt);
        //std::cout<<"Returned from semi_lag_adv() for u\n";
        //std::cout<<"Calling semi_lag_adv() for v\n";
        semi_lagrangian_advection(v_star, v, u, v, w, dt);
        //std::cout<<"Returned from semi_lag_adv() for v\n";
        //std::cout<<"Calling semi_lag_adv() for w\n";
        semi_lagrangian_advection(w_star, w, u, v, w, dt);
        //std::cout<<"Returned from semi_lag_adv() for w\n";
        // Step 2: Advect smoke density and temperature
        //std::cout<<"Calling semi_lag_adv() for rho\n";
        semi_lagrangian_advection(rho_star, rho, u, v, w, dt);
        //std::cout<<"Returned from semi_lag_adv() for rho\n";
        //std::cout<<"Calling semi_lag_adv() for Temp\n";
        semi_lagrangian_advection(T_star, T, u, v, w, dt);
        //std::cout<<"Returned from semi_lag_adv() for Temp\n";

        // Swap buffers for next timestep
        //std::cout<<"Calling swap buffer()\n";
        std::swap(u, u_star);
        std::swap(v, v_star);
        std::swap(w, w_star);
        std::swap(rho, rho_star);
        std::swap(T, T_star);
        //std::cout<<"Finished swapping\n";
        // Step 3: Divergence of velocity
        //std::cout<<"Calling Divergence()\n";
        computeDivergence(u, v, w, divergence);
        //std::cout<<"Returned from Divergence()\n";
        // Step 4: Iterative solver
        //std::cout<<"Calling CG()\n";
        solvePressureCG(p, divergence);
        //std::cout<<"Returned from CG()\n";
        // Step 5: Velocity correction
        //std::cout<<"Velocity correction initiated()\n";
        #pragma omp parallel for
        for (int i = 1; i < XDIM-1; ++i)
        for (int j = 1; j < YDIM-1; ++j)
        for (int k = 1; k < ZDIM-1; ++k){
            u[i][j][k] -= (p[i+1][j][k] - p[i-1][j][k]) / (2.0f * dx);
            v[i][j][k] -= (p[i][j+1][k] - p[i][j-1][k]) / (2.0f * dx);
            w[i][j][k] -= (p[i][j][k+1] - p[i][j][k-1]) / (2.0f * dx);
        }
        //std::cout<<"Velocity Correction done\n";
        // Step 6: Boundary Condition
        //std::cout<<"Calling boundary()\n";
        applyBoundaryConditions(u,v,w);
        //std::cout<<"Returned from boundary()\n";
        //if (t % 10 == 0)
        //    writetoCSV(rho, "density_frame_" + std::to_string(t) + ".csv","density");
    }
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout<<"Time for rendering "<<totalSteps<<" frames: "<<duration_sec.count()/1000<<" sec\n";
    delete[] uRaw;
    delete[] vRaw;
    delete[] wRaw;
    delete[] rhoRaw;
    delete[] TRaw;
    delete[] divergenceRaw;
    delete[] PRaw;

    return 0;
}