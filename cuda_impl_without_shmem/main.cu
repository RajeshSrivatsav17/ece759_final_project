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
#include "cga.cuh"
#include "boundary_cond.h"
#include "cuda.h"
#include "velocity_correction.h"
#include "cpu_functions.cpp"

void MatrixMaxDifference(float* compare, const float* A, const float* B,const int n,const char* variable, const char * kernel , int iter)
{
    float result = 0.;
    cudaMemcpy(compare, A, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        result = std::max( result, std::abs( compare[i] - B[i] ) );
    std::cout << "Total discrepancy " << variable << " in kernel " << kernel << " in iter " << iter << ": " << result << std::endl;
}

void compareResults(float * compare, const float* u,const float* v,const float* w,
                    const float* rho, const float* T,
                    const float* uRaw_d, const float* vRaw_d,const float* wRaw_d,
                    const float* rhoRaw_d,const float* TRaw_d, int totalSize, const char * kernel ,int i){
    float result = 0.;
    MatrixMaxDifference(compare, uRaw_d, u, totalSize, "U", kernel, i);
    MatrixMaxDifference(compare, vRaw_d, v, totalSize, "V", kernel, i);
    MatrixMaxDifference(compare, wRaw_d, w, totalSize, "W", kernel, i);
    MatrixMaxDifference(compare, rhoRaw_d, rho, totalSize, "Rho", kernel, i);
    MatrixMaxDifference(compare, TRaw_d, T, totalSize, "T", kernel, i);
}

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

    float* uRaw_star = new float[totalSize];
    float* vRaw_star = new float[totalSize];
    float* wRaw_star = new float[totalSize];
    float* rhoRaw_next = new float[totalSize];
    float* TRaw_next = new float[totalSize];
    int totalSteps = 500;

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

    int threadsPerBlock = 512; // 8*8*8 tile
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(startEvent_totalSteps, 0);

    for (int t = 0; t < totalSteps; ++t) {
        cudaEventRecord(startEvent, 0);
        // Step 1
        //std::cout<<"Calling buoyantforce()\n";
        buoyantforce(rho,T,v);
        buoyantforce_kernel<<<blocksPerGrid, threadsPerBlock>>>(rhoRaw_d,TRaw_d,vRaw_d); //applying buoyant force on pressure and temperature of smoke from vertical velocity compoenent
        cudaDeviceSynchronize();
        compareResults(compare,uRaw,vRaw,wRaw,rhoRaw,TRaw,uRaw_d, vRaw_d,wRaw_d,rhoRaw_d,TRaw_d,totalSize, "buoyantforce_kernel", t);
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

        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(uRaw_star_d, uRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();
	//std::cout<<"Returned from semi_lag_adv() for u\n";
        //std::cout<<"Calling semi_lag_adv() for v\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(vRaw_star_d, vRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();
	//std::cout<<"Returned from semi_lag_adv() for v\n";
        //std::cout<<"Calling semi_lag_adv() for w\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(wRaw_star_d, wRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();        
	//std::cout<<"Returned from semi_lag_adv() for w\n";
        // Step 2: Advect smoke density and temperature
        //std::cout<<"Calling semi_lag_adv() for rho\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(rhoRaw_next_d, rhoRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();        
	//std::cout<<"Returned from semi_lag_adv() for rho\n";
        //std::cout<<"Calling semi_lag_adv() for Temp\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(TRaw_next_d, TRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();        
        compareResults(compare,uRaw,vRaw,wRaw,rhoRaw,TRaw,uRaw_d, vRaw_d,wRaw_d,rhoRaw_d,TRaw_d,totalSize,"semi_lagrangian_advection_kernel",t);
	//std::cout<<"Returned from semi_lag_adv() for Temp\n";

        // Swap buffers for next timestep
        //std::cout<<"Calling swap buffer()\n";
        std::swap(u, u_star);
        std::swap(v, v_star);
        std::swap(w, w_star);
        std::swap(rho, rho_star);
        std::swap(T, T_star);

        std::swap(uRaw_d, uRaw_star_d);
        cudaDeviceSynchronize();
	    std::swap(vRaw_d, vRaw_star_d);
        cudaDeviceSynchronize();
        std::swap(wRaw_d, wRaw_star_d);
        cudaDeviceSynchronize();
	    std::swap(rhoRaw_d, rhoRaw_next_d);
        cudaDeviceSynchronize();
	    std::swap(TRaw_d, TRaw_next_d);
	    cudaDeviceSynchronize();
        //std::cout<<"Finished swapping\n";
        // Step 3: Divergence of velocity
        //std::cout<<"Calling Divergence()\n";
        compareResults(compare,uRaw,vRaw,wRaw,rhoRaw,TRaw,uRaw_d, vRaw_d,wRaw_d,rhoRaw_d,TRaw_d,totalSize,"swap",t);
        computeDivergence(u, v, w, divergence);
        computeDivergence_kernel<<<blocksPerGrid,threadsPerBlock>>>(uRaw_d, vRaw_d, wRaw_d, divergenceRaw_d);
        cudaDeviceSynchronize();
        compareResults(compare,uRaw,vRaw,wRaw,rhoRaw,TRaw,uRaw_d, vRaw_d,wRaw_d,rhoRaw_d,TRaw_d,totalSize,"computeDivergence_kernel",t);
        //std::cout<<"Returned from Divergence()\n";
        // Step 4: Iterative solver
        //std::cout<<"Calling CG()\n";
        solvePressureCG(p, divergence);
        solvePressureCG_kernel(pRaw_d, divergenceRaw_d);
        cudaDeviceSynchronize();
        compareResults(compare,uRaw,vRaw,wRaw,rhoRaw,TRaw,uRaw_d, vRaw_d,wRaw_d,rhoRaw_d,TRaw_d,totalSize,"solvePressureCG", t);
        //std::cout<<"Returned from CG()\n";
        // Step 5: Velocity correction
        //std::cout<<"Velocity correction initiated()\n";
        for (int i = 1; i < XDIM-1; ++i)
        for (int j = 1; j < YDIM-1; ++j)
        for (int k = 1; k < ZDIM-1; ++k){
            u[i][j][k] -= (p[i+1][j][k] - p[i-1][j][k]) / (2.0f * dx);
            v[i][j][k] -= (p[i][j+1][k] - p[i][j-1][k]) / (2.0f * dx);
            w[i][j][k] -= (p[i][j][k+1] - p[i][j][k-1]) / (2.0f * dx);
        } 
        velocityCorrection_kernel<<<blocksPerGrid,threadsPerBlock>>>(uRaw_d, vRaw_d, wRaw_d, pRaw_d);
        cudaDeviceSynchronize();        
        compareResults(compare,uRaw,vRaw,wRaw,rhoRaw,TRaw,uRaw_d, vRaw_d,wRaw_d,rhoRaw_d,TRaw_d,totalSize,"velocityCorrection_kernel", t);
	//std::cout<<"Velocity Correction done\n";
        // Step 6: Boundary Condition
        //std::cout<<"Calling boundary()\n";
        applyBoundaryConditions(u,v,w);
        applyBoundaryConditions_kernel(uRaw_d,vRaw_d,wRaw_d);
        cudaDeviceSynchronize();        
        compareResults(compare,uRaw,vRaw,wRaw,rhoRaw,TRaw,uRaw_d, vRaw_d,wRaw_d,rhoRaw_d,TRaw_d,totalSize,"applyBoundaryConditions", t);
	//std::cout<<"Returned from boundary()\n";
        cudaMemcpy(rhoRaw, rhoRaw_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
        if (t % 10 == 0)
                writetoCSV(rho, "density_frame_" + std::to_string(t) + ".csv","density");
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
