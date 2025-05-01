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

void MatrixMaxDifference(float* compare, const float* A, const float* B,const int n,const char* variable, const char * kernel , int iter)
{
    float result = 0.;
    cudaMemcpy(compare, A, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n*n; i++)
        result = std::max( result, std::abs( compare[i] - B[i] ) );
    std::cout << "Total discrepancy " << variable << " in kernel " << kernel << " in iter " << iter << ": " << result << std::endl;
}

void buoyantforce(float (&rho) [XDIM][YDIM][ZDIM],float (&T) [XDIM][YDIM][ZDIM],float (&v) [XDIM][YDIM][ZDIM]){
    for(int i = 0; i < XDIM; i++){
        for(int j = 0; j < YDIM; j++){
            for(int k = 0; k < ZDIM; k++){
                double buoy_force = alpha * rho[i][j][k] * beta * (T[i][j][k] - T_ambient);
                v[i][j][k] += buoy_force*dt;
            }
        }
    }
}

float trilinear_sample(const float (&field) [XDIM][YDIM][ZDIM], float x, float y, float z) {
    int i = static_cast<int>(floor(x));
    int j = static_cast<int>(floor(y));
    int k = static_cast<int>(floor(z));

    float tx = x - i;
    float ty = y - j;
    float tz = z - k;

    auto clamp = [](int v, int minv, int maxv) { return std::max(minv, std::min(v, maxv)); };

    i = clamp(i, 0, XDIM - 2);
    j = clamp(j, 0, YDIM - 2);
    k = clamp(k, 0, ZDIM - 2);

    float c000 = field[i][j][k];
    float c100 = field[i+1][j][k];
    float c010 = field[i][j+1][k];
    float c110 = field[i+1][j+1][k];
    float c001 = field[i][j][k+1];
    float c101 = field[i+1][j][k+1];
    float c011 = field[i][j+1][k+1];
    float c111 = field[i+1][j+1][k+1];

    float c00 = c000 * (1 - tx) + c100 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}

void semi_lagrangian_advection(float (&dst) [XDIM][YDIM][ZDIM], const float (&src) [XDIM][YDIM][ZDIM], const float (&u) [XDIM][YDIM][ZDIM], const float (&v) [XDIM][YDIM][ZDIM], const float(&w) [XDIM][YDIM][ZDIM], float dt) {
    for (int k = 0; k < ZDIM; ++k) {
        for (int j = 0; j < YDIM; ++j) {
            for (int i = 0; i < XDIM; ++i) {

                float x = i - dt * u[i][j][k];
                float y = j - dt * v[i][j][k];
                float z = k - dt * w[i][j][k];

                x = std::max(0.0f, std::min((float)(XDIM - 1.001f), x));
                y = std::max(0.0f, std::min((float)(YDIM - 1.001f), y));
                z = std::max(0.0f, std::min((float)(ZDIM - 1.001f), z));

                dst[i][j][k] = trilinear_sample(src, x, y, z );
            }
        }
    }
}

void computeDivergence(
    float (&u)[XDIM][YDIM][ZDIM],
    float (&v)[XDIM][YDIM][ZDIM],
    float (&w)[XDIM][YDIM][ZDIM],
    float (&divergence)[XDIM][YDIM][ZDIM]
)
{
    for (int i = 1; i < XDIM - 1; i++) {
        for (int j = 1; j < YDIM - 1; j++) {
            for (int k = 1; k < ZDIM - 1; k++) {
                float du_dx = (u[i + 1][j][k] - u[i - 1][j][k]) / (2.0f * dx);
                float dv_dy = (v[i][j + 1][k] - v[i][j - 1][k]) / (2.0f * dx);
                float dw_dz = (w[i][j][k + 1] - w[i][j][k - 1]) / (2.0f * dx);

                divergence[i][j][k] = du_dx + dv_dy + dw_dz;
            }
        }
    }
}

void solvePressureCG_CPU(
    float (&p)[XDIM][YDIM][ZDIM],
    float (&b)[XDIM][YDIM][ZDIM]
) {
    float* rRaw = new float[XDIM * YDIM * ZDIM]();
    float* dRaw = new float[XDIM * YDIM * ZDIM]();
    float* qRaw = new float[XDIM * YDIM * ZDIM]();

    using array_t = float (&)[XDIM][YDIM][ZDIM];
    array_t r = reinterpret_cast<array_t>(*rRaw);
    array_t d = reinterpret_cast<array_t>(*dRaw);
    array_t q = reinterpret_cast<array_t>(*qRaw);

    int maxIterations = 100;
    float tolerance = 1e-5f;
    // Initial guess: p = 0
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k)
        p[i][j][k] = 0.0f;

    // r = b - A*p (but p = 0, so r = b)
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k) {
        r[i][j][k] = b[i][j][k];
        d[i][j][k] = r[i][j][k];
    }
    float delta_new = 0.0f;
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k)
        delta_new += r[i][j][k] * r[i][j][k];

    for (int iter = 0; iter < maxIterations && delta_new > tolerance * tolerance; ++iter) {
        // q = A * d
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            q[i][j][k] = laplacian(d, i, j, k);

        // alpha = delta_new / dot(d, q)
        float dq = 0.0f;
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            dq += d[i][j][k] * q[i][j][k];

        float alpha = delta_new / dq;

        // p = p + alpha * d
        // r = r - alpha * q
        float delta_old = delta_new;
        delta_new = 0.0f;
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k) {
            p[i][j][k] += alpha * d[i][j][k];
            r[i][j][k] -= alpha * q[i][j][k];
            delta_new += r[i][j][k] * r[i][j][k];
        }

        // beta = delta_new / delta_old
        float beta = delta_new / delta_old;

        // d = r + beta * d
        for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
        for (int k = 0; k < ZDIM; ++k)
            d[i][j][k] = r[i][j][k] + beta * d[i][j][k];
    }
    delete[] rRaw;
    delete[] dRaw;
    delete[] qRaw;

}
void velocityCorrection(float (&u)[XDIM][YDIM][ZDIM], float (&v)[XDIM][YDIM][ZDIM], float (&w)[XDIM][YDIM][ZDIM], float (&p)[XDIM][YDIM][ZDIM])
{
    for (int i = 1; i < XDIM-1; ++i)
    for (int j = 1; j < YDIM-1; ++j)
    for (int k = 1; k < ZDIM-1; ++k){
        u[i][j][k] -= (p[i+1][j][k] - p[i-1][j][k]) / (2.0f * dx);
        v[i][j][k] -= (p[i][j+1][k] - p[i][j-1][k]) / (2.0f * dx);
        w[i][j][k] -= (p[i][j][k+1] - p[i][j][k-1]) / (2.0f * dx);
    }
}
void applyBoundaryConditions_CPU(float (&u)[XDIM][YDIM][ZDIM],
                              float (&v)[XDIM][YDIM][ZDIM],
                              float (&w)[XDIM][YDIM][ZDIM]) {
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k) {
        u[0][j][k] = 0; u[XDIM-1][j][k] = 0; // Applies zero at X boundary velocity
    }
    for (int i = 0; i < XDIM; ++i)
    for (int k = 0; k < ZDIM; ++k) {
        v[i][0][k] = 0; v[i][YDIM-1][k] = 0; // Applies zero at Y boundary velocity
    }
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j) {
        w[i][j][0] = 0; w[i][j][ZDIM-1] = 0; // Applies zero at Z boundary velocity
    }
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
    float *rhoRaw_for_GPU = new float [totalSize];

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

    int threadsPerBlock = 64; // 8*8*8 tile
    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;    

    cudaEventRecord(startEvent_totalSteps, 0);

    for (int t = 0; t < totalSteps; ++t) {
        cudaEventRecord(startEvent, 0);
        // Step 1
        //std::cout<<"Calling buoyantforce()\n";
        buoyantforce_kernel<<<blocksPerGrid, threadsPerBlock>>>(rhoRaw_d,TRaw_d,vRaw_d); //applying buoyant force on pressure and temperature of smoke from vertical velocity compoenent
	cudaDeviceSynchronize();
	buoyantforce(rho,T,v);
	MatrixMaxDifference(compare, vRaw_d, vRaw, totalSize, "V*", "buoyantforce_kernel" , t);
        //std::cout<<"Returned from buoyantforce()\n";
        // Step 2: Advect velocity (u*, v*, w*)
        //std::cout<<"Calling semi_lag_adv() for u\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(uRaw_star_d, uRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();
	semi_lagrangian_advection(u_star, u, u, v, w, dt);
	MatrixMaxDifference(compare, uRaw_star_d, uRaw_star, totalSize, "U*", "u advect_kernel" , t);
	//std::cout<<"Returned from semi_lag_adv() for u\n";
        //std::cout<<"Calling semi_lag_adv() for v\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(vRaw_star_d, vRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();
	semi_lagrangian_advection(v_star, v, u, v, w, dt);
	MatrixMaxDifference(compare,vRaw_star_d,vRaw_star, totalSize, "V*", "v advect_kernel", t);
	//std::cout<<"Returned from semi_lag_adv() for v\n";
        //std::cout<<"Calling semi_lag_adv() for w\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(wRaw_star_d, wRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();
	semi_lagrangian_advection(w_star, w, u, v, w, dt);
	MatrixMaxDifference(compare, wRaw_star_d, wRaw_star, totalSize, "W*", "w advect_kernel", t);
	//std::cout<<"Returned from semi_lag_adv() for w\n";
        // Step 2: Advect smoke density and temperature
        //std::cout<<"Calling semi_lag_adv() for rho\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(rhoRaw_next_d, rhoRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();
	semi_lagrangian_advection(rho_star, rho, u, v, w, dt);
	MatrixMaxDifference(compare, rhoRaw_next_d, rhoRaw_next, totalSize, "rho_next", "rho advect_kernel", t);
	//std::cout<<"Returned from semi_lag_adv() for rho\n";
        //std::cout<<"Calling semi_lag_adv() for Temp\n";
        semi_lagrangian_advection_kernel<<<blocksPerGrid, threadsPerBlock>>>(TRaw_next_d, TRaw_d, uRaw_d, vRaw_d, wRaw_d, dt);
        cudaDeviceSynchronize();
	semi_lagrangian_advection(T_star, T, u, v, w, dt);
	MatrixMaxDifference(compare, TRaw_next_d, TRaw_next, totalSize, "T_next", "T advect_kernel", t);
	//std::cout<<"Returned from semi_lag_adv() for Temp\n";

        // Swap buffers for next timestep
        //std::cout<<"Calling swap buffer()\n";
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
	std::swap(u, u_star);
    	std::swap(v, v_star);
    	std::swap(w, w_star);
    	std::swap(rho, rho_star);
    	std::swap(T, T_star);
        //std::cout<<"Finished swapping\n";
        MatrixMaxDifference(compare, uRaw_d, uRaw, totalSize, "uRaw", "swapped u", t);
	MatrixMaxDifference(compare, vRaw_d, vRaw, totalSize, "vRaw", "swapped v", t);
	MatrixMaxDifference(compare, wRaw_d, wRaw, totalSize, "wRaw", "swapped w", t);
	MatrixMaxDifference(compare, rhoRaw_d, rhoRaw, totalSize, "rhoRaw", "swapped rhoRaw", t);
	MatrixMaxDifference(compare, TRaw_d, TRaw, totalSize, "TRaw", "swapped TRaw", t);
	// Step 3: Divergence of velocity
        //std::cout<<"Calling Divergence()\n";
        computeDivergence_kernel<<<blocksPerGrid,threadsPerBlock>>>(uRaw_d, vRaw_d, wRaw_d, divergenceRaw_d);
        cudaDeviceSynchronize();
        computeDivergence(u, v, w, divergence);
	MatrixMaxDifference(compare, divergenceRaw_d, divergenceRaw, totalSize, "uRaw", "Divergence kernel", t);
	//std::cout<<"Returned from Divergence()\n";
        // Step 4: Iterative solver
        //std::cout<<"Calling CG()\n";
        solvePressureCG(pRaw_d, divergenceRaw_d);
        cudaDeviceSynchronize();
        solvePressureCG_CPU(p, divergence);
	MatrixMaxDifference(compare, pRaw_d, pRaw, totalSize,"CG", "CG kernel", t);
	//std::cout<<"Returned from CG()\n";
        // Step 5: Velocity correction
        //std::cout<<"Velocity correction initiated()\n";
        velocityCorrection_kernel<<<blocksPerGrid,threadsPerBlock>>>(uRaw_d, vRaw_d, wRaw_d, pRaw_d);
        cudaDeviceSynchronize();        
	velocityCorrection(u,v,w,p);
	MatrixMaxDifference(compare, uRaw_d, uRaw, totalSize, "uRaw", "correction kernel", t);
	MatrixMaxDifference(compare, vRaw_d, vRaw, totalSize, "vRaw", "correction kernel", t);
	MatrixMaxDifference(compare, wRaw_d, wRaw, totalSize, "wRaw", "correction kernel", t);
	//std::cout<<"Velocity Correction done\n";
        // Step 6: Boundary Condition
        //std::cout<<"Calling boundary()\n";
        applyBoundaryConditions(uRaw_d,vRaw_d,wRaw_d);
        cudaDeviceSynchronize();        
	applyBoundaryConditions_CPU(u,v,w);
	MatrixMaxDifference(compare, uRaw_d, uRaw, totalSize, "uRaw", "boundary kernel", t);
	MatrixMaxDifference(compare, vRaw_d, vRaw, totalSize, "vRaw", "boundary kernel", t);
	MatrixMaxDifference(compare, wRaw_d, wRaw, totalSize, "wRaw", "boundary kernel", t);
	//std::cout<<"Returned from boundary()\n";
        cudaMemcpy(rhoRaw_for_GPU, rhoRaw_d, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (t % 10 == 0){
            writetoCSV(rho_for_GPU, "GPUdensity_frame_" + std::to_string(t) + ".csv","density");
            writetoCSV(rho, "CPUdensity_frame_" + std::to_string(t) + ".csv","density");
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
