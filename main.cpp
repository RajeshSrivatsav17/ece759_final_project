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

int main(){
    using array_t = float (&) [XDIM][YDIM][ZDIM];
    float *uRaw = new float [XDIM*YDIM*ZDIM]; //Velocity in x direction
    float *vRaw = new float [XDIM*YDIM*ZDIM]; //Velocity in y direction
    float *wRaw = new float [XDIM*YDIM*ZDIM]; //Velocity in z direction
    float *rhoRaw = new float [XDIM*YDIM*ZDIM]; //Density
    float *TRaw = new float [XDIM*YDIM*ZDIM]; //Temperature
    float *divergenceRaw = new float [XDIM*YDIM*ZDIM]; //Divergence
    float *PRaw = new float [XDIM*YDIM*ZDIM]; //Pressure

    float* uRaw_star = new float[XDIM * YDIM * ZDIM];
    float* vRaw_star = new float[XDIM * YDIM * ZDIM];
    float* wRaw_star = new float[XDIM * YDIM * ZDIM];
    float* rhoRaw_next = new float[XDIM * YDIM * ZDIM];
    float* TRaw_next = new float[XDIM * YDIM * ZDIM];


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

    //Velocity//
    array_t u_star = reinterpret_cast<array_t>(*uRaw_star); //Velocity in x direction
    array_t v_star = reinterpret_cast<array_t>(*vRaw_star); //Velocity in y direction
    array_t w_star = reinterpret_cast<array_t>(*wRaw_star); //Velocity in z direction
    //Density//
    array_t rho_star = reinterpret_cast<array_t>(*rhoRaw_next);
    //Temperature//
    array_t T_star = reinterpret_cast<array_t>(*TRaw_next);

    Clear(u);Clear(v);Clear(w);Clear(divergence);Clear(p);
    Clear(u_star);Clear(v_star);Clear(w_star);

    InitializeProblem(rho,T);
    std ::cout << "density = " << std::endl;
    /*for(int i = 0; i < XDIM; i++){
        for(int j = 0; j < YDIM; j++){
            for(int k = 0; k < ZDIM; k++){
                std::cout << rho[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }*/
    std ::cout << "Temperature = " << std::endl;
    /*
    for(int i = 0; i < XDIM; i++){
        for(int j = 0; j < YDIM; j++){
            for(int k = 0; k < ZDIM; k++){
                std::cout << T[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }*/
    writetoCSV(rho,"density.csv");
    writetoCSV(T,"Temperature.csv");
    // Step 1
    buoyantforce(rho,T,v); //applying buoyant force on pressure and temperature of smoke from vertical velocity compoenent

    // Step 2: Advect velocity (u*, v*, w*)
    semi_lagrangian_advection(u_star, u, u, v, w, dt);
    semi_lagrangian_advection(v_star, v, u, v, w, dt);
    semi_lagrangian_advection(w_star, w, u, v, w, dt);

    // Step 2: Advect smoke density and temperature
    semi_lagrangian_advection(rho_star, rho, u, v, w, dt);
    semi_lagrangian_advection(T_star, T, u, v, w, dt);

    // Swap buffers for next timestep
    std::swap(u, u_star);
    std::swap(v, v_star);
    std::swap(w, w_star);
    std::swap(rho, rho_star);
    std::swap(T, T_star);

    computeDivergence(u, v, w, divergence);
    solvePressureCG(p, divergence);

    for (int i = 1; i < XDIM-1; ++i)
    for (int j = 1; j < YDIM-1; ++j)
    for (int k = 1; k < ZDIM-1; ++k){
        u[i][j][k] -= (p[i+1][j][k] - p[i-1][j][k]) / (2.0f * dx);
        v[i][j][k] -= (p[i][j+1][k] - p[i][j-1][k]) / (2.0f * dx);
        w[i][j][k] -= (p[i][j][k+1] - p[i][j][k-1]) / (2.0f * dx);
    }

    writetoCSV(u,"velocity_u.csv");
    writetoCSV(v,"velocity_v.csv");
    writetoCSV(w,"velocity_w.csv");
    
    Clear(u_star);Clear(v_star);Clear(w_star);
    Clear(rho_star); Clear(T_star);

    delete[] uRaw;
    delete[] vRaw;
    delete[] wRaw;
    delete[] rhoRaw;
    delete[] TRaw;
    delete[] divergenceRaw;
    delete[] PRaw;

    return 0;
}