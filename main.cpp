#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "parameters.h"
#include "Utilities.h"
#include "buoyantforce.h"

int main(){
    using array_t = float (&) [XDIM][YDIM][ZDIM];
    float *uRaw = new float [XDIM*YDIM*ZDIM]; //Velocity in x direction
    float *vRaw = new float [XDIM*YDIM*ZDIM]; //Velocity in y direction
    float *wRaw = new float [XDIM*YDIM*ZDIM]; //Velocity in z direction
    float *rhoRaw = new float [XDIM*YDIM*ZDIM]; //Density
    float *TRaw = new float [XDIM*YDIM*ZDIM]; //Temperature
    float *divergenceRaw = new float [XDIM*YDIM*ZDIM]; //Divergence
    float *PRaw = new float [XDIM*YDIM*ZDIM]; //Pressure

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
    array_t P = reinterpret_cast<array_t>(*PRaw);

    Clear(u);Clear(v);Clear(w);Clear(divergence);Clear(P);

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
    buoyantforce(rho,T,v); //applying buoyant force on pressure and temperature of smoke from vertical velocity compoenent
    writetoCSV(v,"velocity.csv");
    delete[] uRaw;
    delete[] vRaw;
    delete[] wRaw;
    delete[] rhoRaw;
    delete[] TRaw;
    delete[] divergenceRaw;
    delete[] PRaw;

    return 0;
}