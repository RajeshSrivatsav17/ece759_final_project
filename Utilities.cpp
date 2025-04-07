#include "Utilities.h"
#include "parameters.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

void Clear(float (&x)[XDIM][YDIM][ZDIM])
{
#pragma omp parallel for
    for (int i = 0; i < XDIM; i++)
    for (int j = 0; j < YDIM; j++)
    for (int k = 0; k < ZDIM; k++)
        x[i][j][k] = 0.;
}

void InitializeProblem(float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM]){

    // Start by zeroing out x and b
    Clear(x);
    Clear(y);

    /* Set the initial conditions for the density and temperature
    Sets:
    rho = 1.0 → meaning smoke exists in that region
    T = 300.0 → this region is hotter than surrounding cells (used for buoyancy)
    */
    for(int k = ceil(0.06*ZDIM); k < ceil(0.12*ZDIM); ++k) //6% to 12% Near the bottom of the domain
        for(int j = YDIM/2-16; j < YDIM/2+16; j++)
        for(int i = XDIM/2-16; i < XDIM/2+16; i++){
            x[i][j][k] = 1.; //Density
            y[i][j][k] = 300.; //Temperature 
        }
            
}


void writetoCSV(float (&x)[XDIM][YDIM][ZDIM], const std::string& filename) {
    std::ofstream file(filename);
    file << "x,y,z,density\n";
    for (int k = 0; k < ZDIM; ++k)
    for (int j = 0; j < YDIM; ++j)
    for (int i = 0; i < XDIM; ++i) {
        file << i << "," << j << "," << k << "," << x[i][j][k] << "\n";
    }

    file.close();
}