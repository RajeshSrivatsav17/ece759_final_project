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

void InitializeProblem(float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM],float (&z)[XDIM][YDIM][ZDIM]){

    // Start by zeroing out x and b
    Clear(x);
    Clear(y);
    Clear(z);
    /* Set the initial conditions for the density and temperature
    Sets:
    rho = 1.0 → meaning smoke exists in that region
    T = 300.0 → this region is hotter than surrounding cells (used for buoyancy)
    */
    int Z_initial_val = ceil(0.06*ZDIM);
    int Z_limit = ceil(0.12*ZDIM);
    #pragma omp parallel for
    for(int k = Z_initial_val; k < Z_limit; ++k)
        for(int j = YDIM/2-16; j < YDIM/2+16; j++)
        for(int i = XDIM/2-16; i < XDIM/2+16; i++){ //6% to 12% Near the bottom of the domain
            x[i][j][k] = 1.; //Density
            y[i][j][k] = 300.; //Temperature 
            z[i][j][k] = 2.0f;
        }
            
}


void writetoCSV(float (&x)[XDIM][YDIM][ZDIM], const std::string& filename, const std::string& ftype) {
    std::ofstream file(filename);
    file << "x,y,z,density\n";
    for (int k = 0; k < ZDIM; ++k)
    for (int j = 0; j < YDIM; ++j)
    for (int i = 0; i < XDIM; ++i) {
        file << i << "," << j << "," << k << "," << x[i][j][k] << "\n";
    }

    file.close();
}

float laplacian(float (&p)[XDIM][YDIM][ZDIM], int i, int j, int k) {
    float sum = 0.0f;
    if (i > 0)       sum += p[i-1][j][k];
    if (i < XDIM-1)  sum += p[i+1][j][k];
    if (j > 0)       sum += p[i][j-1][k];
    if (j < YDIM-1)  sum += p[i][j+1][k];
    if (k > 0)       sum += p[i][j][k-1];
    if (k < ZDIM-1)  sum += p[i][j][k+1];
    sum -= 6.0f * p[i][j][k];
    return sum / (dx * dx);
}
