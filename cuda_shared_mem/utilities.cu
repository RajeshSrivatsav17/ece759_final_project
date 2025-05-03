#include "parameters.h"

__host__ void Clear(float (&arr)[XDIM][YDIM][ZDIM]) {
    for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
            for (int k = 0; k < ZDIM; ++k)
                arr[i][j][k] = 0.0f;
}

// __host__ void InitializeProblem(float (&rho)[XDIM][YDIM][ZDIM], float (&T)[XDIM][YDIM][ZDIM], float (&v)[XDIM][YDIM][ZDIM]) {
//     for (int i = 0; i < XDIM; ++i)
//         for (int j = 0; j < YDIM; ++j)
//             for (int k = 0; k < ZDIM; ++k) {
//                 rho[i][j][k] = (i > 50 && i < 70 && j > 50 && j < 70) ? 1.0f : 0.0f;
//                 T[i][j][k] = 0.0f;
//                 v[i][j][k] = 0.0f;
//             }
// }

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
    for(int k = Z_initial_val; k < Z_limit; ++k)
        for(int j = YDIM/2-YDIM/4; j < YDIM/2+YDIM/4; j++)
        for(int i = XDIM/2-YDIM/4; i < XDIM/2+YDIM/4; i++){ //6% to 12% Near the bottom of the domain
            x[i][j][k] = 1.; //Density
            y[i][j][k] = 300.; //Temperature 
            z[i][j][k] = 2.0f;
        }
            
}
