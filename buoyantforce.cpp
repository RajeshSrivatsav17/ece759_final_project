#include <iostream>
#include "buoyantforce.h"

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