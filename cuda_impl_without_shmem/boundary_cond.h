#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H
#include "parameters.h"

void applyBoundaryConditions(float (&u)[XDIM][YDIM][ZDIM],
                              float (&v)[XDIM][YDIM][ZDIM],
                              float (&w)[XDIM][YDIM][ZDIM]) {
    #pragma omp parallel for
    for (int j = 0; j < YDIM; ++j)
    for (int k = 0; k < ZDIM; ++k) {
        u[0][j][k] = 0; u[XDIM-1][j][k] = 0; // Applies zero at X boundary velocity
    }
    #pragma omp parallel for
    for (int i = 0; i < XDIM; ++i)
    for (int k = 0; k < ZDIM; ++k) {
        v[i][0][k] = 0; v[i][YDIM-1][k] = 0; // Applies zero at Y boundary velocity
    }
    #pragma omp parallel for
    for (int i = 0; i < XDIM; ++i)
    for (int j = 0; j < YDIM; ++j) {
        w[i][j][0] = 0; w[i][j][ZDIM-1] = 0; // Applies zero at Z boundary velocity
    }
}

#endif
