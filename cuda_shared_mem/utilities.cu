#include "parameters.h"

__host__ void Clear(float (&arr)[XDIM][YDIM][ZDIM]) {
    for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
            for (int k = 0; k < ZDIM; ++k)
                arr[i][j][k] = 0.0f;
}

__host__ void InitializeProblem(float (&rho)[XDIM][YDIM][ZDIM], float (&T)[XDIM][YDIM][ZDIM], float (&v)[XDIM][YDIM][ZDIM]) {
    for (int i = 0; i < XDIM; ++i)
        for (int j = 0; j < YDIM; ++j)
            for (int k = 0; k < ZDIM; ++k) {
                rho[i][j][k] = (i > 50 && i < 70 && j > 50 && j < 70) ? 1.0f : 0.0f;
                T[i][j][k] = 0.0f;
                v[i][j][k] = 0.0f;
            }
}