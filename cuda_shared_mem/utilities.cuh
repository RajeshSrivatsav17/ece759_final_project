#ifndef UTILITIES_H
#define UTILITIES_H

#include "parameters.h"

// Function to clear a 3D array
inline __host__ void Clear(float (&arr)[XDIM][YDIM][ZDIM]);

// Function to initialize the problem with density, temperature, and velocity
inline __host__ void InitializeProblem(float (&rho)[XDIM][YDIM][ZDIM], float (&T)[XDIM][YDIM][ZDIM], float (&v)[XDIM][YDIM][ZDIM]);

#endif // UTILITIES_H