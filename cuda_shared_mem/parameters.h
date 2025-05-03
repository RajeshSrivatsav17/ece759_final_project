#pragma once

#define XDIM 64
#define YDIM 64
#define ZDIM 64

// Constants for buoyancy and temperature
constexpr double alpha = 0.05; // Buoyancy coefficient (0.05 – 0.1)
constexpr double beta = 0.1;   // Temperature lift coefficient
constexpr double T_ambient = 0.0; // Ambient temperature

// Grid spacing
const double dx = 1.0 / (XDIM - 1); // Grid spacing in x-axis
const double dy = 1.0 / (YDIM - 1); // Grid spacing in y-axis
const double dz = 1.0 / (ZDIM - 1); // Grid spacing in z-axis

// Time increment
const double dt = 0.5;

// Conjugate Gradient solver parameters
const int cg_max_iterations = 100;  // Maximum number of iterations
const float cg_tolerance = 1e-5f;   // Convergence criterion

// Simulation parameters
#ifndef TOTAL_STEPS
#define TOTAL_STEPS 500
#endif

// Block size for shared memory
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8 // Adjusted for shared memory optimization
#endif

#ifndef RESULT
#define RESULT 1 // Enable result output
#endif